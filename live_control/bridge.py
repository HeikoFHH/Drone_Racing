"""Loopback HTTP -> single-owner Tello UDP bridge. Default backend is a demo.

No automatic takeoff, arbitrary SDK commands or public listener.
Real mode requires the operator's computer to be on the selected drone's Wi-Fi.
"""
from __future__ import annotations

import argparse
import io
import ipaddress
import json
import math
from pathlib import Path
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ZERO=(0,0,0,0)
DIRECTIONS={'stop':ZERO,'forward':(0,20,0,0),'back':(0,-20,0,0),
            'left':(-20,0,0,0),'right':(20,0,0,0),'up':(0,0,20,0),'down':(0,0,-20,0)}


class ControlError(ValueError):
    pass


class DemoTransport:
    real=False
    def __init__(self):
        self.sent=[];self.replies=[];self.airborne=False;self.p=[0.,.5,0.];self.v=[0.,0.,0.]
        self.jpeg=None;self.frame_at=None;self.video_error=None
    def send(self,text,now):
        self.sent.append(text);self.sent=self.sent[-200:]
        if text=='takeoff':self.airborne=True;self.p[2]=1.
        if text=='land':self.airborne=False;self.p[2]=0.
        if text.startswith('rc ') and self.airborne:
            r,f,u,_=map(int,text.split()[1:]);self.v=[f*.008,-r*.008,u*.006]
            self.p[0]+=f*.0004;self.p[1]-=r*.0004;self.p[2]=max(.05,self.p[2]+u*.0003)
        if not text.startswith('rc '):self.replies.append('ok')
    def poll(self,now):
        replies=self.replies[:];self.replies.clear();self.frame_at=now
        return {'bat':80.,'h':self.p[2]*100,'yaw':0.,'vgx':0.,'vgy':0.,'vgz':0.},replies
    def image(self,now):
        svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540"><rect width="960" height="540" fill="#121e2a"/><path d="M0 420H960 M480 120V540 M0 540L480 280 960 540" stroke="#415367" fill="none"/><circle cx="{480+self.p[1]*60:.1f}" cy="{300+self.p[0]*20:.1f}" r="40" fill="none" stroke="#77c7ea" stroke-width="4"/><g fill="#edf6ff" font-family="sans-serif"><text x="32" y="54" font-size="26">SIMULATED CAMERA — no drone connected</text><text x="32" y="500" font-size="22">X {self.p[0]:.2f} m · Y {self.p[1]:.2f} m · height {self.p[2]:.2f} m</text></g></svg>'''
        return 'image/svg+xml',svg.encode()
    def close(self):pass


class TelloTransport:
    real=True
    def __init__(self,interface=None,bind_ip='0.0.0.0'):
        # Import before claiming any ports. These are only needed in real mode.
        import av
        from PIL import Image  # noqa: F401
        self.av=av;self.stop=threading.Event();self.sockets=[]
        self.jpeg=None;self.frame_at=None;self.video_error=None;self.frame_packet=None
        try:
            self.command=self._socket(8889,interface,bind_ip)
            self.state=self._socket(8890,interface,bind_ip)
            self.video=self._socket(11111,interface,bind_ip)
            self.command.setblocking(False);self.state.setblocking(False);self.video.settimeout(.2)
        except Exception:
            self.close();raise
        threading.Thread(target=self._decode,daemon=True,name='tello-video').start()
    def _socket(self,port,interface,bind_ip):
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);self.sockets.append(s)
        # Deliberately no SO_REUSEPORT: another runner must not share this link.
        if interface:
            if not hasattr(socket,'SO_BINDTODEVICE'):raise ValueError('--interface is Linux-only; use --bind-ip on Mac')
            s.setsockopt(socket.SOL_SOCKET,socket.SO_BINDTODEVICE,interface.encode())
        s.bind((bind_ip,port));return s
    def send(self,text,now):self.command.sendto(text.encode(),('192.168.10.1',8889))
    def poll(self,now):
        latest=None;replies=[]
        # Bound work per tick even when a socket is flooded.
        for sock,is_state in ((self.state,True),(self.command,False)):
            for _ in range(50):
                try:data,source=sock.recvfrom(4096)
                except BlockingIOError:break
                if source[0]!='192.168.10.1':continue
                text=data.decode(errors='replace').strip()
                if not is_state:
                    replies.append(text.lower());continue
                fields={}
                for part in text.split(';'):
                    try:
                        key,value=part.split(':',1);number=float(value)
                        if math.isfinite(number):fields[key]=number
                    except ValueError:continue
                if all(k in fields for k in ('bat','h','yaw')) and 0<=fields['bat']<=100:
                    latest=fields
        return latest,replies
    def _decode(self):
        transport=self
        class Reader(io.RawIOBase):
            def __init__(self):self.remainder=b''
            def readable(self):return True
            def read(self,size=-1):
                if self.remainder:
                    out=self.remainder if size<0 else self.remainder[:size]
                    self.remainder=self.remainder[len(out):];return out
                while not transport.stop.is_set():
                    try:
                        data,source=transport.video.recvfrom(65536)
                        if source[0]=='192.168.10.1':
                            out=data if size<0 else data[:size];self.remainder=data[len(out):];return out
                    except socket.timeout:continue
                    except OSError:return b''
                return b''
        while not self.stop.is_set():
            try:
                with self.av.open(Reader(),format='h264',buffer_size=65536) as container:
                    for frame in container.decode(video=0):
                        if self.stop.is_set():return
                        image=frame.to_image();image.thumbnail((960,720))
                        output=io.BytesIO();image.save(output,format='JPEG',quality=65)
                        self.jpeg=output.getvalue();self.frame_at=time.monotonic();self.video_error=None
                        self.frame_packet=(self.frame_at,frame.to_ndarray(format='bgr24'))
            except Exception as exc:
                self.video_error=type(exc).__name__+': '+str(exc)[:180]
                if self.stop.wait(.5):return
    def image(self,now):
        if self.jpeg is None or self.frame_at is None or now-self.frame_at>1.:
            raise ControlError('No fresh camera frame')
        return 'image/jpeg',self.jpeg
    def close(self):
        self.stop.set()
        for s in self.sockets:s.close()


class Controller:
    """All SDK writes happen in tick(), called by one worker, never HTTP threads."""
    def __init__(self,transport,allow_flight=False,clock=time.monotonic,pose_source=None):
        self.transport=transport;self.allow_flight=allow_flight;self.clock=clock
        self.lock=threading.RLock();self.owner=None;self.sequence=-1;self.heartbeat=None
        self.intent=ZERO;self.intent_at=None;self.rc=ZERO;self.telemetry={};self.telemetry_at=None
        self.stage='connecting';self.pending=None;self.request=None;self.owned_flight=False
        self.land_latched=False;self.land_ack=False;self.land_sent=None;self.land_attempts=0
        self.message='Starting SDK and camera; no automatic takeoff';self.last_tick=None
        self.last_rc=None;self.errors=[];self.last_reply=None
        self.takeoff_at=None
        self.pose_source=pose_source;self.policy_session=None;self.policy_active=False
        self.policy_used=False;self.policy_name=None;self.policy_info={}
    def _age(self,t,now):return None if t is None else max(0,now-t)
    def _fresh(self,now):return self.telemetry_at is not None and now-self.telemetry_at<=.5
    def _camera(self,now):return self.transport.frame_at is not None and now-self.transport.frame_at<=1.
    def _land(self,reason):
        self.intent=ZERO;self.rc=ZERO;self.land_latched=True;self.request=None
        self.policy_active=False
        self.message=reason+' · landing requested, touchdown unconfirmed'
    def accept(self,data,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            action=data.get('action');owner=data.get('session');seq=data.get('seq')
            if action not in ('heartbeat','stop','stop_policy','run_policy','land','move','takeoff'):raise ControlError('Unknown action')
            if not isinstance(owner,str) or not 16<=len(owner)<=100:raise ControlError('Invalid browser session')
            if type(seq) is not int or not 0<=seq<2**53:raise ControlError('Invalid sequence')
            if self.owner is None:
                if action!='heartbeat':raise ControlError('Establish browser heartbeat first')
                self.owner=owner
            if owner!=self.owner:raise ControlError('Another browser owns control; restart bridge to change operator')
            if seq<=self.sequence:raise ControlError('Old or reordered request rejected')
            self.sequence=seq
            if action=='heartbeat':self.heartbeat=now;return
            if action in ('stop','stop_policy'):
                self.policy_active=False;self.intent=ZERO;self.intent_at=now
                self.message='Policy stopped · zero RC requested · operator control';return
            if action=='land':
                if not self.allow_flight:raise ControlError('Camera-only mode')
                if self.owned_flight:self._land('Operator requested land')
                return
            if action not in ('move','takeoff','run_policy'):raise ControlError('Unknown action')
            if not self.allow_flight:raise ControlError('Camera-only mode; flight was not enabled at launch')
            if self.land_latched:raise ControlError('Landing/abort latched; restart only after landing and review')
            if self.heartbeat is None or now-self.heartbeat>.6:raise ControlError('Browser heartbeat is stale')
            if not self._fresh(now) or not self._camera(now):raise ControlError('Fresh telemetry and video required')
            if self.telemetry.get('bat',0)<25:raise ControlError('Battery below prototype 25% threshold')
            if action=='takeoff':
                if self.stage!='ready' or self.owned_flight:raise ControlError('Not ready for takeoff')
                if self.telemetry['h']>10:raise ControlError('Height is inconsistent with ground; verify drone state')
                self.request='takeoff';self.stage='takeoff_queued';self.owned_flight=True;self.takeoff_at=now
                self.message='Takeoff queued by operator';return
            if action=='run_policy':
                if self.stage!='airborne':raise ControlError('Take off explicitly before running policy')
                if self.pose_source is None:raise ControlError('Mapped pose unavailable; launch with --policy-setup')
                if self.policy_used:raise ControlError('One policy run per flight session; land and restart to run again')
                from .policies import PolicySession
                name=data.get('policy','baseline')
                try:
                    session=PolicySession(name,self.pose_source,self.transport)
                    ready,reason=session.ready(now,self.telemetry,self.telemetry_at)
                except Exception as exc:raise ControlError('Policy setup: '+str(exc)) from exc
                if not ready:raise ControlError('Policy not ready: '+reason)
                self.policy_session=session;self.policy_name=name;self.policy_active=True;self.policy_used=True
                self.intent=ZERO;self.message='Policy running: '+name;return
            direction=data.get('direction')
            if not isinstance(direction,str) or direction not in DIRECTIONS:raise ControlError('Invalid movement direction')
            if self.stage!='airborne':raise ControlError('Movement requires acknowledged takeoff')
            self.policy_active=False
            self.intent=DIRECTIONS[direction];self.intent_at=now
            self.message='Manual takeover · policy stopped'
    def tick(self,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            state,replies=self.transport.poll(now)
            if state is not None:self.telemetry=state;self.telemetry_at=now
            for reply in replies:
                self.last_reply=reply
                if self.land_sent is not None:
                    if reply=='ok':self.land_ack=True;self.stage='landing_acknowledged'
                    continue
                if self.pending:
                    command,_=self.pending;self.pending=None
                    if reply!='ok':
                        self.message=command+' returned '+reply
                        if command=='takeoff':self._land('Takeoff response uncertain')
                        else:self.stage='error'
                    elif command=='command':self.request='streamon';self.stage='starting_video'
                    elif command=='streamon':self.stage='ready';self.message='Connected · waiting for operator'
                    elif command=='takeoff':self.stage='airborne';self.message='Takeoff acknowledged · manual control'
            if self.owned_flight and not self.land_latched:
                reason=None
                if self.last_tick is not None and now-self.last_tick>.25:reason='Control worker deadline missed'
                elif not self._fresh(now):reason='Telemetry lost'
                elif not self._camera(now):reason='Camera stale'
                elif self.heartbeat is None or now-self.heartbeat>1.:reason='Browser connection lost'
                elif self.telemetry.get('bat',0)<25:reason='Low battery'
                elif self.takeoff_at is not None and now-self.takeoff_at>=90.:reason='90 second flight-session limit'
                if reason:self._land(reason)
            self.last_tick=now
            if self.land_latched:
                if self.land_sent is None:
                    self.transport.send('rc 0 0 0 0',now);self.rc=ZERO
                if not self.land_ack and self.land_attempts<3 and (self.land_sent is None or now-self.land_sent>=2.):
                    self.transport.send('land',now);self.land_sent=now;self.land_attempts+=1;self.stage='landing_requested';self.pending=None
                # No endless zero-RC/keepalive after landing request. No auto resume.
                return
            if self.pending:
                command,sent=self.pending
                if now-sent>(15. if command=='takeoff' else 4.):
                    self.pending=None
                    if command=='takeoff':self._land('Takeoff acknowledgment timed out')
                    else:self.stage='error';self.message=command+' acknowledgment timed out; restart bridge'
                return
            command=self.request
            if command is None and self.stage=='connecting':command='command'
            if command:
                self.request=None;self.transport.send(command,now);self.pending=(command,now)
                if command=='takeoff':self.stage='taking_off'
                return
            if self.stage=='airborne':
                if self.policy_active:
                    try:
                        result=self.policy_session.step(now,self.telemetry,self.telemetry_at)
                        self.policy_info=self.policy_session.last_info
                        self.rc=result.rc
                        if result.request_land:
                            self._land('Policy: '+result.reason)
                            self.transport.send('rc 0 0 0 0',now);return
                    except Exception as exc:
                        self._land('Policy failure: '+str(exc)[:140]);self.transport.send('rc 0 0 0 0',now);return
                else:self.rc=self.intent if self.intent_at is not None and now-self.intent_at<=.3 else ZERO
                self.transport.send('rc '+' '.join(map(str,self.rc)),now);self.last_rc=now
    def status(self,now=None):
        now=self.clock() if now is None else now
        with self.lock:
            return {'backend':'real' if self.transport.real else 'demo','flight_enabled':self.allow_flight,
                    'stage':self.stage,'message':self.message,'rc':self.rc,'telemetry':self.telemetry,
                    'telemetry_age_s':self._age(self.telemetry_at,now),'video_age_s':self._age(self.transport.frame_at,now),
                    'video_error':self.transport.video_error,'link_fresh':self._fresh(now),'video_fresh':self._camera(now),
                    'land_requested':self.land_latched,'land_acknowledged':self.land_ack,'touchdown_confirmed':False,
                    'last_sdk_reply':self.last_reply,'policy_available':self.pose_source is not None,
                    'policy_active':self.policy_active,'policy_name':self.policy_name,'policy_used':self.policy_used,
                    'policy_info':self.policy_info,'pose_error':self.pose_source.error if self.pose_source else 'No mapped positioner configured'}


def make_server(controller,port=8780):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def _allowed(self,post=False):
            host=self.headers.get('Host')
            if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
                return False
            if self.headers.get('Sec-Fetch-Site')=='cross-site':return False
            if post:
                origin=self.headers.get('Origin')
                return (origin=='http://'+host and secrets.compare_digest(self.headers.get('X-Bridge-Token',''),token))
            return True
        def respond(self,code,kind,body):
            self.send_response(code);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            if not self._allowed():self.respond(403,'text/plain',b'Loopback origin only');return
            route=self.path.split('?',1)[0];folder=Path(__file__).parent
            if route=='/':self.respond(200,'text/html; charset=utf-8',(folder/'index.html').read_bytes().replace(b'__TOKEN__',token.encode()))
            elif route in ('/app.js','/style.css'):
                self.respond(200,'text/javascript' if route.endswith('js') else 'text/css',(folder/route[1:]).read_bytes())
            elif route=='/api/status':self.respond(200,'application/json',json.dumps(controller.status()).encode())
            elif route=='/camera':
                try:kind,body=controller.transport.image(time.monotonic());self.respond(200,kind,body)
                except ControlError as exc:self.respond(503,'text/plain',str(exc).encode())
            else:self.respond(404,'text/plain',b'Not found')
        def do_POST(self):
            if not self._allowed(post=True):self.respond(403,'text/plain',b'Invalid origin or bridge token');return
            if self.path!='/api/control':self.respond(404,'text/plain',b'Not found');return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=2048 or self.headers.get('Content-Type')!='application/json':raise ControlError('Invalid request body')
                self.connection.settimeout(1.)
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict):raise ControlError('Expected a JSON object')
                controller.accept(data)
            except (ValueError,TypeError,TimeoutError) as exc:
                self.respond(409,'application/json',json.dumps({'error':str(exc)}).encode());return
            self.respond(200,'application/json',b'{"accepted":true}')
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real',action='store_true',help='connect to one real Tello; default is simulated')
    p.add_argument('--allow-flight',action='store_true',help='enable explicit takeoff and manual controls in real mode')
    p.add_argument('--port',type=int,default=8780)
    p.add_argument('--interface',help='Linux Wi-Fi interface for the selected Tello')
    p.add_argument('--bind-ip',default='0.0.0.0',help='local IP on drone Wi-Fi; never a remote/cloud IP')
    p.add_argument('--policy-setup',type=Path,help='verified camera/map/course/transform JSON for real policy runs')
    args=p.parse_args(argv)
    if not 1<=args.port<=65535:p.error('invalid HTTP port')
    try:ipaddress.IPv4Address(args.bind_ip)
    except ValueError:p.error('invalid local IPv4 address')
    try:transport=TelloTransport(args.interface,args.bind_ip) if args.real else DemoTransport()
    except (ImportError,OSError,ValueError) as exc:p.exit(2,'Bridge setup failed: '+str(exc)+'\n')
    from .policies import DemoPose,MappedPose
    try:source=MappedPose(transport,args.policy_setup) if args.real and args.policy_setup else DemoPose(transport) if not args.real else None
    except Exception as exc:transport.close();p.exit(2,'Policy setup failed: '+str(exc)+'\n')
    if source:
        try:
            from race.race_policy import RacePolicy  # preload outside the flight/control tick
        except ImportError as exc:
            source.close();transport.close();p.exit(2,'Install live_control/requirements.txt and -e cal_and_map: '+str(exc)+'\n')
    control=Controller(transport,allow_flight=args.allow_flight or not args.real,pose_source=source)
    try:server=make_server(control,args.port)
    except OSError:transport.close();raise
    stop=threading.Event()
    def worker():
        while not stop.is_set():
            started=time.monotonic()
            try:control.tick(started)
            except Exception as exc:
                with control.lock:
                    control.message='Transport failure: '+str(exc)[:160]
                    if control.owned_flight:control._land('Transport failure')
                    else:control.stage='error'
            stop.wait(max(0,.05-(time.monotonic()-started)))
    threading.Thread(target=worker,daemon=True,name='sole-sdk-writer').start()
    print(('REAL TELLO · '+('FLIGHT ENABLED' if args.allow_flight else 'CAMERA ONLY')) if args.real else 'DEMO · no drone connected',flush=True)
    print(f'Open http://127.0.0.1:{args.port}/ on THIS computer',flush=True)
    try:server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:pass
    finally:
        # Worker remains the only SDK writer, including shutdown landing requests.
        with control.lock:
            if control.owned_flight and not control.land_latched:control._land('Bridge shutdown')
        if control.owned_flight:
            deadline=time.monotonic()+5.
            while time.monotonic()<deadline and not control.land_ack:time.sleep(.05)
        stop.set();server.server_close();transport.close()
        if source:source.close()
        if control.owned_flight:print('Bridge closed; landing request sent where possible. Touchdown NOT verified.',flush=True)


if __name__=='__main__':main()
