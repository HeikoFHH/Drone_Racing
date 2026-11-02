import http.client
import json
import re
import threading
import unittest

from .bridge import Controller,ControlError,DemoTransport,DIRECTIONS,ZERO,make_server


class FaultTransport(DemoTransport):
    def __init__(self):super().__init__();self.state_available=True;self.video_available=True;self.reply_available=True
    def poll(self,now):
        frame_at=self.frame_at;state,replies=super().poll(now)
        if not self.video_available:self.frame_at=frame_at
        return state if self.state_available else None,replies if self.reply_available else []


class BridgeTests(unittest.TestCase):
    def boot(self,allow=True,transport=None):
        self.transport=transport or FaultTransport();self.c=Controller(self.transport,allow)
        self.seq=0;self.owner='browser-session-123456789'
        for now in (0,.05,.1):self.c.tick(now)
        self.assertEqual(self.c.stage,'ready');self.send('heartbeat',now=.1)
    def send(self,action,now=.1,**extra):
        data={'session':self.owner,'seq':self.seq,'action':action,**extra};self.seq+=1
        self.c.accept(data,now)
    def airborne(self):
        self.boot();self.send('takeoff');self.c.tick(.15);self.c.tick(.2)
        self.assertEqual(self.c.stage,'airborne')

    def test_no_automatic_takeoff_or_ground_rc(self):
        self.boot();self.assertEqual(self.transport.sent,['command','streamon'])
    def test_camera_only_rejects_motion_and_takeoff(self):
        self.boot(allow=False)
        for action in ('takeoff','move','land'):
            with self.assertRaises(ControlError):self.send(action,direction='forward')
        self.assertNotIn('takeoff',self.transport.sent)
    def test_six_directions_and_stop_have_correct_sdk_signs(self):
        self.airborne()
        for i,(direction,rc) in enumerate(DIRECTIONS.items()):
            now=round(.25+i*.05,8);self.send('move',now=now,direction=direction);self.c.tick(now)
            self.assertEqual(self.c.rc,rc)
            self.assertEqual(self.transport.sent[-1],'rc '+' '.join(map(str,rc)))
        self.send('stop',now=.65);self.c.tick(.65);self.assertEqual(self.c.rc,ZERO)
    def test_movement_deadman_does_not_need_a_release_packet(self):
        self.airborne();self.send('move',now=.25,direction='forward')
        for n in range(5,13):self.c.tick(n*.05)
        self.assertEqual(self.c.rc,ZERO);self.assertFalse(self.c.land_latched)
    def test_browser_loss_latches_land_without_continuous_keepalive(self):
        self.airborne()
        for n in range(5,28):self.c.tick(n*.05)
        self.assertTrue(self.c.land_latched);self.assertIn('land',self.transport.sent)
        first=self.transport.sent.index('land')
        self.assertFalse(any(s.startswith('rc ') for s in self.transport.sent[first+1:]))
        self.send('heartbeat',now=1.4)
        with self.assertRaises(ControlError):self.send('move',now=1.4,direction='forward')
        self.assertFalse(self.c.status(1.4)['touchdown_confirmed'])
    def test_video_and_telemetry_loss_each_request_landing(self):
        for fault in ('video_available','state_available'):
            self.airborne();setattr(self.transport,fault,False)
            for n in range(5,30):
                now=n*.05;self.send('heartbeat',now=now);self.c.tick(now)
            self.assertTrue(self.c.land_latched,fault)
    def test_low_battery_and_control_deadline(self):
        self.airborne();self.c.tick(.6)
        self.assertTrue(self.c.land_latched);self.assertIn('deadline',self.c.message)
        self.boot();self.c.telemetry['bat']=20
        with self.assertRaises(ControlError):self.send('takeoff')
    def test_takeoff_timeout_never_retries_takeoff(self):
        self.boot();self.transport.reply_available=False;self.send('takeoff')
        for n in range(3,330):
            now=n*.05;self.send('heartbeat',now=now);self.c.tick(now)
        self.assertEqual(self.transport.sent.count('takeoff'),1);self.assertTrue(self.c.land_latched)
        self.assertIn('land',self.transport.sent)
    def test_explicit_land_is_latched_and_never_claims_touchdown(self):
        self.airborne();self.send('land',now=.25);self.c.tick(.25);self.c.tick(.3)
        state=self.c.status(.3)
        self.assertTrue(state['land_requested']);self.assertTrue(state['land_acknowledged'])
        self.assertFalse(state['touchdown_confirmed'])
        with self.assertRaises(ControlError):self.send('takeoff',now=.3)
    def test_old_packets_and_other_tabs_cannot_override_owner(self):
        self.airborne();self.send('move',now=.25,direction='forward');self.send('stop',now=.26)
        with self.assertRaises(ControlError):
            self.c.accept({'session':self.owner,'seq':self.seq-2,'action':'move','direction':'forward'},.27)
        with self.assertRaises(ControlError):
            self.c.accept({'session':'different-session-12345','seq':999,'action':'heartbeat'},.27)
        self.assertEqual(self.c.intent,ZERO)
    def test_invalid_requests_cannot_claim_control(self):
        c=Controller(DemoTransport(),True)
        for data in ({},{'session':'browser-session-123456789','seq':0,'action':'execute'},
                     {'session':'browser-session-123456789','seq':True,'action':'heartbeat'}):
            with self.assertRaises(ControlError):c.accept(data,0)
            self.assertIsNone(c.owner)
    def test_altitude_not_consistent_with_ground_blocks_takeoff(self):
        self.boot();self.c.telemetry['h']=80
        with self.assertRaises(ControlError):self.send('takeoff')


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(DemoTransport(),True);self.server=make_server(self.c,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.port=self.server.server_port
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,method,path,body=None,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=2)
        conn.request(method,path,body=body,headers=headers or {});response=conn.getresponse()
        result=(response.status,dict(response.getheaders()),response.read());conn.close();return result
    def test_page_status_camera_and_assets(self):
        code,headers,page=self.request('GET','/');self.assertEqual(code,200)
        self.assertNotIn(b'__TOKEN__',page);self.assertIn('frame-ancestors',headers['Content-Security-Policy'])
        for path in ('/api/status','/camera','/app.js','/style.css'):
            self.assertEqual(self.request('GET',path)[0],200,path)
        self.assertIn(b'SIMULATED CAMERA',self.request('GET','/camera')[2])
        self.assertEqual(self.request('GET','/../../etc/passwd')[0],404)
    def test_origin_token_host_and_cross_site_guards(self):
        _,_,page=self.request('GET','/');token=re.search(b'name="bridge-token" content="([^"]+)"',page)[1].decode()
        body=json.dumps({'action':'heartbeat','seq':0,'session':'http-browser-session-12345'})
        origin=f'http://127.0.0.1:{self.port}'
        headers={'Content-Type':'application/json','Origin':origin,'X-Bridge-Token':token}
        self.assertEqual(self.request('POST','/api/control',body,headers)[0],200)
        self.assertEqual(self.request('POST','/api/control',body,{**headers,'Origin':'https://example.com'})[0],403)
        self.assertEqual(self.request('POST','/api/control',body,{**headers,'X-Bridge-Token':'wrong'})[0],403)
        self.assertEqual(self.request('GET','/',headers={'Host':'example.com'})[0],403)
        self.assertEqual(self.request('GET','/',headers={'Sec-Fetch-Site':'cross-site'})[0],403)
        self.assertEqual(self.request('POST','/api/control',body,headers)[0],409)
    def test_oversized_and_malformed_commands_are_rejected(self):
        _,_,page=self.request('GET','/');token=re.search(b'name="bridge-token" content="([^"]+)"',page)[1].decode()
        headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{self.port}','X-Bridge-Token':token}
        for body in ('[1,2]','not json','x'*3000):self.assertEqual(self.request('POST','/api/control',body,headers)[0],409)


if __name__=='__main__':unittest.main()
