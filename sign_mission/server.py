"""Web control interface: drone video, sign detected + distance, Start (take off and fly) and Stop (stop and land).

    python -m sign_mission.server --sim                  # simulated Tello, open http://127.0.0.1:8765
    python -m sign_mission.server --live --iface wlx...  # the real drone (Linux laptop on the Tello's Wi-Fi)

Standard library only (plus OpenCV/numpy for the picture). Binds to 127.0.0.1 unless told otherwise; a live drone on
a non-local address requires --token. The page needs no internet.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .drone import RealClock
from .perception import annotate
from .policies import POLICIES, PolicyCfg
from .runner import MissionRunner
from .sim import Scenario, SimDrone
from .spec import SIGN_WIDTH_IN, SignSpec

WEB = Path(__file__).resolve().parent / "web"


def placeholder(text: str, w: int = 960, h: int = 720) -> np.ndarray:
    img = np.full((h, w, 3), 40, np.uint8)
    cv2.putText(img, text, (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
    return img


class App:
    def __init__(self, drone, runner: MissionRunner, token: str | None = None):
        self.drone, self.runner, self.token = drone, runner, token
        self.sim = not drone.live

    def frame_jpeg(self) -> bytes:
        f = self.drone.preview()
        img = placeholder("no video") if f is None else f
        with self.runner.lock:
            dets, est, phase, state = list(self.runner.dets), self.runner.est, self.runner.phase, self.runner.info.get("state", "")
        banner = f"{'SIM' if self.sim else 'LIVE'}  {phase}" + (f"  {state}" if state else "")
        ok, buf = cv2.imencode(".jpg", annotate(img, dets, est, banner), [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buf.tobytes()

    def status(self) -> dict:
        s = self.runner.status()
        s["policies"] = [{"name": n, "title": c.title} for n, c in POLICIES.items()]
        s["sign_width_in"] = round(self.runner.sign.width_m / 0.0254, 2)
        s["shadow"] = self.runner.cfg.shadow
        if self.sim:
            sc = self.drone.sc
            s["sim"] = {"distance_m": sc.distance_m, "lateral_m": sc.lateral_m, "heading_error_deg": sc.heading_error_deg,
                        "wind_mps": list(sc.wind_mps), "speed": getattr(self.drone.clock, "speed", 1.0)}
            s["truth"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.drone.truth().items()}
        return s


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):            # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode())

        def _authorized(self) -> bool:
            if not app.token:
                return True
            q = parse_qs(urlparse(self.path).query)
            return self.headers.get("X-Token") == app.token or q.get("token", [None])[0] == app.token

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(app.status())
            elif path == "/video.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        jpg = app.frame_jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            elif path == "/frame.jpg":
                self._send(200, app.frame_jpeg(), "image/jpeg")
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if not self._authorized():
                return self._json({"ok": False, "message": "token required"}, 403)
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "message": "bad JSON"}, 400)
            if path == "/api/start":
                ok, msg = app.runner.start(str(body.get("policy", "approach")), confirmed=bool(body.get("confirmed")))
                self._json({"ok": ok, "message": msg}, 200 if ok else 409)
            elif path == "/api/stop":
                self._json({"ok": True, "message": app.runner.stop()})
            elif path == "/api/sim" and app.sim:
                if app.runner.busy:
                    return self._json({"ok": False, "message": "a mission is running"}, 409)
                sc = app.drone.sc
                kw = {k: float(body[k]) for k in ("distance_m", "lateral_m", "heading_error_deg") if k in body}
                if "wind_mps" in body:
                    kw["wind_mps"] = tuple(float(v) for v in body["wind_mps"])
                from dataclasses import replace
                app.drone.reset(replace(sc, **kw))
                if "speed" in body and hasattr(app.drone.clock, "speed"):
                    app.drone.clock.speed = max(0.25, min(20.0, float(body["speed"])))
                self._json({"ok": True, "message": "scenario applied"})
            else:
                self._json({"ok": False, "message": "not found"}, 404)

    return H


def serve(drone, host: str = "127.0.0.1", port: int = 8765, token: str | None = None, sign: SignSpec | None = None,
          policy_cfg: PolicyCfg | None = None, block: bool = True, shadow: bool = False):
    from .runner import RunnerCfg
    runner = MissionRunner(drone, cfg=RunnerCfg(shadow=shadow), sign=sign, policy_cfg=policy_cfg)
    runner.start_monitor()
    app = App(drone, runner, token)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    app.httpd = httpd
    if block:
        print(f"{'SIMULATION' if app.sim else 'LIVE DRONE'} control interface: http://{host}:{port}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down (landing if flying)")
        finally:
            runner.stop()
            if runner.thread:
                runner.thread.join(timeout=30)
            runner.close()
            drone.close()
    else:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sim", action="store_true", help="simulated Tello (default)")
    g.add_argument("--live", action="store_true", help="the real Tello")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", help="required for POSTs when set (mandatory for --live on a non-local --host)")
    ap.add_argument("--sign-width-in", type=float, default=SIGN_WIDTH_IN, help="edge of the sign's black square, inches")
    ap.add_argument("--tag-id", type=int, help="only this AprilTag id is the sign (default: the largest tag in view)")
    ap.add_argument("--stop-m", type=float, default=0.5, help="stop this far in front of the sign (camera to sign plane)")
    ap.add_argument("--iface", help="live: pin sockets to this network interface (Linux)")
    ap.add_argument("--trust-velocity", action="store_true", help="live: use state-packet vgx/vgy as body-frame velocity (verify units first)")
    ap.add_argument("--vel-scale", type=float, default=0.01, help="live: metres/second per unit of vgx/vgy")
    ap.add_argument("--shadow", action="store_true", help="live: run the policy on live video but never take off or send sticks (carry the drone by hand)")
    ap.add_argument("--sim-speed", type=float, default=1.0, help="sim: run faster than real time")
    ap.add_argument("--sim-distance", type=float, default=8.0)
    ap.add_argument("--perception", choices=["render", "analytic"], default="render", help="sim: render images and run the real detector")
    a = ap.parse_args(argv)
    sign = SignSpec.from_inches(a.sign_width_in, tag_id=a.tag_id)
    pcfg = PolicyCfg(stop_m=a.stop_m)
    if a.live:
        if a.host not in ("127.0.0.1", "localhost") and not a.token:
            ap.error("--live on a non-local --host requires --token")
        from .tello import TelloDrone
        drone = TelloDrone(iface=a.iface, sign=sign, trust_velocity=a.trust_velocity, vel_scale=a.vel_scale)
        print("connecting to the Tello ...")
        if not drone.connect():
            raise SystemExit("no answer from the drone: join its Wi-Fi (TELLO-xxxxxx) and try again")
        if not drone.start_video():
            raise SystemExit("no video from the drone (needs OpenCV with FFmpeg; allow incoming UDP 11111)")
    else:
        drone = SimDrone(Scenario(distance_m=a.sim_distance, perception=a.perception), clock=RealClock(a.sim_speed), sign=sign)
    serve(drone, a.host, a.port, a.token, sign, pcfg, shadow=a.shadow)


if __name__ == "__main__":
    main()
