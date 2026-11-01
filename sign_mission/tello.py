"""The real Tello behind the Drone interface (live mode).

NOT YET FLOWN. Everything here was written from the Tello SDK 2.0 text protocol and from the repo's own link code
(connect-test/tello_basic.py, core/test_link.py) and is exercised in tests only against a fake drone on localhost.
First real use: props off or tethered, then the staged checks in docs/SIGN_MISSION.md.

  commands   UDP to <drone>:8889, one at a time (a lock), replies "ok" / "error" / a value
  rc sticks  "rc a b c d", fire and forget, only sent while no blocking command is in flight
  telemetry  state packets on UDP :8890 (h in cm, yaw in degrees, vgx/vgy, bat); parsed by a thread
  video      H.264 on UDP :11111 decoded with OpenCV/FFmpeg (as tello_basic.py); a thread detects the sign on the newest frame
Safety: close() always sends "rc 0 0 0 0" and lands a flying drone; the emergency motor-cut command is never used.
"""

from __future__ import annotations

import math
import os
import socket
import sys
import threading
import time

import numpy as np

from .common import SignObs
from .drone import Clock, Drone, FrameResult, RealClock, Telemetry
from .perception import SignDetector
from .spec import CameraModel, SignSpec, load_camera

SO_BINDTODEVICE = 25          # Linux


class TelloDrone(Drone):
    live = True
    name = "DJI Tello"

    def __init__(self, drone_ip: str = "192.168.10.1", iface: str | None = None, sign: SignSpec | None = None,
                 cam: CameraModel | None = None, vel_scale: float = 0.01, drone_cmd_port: int = 8889,
                 local_cmd_port: int = 8889, local_state_port: int = 8890, video_port: int = 11111,
                 detect_hz: float = 15.0, verbose: bool = False, trust_velocity: bool = False):
        self.clock: Clock = RealClock()
        self.addr = (drone_ip, drone_cmd_port)
        self.sign, self.cam = sign or SignSpec(), cam or load_camera()
        self.vel_scale, self.video_port, self.detect_hz, self.verbose = vel_scale, video_port, detect_hz, verbose
        self.trust_velocity = trust_velocity     # vgx/vgy as (forward, right) in units of vel_scale m/s: VERIFY on the drone first
        self.cmd = self._socket(iface, local_cmd_port)
        self.state_sock = self._socket(iface, local_state_port)
        self.lock = threading.Lock()                  # one command in flight; rc waits for it
        self.stop_evt = threading.Event()
        self.state: dict[str, float | str] = {}
        self.state_t = 0.0
        self.flying = False
        self.frame: np.ndarray | None = None
        self.frame_t = 0.0
        self._result: FrameResult | None = None
        self._returned_t = 0.0
        self._detector: SignDetector | None = None
        self._threads: list[threading.Thread] = []
        t = threading.Thread(target=self._state_loop, daemon=True, name="tello-state")
        t.start()
        self._threads.append(t)

    # -- sockets -------------------------------------------------------------------------------
    @staticmethod
    def _socket(iface: str | None, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if iface:
            s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, iface.encode())    # Linux; needs privileges
        s.bind(("0.0.0.0", port))
        return s

    def _drain(self) -> None:
        self.cmd.setblocking(False)
        try:
            while True:
                self.cmd.recvfrom(1024)
        except (BlockingIOError, OSError):
            pass
        finally:
            self.cmd.setblocking(True)

    def _wait(self, timeout: float) -> str | None:
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            self.cmd.settimeout(left)
            try:
                data, src = self.cmd.recvfrom(1024)
            except (socket.timeout, OSError):
                return None
            if src[0] == self.addr[0]:
                return data.decode("utf-8", errors="replace").strip().lower()
        return None

    def send(self, command: str, timeout: float = 3.0, retries: int = 0) -> str | None:
        """One SDK command, blocking until it is answered. Returns the reply (lower case) or None on timeout."""
        with self.lock:
            for _ in range(1 + retries):
                self._drain()
                if self.verbose:
                    print(f"  -> {command}", file=sys.stderr)
                self.cmd.sendto(command.encode("ascii"), self.addr)
                reply = self._wait(timeout)
                if reply is not None:
                    if self.verbose:
                        print(f"  <- {reply}", file=sys.stderr)
                    return reply
        return None

    # -- telemetry -----------------------------------------------------------------------------
    def _state_loop(self) -> None:
        self.state_sock.settimeout(0.5)
        while not self.stop_evt.is_set():
            try:
                data, _ = self.state_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            self.state, self.state_t = self.parse_state(data.decode("ascii", errors="replace")), time.time()

    @staticmethod
    def parse_state(text: str) -> dict:
        vals: dict[str, float | str] = {}
        for item in text.strip().split(";"):
            k, sep, v = item.partition(":")
            if sep:
                try:
                    vals[k] = float(v)
                except ValueError:
                    vals[k] = v
        return vals

    def telemetry(self) -> Telemetry:
        s, age = self.state, time.time() - self.state_t if self.state_t else math.inf
        fresh = age < 1.0
        speed = None
        if fresh and "vgx" in s and "vgy" in s:
            speed = math.hypot(float(s["vgx"]), float(s["vgy"])) * self.vel_scale
        vb = (float(s["vgx"]) * self.vel_scale, float(s["vgy"]) * self.vel_scale) if (
            self.trust_velocity and fresh and "vgx" in s and "vgy" in s) else None
        return Telemetry(t=self.clock.now(), height_m=float(s["h"]) / 100.0 if fresh and "h" in s else None,
                         yaw_deg=float(s["yaw"]) if fresh and "yaw" in s else None, speed_mps=speed, vel_body_mps=vb,
                         battery_pct=float(s["bat"]) if "bat" in s else None, link_age_s=age,
                         flying=self.flying, raw=dict(s))

    # -- lifecycle -----------------------------------------------------------------------------
    def connect(self) -> bool:
        for _ in range(10):
            if self.send("command", timeout=1.5) == "ok":
                t0 = time.time()
                while time.time() - t0 < 3 and not self.state_t:
                    time.sleep(0.1)
                return bool(self.state_t)
            time.sleep(0.4)
        return False

    def start_video(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False
        if self.send("streamon", timeout=3.0, retries=1) != "ok":
            return False
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")
        self._detector = SignDetector(self.sign, self.cam)
        for target, name in ((self._reader, "tello-video"), (self._detect_loop, "tello-detect")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        t0 = time.time()
        while self.frame_t <= t0 and time.time() - t0 < 10:
            time.sleep(0.05)
        return self.frame_t > t0

    def _reader(self) -> None:
        import cv2
        url = f"udp://0.0.0.0:{self.video_port}?overrun_nonfatal=1&fifo_size=5000000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        while not self.stop_evt.is_set():
            if not cap.isOpened():
                time.sleep(0.5)
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                continue
            ok, f = cap.read()
            if ok and f is not None:
                self.frame, self.frame_t = f, time.time()
            else:
                time.sleep(0.01)
        cap.release()

    def _detect_loop(self) -> None:
        last = 0.0
        while not self.stop_evt.is_set():
            if self.frame is None or self.frame_t == last:
                time.sleep(0.005)
                continue
            f, ft = self.frame, self.frame_t
            last = ft
            try:
                dets: list[SignObs] = self._detector.detect(f)
            except Exception:                                           # noqa: BLE001 -- a bad frame must not kill perception
                dets = []
            self._result = FrameResult(self.clock.now(), f, dets)
            time.sleep(max(0.0, 1.0 / self.detect_hz - (time.time() - ft)))

    def poll_frame(self) -> FrameResult | None:
        r = self._result
        if r is None or r.t == self._returned_t:
            return None
        self._returned_t = r.t
        return r

    def preview(self) -> np.ndarray | None:
        return self.frame

    # -- flight commands -----------------------------------------------------------------------
    def takeoff(self) -> bool:
        reply = self.send("takeoff", timeout=15.0)
        if reply != "ok":
            return False
        self.flying = True
        t0 = time.time()
        while time.time() - t0 < 6.0:                                   # wait for the climb to finish
            h = self.telemetry().height_m
            if h is not None and h > 0.4:
                break
            time.sleep(0.1)
        return True

    def land(self) -> bool:
        ok = self.send("land", timeout=10.0) == "ok"
        if ok:
            t0 = time.time()
            while time.time() - t0 < 8.0:
                h = self.telemetry().height_m
                if h is not None and h < 0.12:
                    break
                time.sleep(0.1)
            self.flying = False
        return ok

    def send_rc(self, right: int, forward: int, up: int, yaw: int) -> None:
        if self.lock.acquire(blocking=False):                           # skip a tick while a blocking command runs
            try:
                self.cmd.sendto(f"rc {int(right)} {int(forward)} {int(up)} {int(yaw)}".encode("ascii"), self.addr)
            finally:
                self.lock.release()

    def close(self) -> None:
        try:
            if self.flying:
                self.cmd.sendto(b"rc 0 0 0 0", self.addr)
                for _ in range(3):
                    if self.send("land", timeout=8.0) == "ok":
                        break
                self.flying = False
            self.send("streamoff", timeout=1.0)
        finally:
            self.stop_evt.set()
            for s in (self.cmd, self.state_sock):
                try:
                    s.close()
                except OSError:
                    pass
