#!/usr/bin/env python3
"""Connect to one Tello and send basic SDK commands -- interactively or as a script.

Standard library only. Speaks the Tello SDK directly over UDP (commands -> 192.168.10.1:8889,
telemetry <- :8890), the same way the rest of this repo does.

How it reaches the drone (pick one):

  python tools/tello_basic.py                    # laptop Wi-Fi already joined to TELLO-XXXXXX
  python tools/tello_basic.py --iface wlx6c4c...  # pin to a USB stick (Linux, SO_BINDTODEVICE)
  python tools/tello_basic.py --drone TELLO-1     # look the stick up in config/dongles.json,
                                                 # bring its nmcli profile up, then pin to it

Interactive shell (default) -- type `help` once connected:

  tello> takeoff
  tello> up 50
  tello> cw 90
  tello> forward 100
  tello> land

One-shot sequence, then exit (lands automatically at the end if still flying):

  python tools/tello_basic.py --do "takeoff; up 50; cw 360; land"
  python tools/tello_basic.py --do "battery?; wifi?"          # read-only check, no takeoff
  python tools/tello_basic.py --do "telemetry"                # print telemetry once, no takeoff
  python tools/tello_basic.py --do "watch 10"                 # live telemetry for 10 s
  python tools/tello_basic.py --tag-id 1 --do "detect 15"     # look for AprilTag id 1 (no takeoff)
  python tools/tello_basic.py --tag-family 36h11              # then type `video` at the prompt

Safety:
  * Ctrl-C, `quit`, an exception, or the end of --do all LAND the drone if it is flying.
  * `stop` hovers in place; `emergency` (or `e`) cuts the motors instantly -- it will fall.
  * rc stick values are capped at +/-30 (repo's project limit; change with --rc-limit).
  * A keepalive is sent every 5 s so the drone doesn't auto-land after 15 s idle.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DONGLES = REPO / "config" / "dongles.json"

TELLO_IP = "192.168.10.1"
CMD_PORT = 8889
STATE_PORT = 8890
SO_BINDTODEVICE = 25  # Linux socket option number

# Reply timeouts (seconds). Motion commands only answer once the move is finished.
T_QUERY = 3.0
T_MOVE = 20.0
T_TAKEOFF = 15.0


# --------------------------------------------------------------------------- link layer
class Tello:
    """Minimal Tello SDK client: request/reply on 8889, state listener on 8890."""

    def __init__(self, drone_ip: str = TELLO_IP, iface: str | None = None,
                 local_cmd_port: int = CMD_PORT, state_port: int = STATE_PORT,
                 verbose: bool = False) -> None:
        self.addr = (drone_ip, CMD_PORT)
        self.verbose = verbose
        self.cmd = self._socket(iface, local_cmd_port)
        self.state_sock = self._socket(iface, state_port)
        self.lock = threading.Lock()          # one command in flight at a time
        self.stop_evt = threading.Event()
        self.state: dict[str, float | str] = {}
        self.state_t = 0.0                    # time of last state packet
        self.flying = False
        self.vision: Vision | None = None     # set by main()
        threading.Thread(target=self._state_loop, daemon=True).start()

    @staticmethod
    def _socket(iface: str | None, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if iface:
            try:
                s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, iface.encode())
            except PermissionError:
                sys.exit(f"SO_BINDTODEVICE on {iface} was refused -- run with sudo, or "
                         "drop --iface and join the drone's Wi-Fi on the laptop instead.")
        s.bind(("0.0.0.0", port))
        return s

    # -- request / reply
    def send(self, cmd: str, timeout: float = T_QUERY, retries: int = 0) -> str | None:
        """Send one SDK command, return its reply (lower-cased) or None on timeout."""
        with self.lock:
            for _ in range(1 + retries):
                self._drain()
                if self.verbose:
                    print(f"  -> {cmd}")
                self.cmd.sendto(cmd.encode("ascii"), self.addr)
                reply = self._wait(timeout)
                if reply is not None:
                    if self.verbose:
                        print(f"  <- {reply}")
                    return reply
        return None

    def send_nowait(self, cmd: str) -> None:
        """Fire-and-forget (rc has no reply)."""
        with self.lock:
            self.cmd.sendto(cmd.encode("ascii"), self.addr)

    def _wait(self, timeout: float) -> str | None:
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            self.cmd.settimeout(left)
            try:
                data, src = self.cmd.recvfrom(1024)
            except (socket.timeout, OSError):
                return None
            if src[0] == self.addr[0]:
                return data.decode("utf-8", errors="replace").strip().lower()
        return None

    def _drain(self) -> None:
        self.cmd.setblocking(False)
        try:
            while True:
                self.cmd.recvfrom(1024)
        except (BlockingIOError, OSError):
            pass
        finally:
            self.cmd.setblocking(True)

    # -- telemetry
    def _state_loop(self) -> None:
        self.state_sock.settimeout(0.5)
        while not self.stop_evt.is_set():
            try:
                data, _ = self.state_sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return
            vals: dict[str, float | str] = {}
            for item in data.decode("ascii", errors="replace").strip().split(";"):
                k, sep, v = item.partition(":")
                if sep:
                    try:
                        vals[k] = float(v)
                    except ValueError:
                        vals[k] = v
            self.state, self.state_t = vals, time.time()

    def link_up(self) -> bool:
        return time.time() - self.state_t < 1.5

    # -- lifecycle
    def connect(self, attempts: int = 10) -> bool:
        for _ in range(attempts):
            if self.send("command", timeout=1.5) == "ok":
                return True
            time.sleep(0.4)
        return False

    def start_keepalive(self, every: float = 5.0) -> None:
        def loop() -> None:
            while not self.stop_evt.wait(every):
                if not self.lock.locked():      # don't queue behind a long move
                    self.send("battery?", timeout=1.0)
        threading.Thread(target=loop, daemon=True).start()

    def safe_land(self) -> None:
        if not self.flying:
            return
        print("landing...")
        self.send_nowait("rc 0 0 0 0")
        for _ in range(5):
            if self.send("land", timeout=8.0) == "ok":
                break
            time.sleep(0.3)
        self.flying = False

    def close(self) -> None:
        try:
            self.safe_land()
        finally:
            if self.vision:
                self.vision.stop()
            self.stop_evt.set()
            self.cmd.close()
            self.state_sock.close()


# --------------------------------------------------------------------------- telemetry
# (key, label, unit) grouped for display. Keys the drone sends that aren't listed here
# are still shown, under "Other".
TELEMETRY_GROUPS = [
    ("Power / temperature", [("bat", "Battery", "%"), ("templ", "Temp low", "C"),
                             ("temph", "Temp high", "C"), ("time", "Motor time", "s")]),
    ("Altitude",            [("h", "Height", "cm"), ("tof", "ToF distance", "cm"),
                             ("baro", "Barometer", "m")]),
    ("Attitude",            [("pitch", "Pitch", "deg"), ("roll", "Roll", "deg"),
                             ("yaw", "Yaw", "deg")]),
    ("Velocity",            [("vgx", "Speed X", ""), ("vgy", "Speed Y", ""),
                             ("vgz", "Speed Z", "")]),
    ("Acceleration",        [("agx", "Accel X", ""), ("agy", "Accel Y", ""),
                             ("agz", "Accel Z", "")]),
    ("Mission pad",         [("mid", "Pad ID (-1 = none)", ""), ("x", "Pad X", "cm"),
                             ("y", "Pad Y", "cm"), ("z", "Pad Z", "cm")]),
]


def _fmt(v: float | str) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def format_telemetry(state: dict, age: float) -> str:
    """Human-readable, grouped telemetry block."""
    lines = [f"  telemetry (age {age:.1f}s)"]
    shown: set[str] = set()
    for title, fields in TELEMETRY_GROUPS:
        rows = [(label, state[k], unit) for k, label, unit in fields if k in state]
        shown.update(k for k, _, _ in fields)
        if rows:
            lines.append(f"  {title}")
            lines += [f"    {label:<20} {_fmt(v)} {unit}".rstrip() for label, v, unit in rows]
    other = {k: v for k, v in state.items() if k not in shown}
    if other:
        lines.append("  Other")
        lines += [f"    {k:<20} {_fmt(v)}" for k, v in other.items()]
    return "\n".join(lines)


def wait_for_state(t: "Tello", timeout: float = 2.0) -> bool:
    """Wait briefly for the first telemetry packet. True if we have state."""
    deadline = time.monotonic() + timeout
    while not t.state and time.monotonic() < deadline:
        time.sleep(0.05)
    return bool(t.state)


def show_telemetry(t: "Tello", args: list[str]) -> None:
    """`telemetry` [raw | <key> [<key> ...]]"""
    if not wait_for_state(t):
        print("  no telemetry yet -- is the drone in SDK mode and is UDP :8890 reachable?")
        return
    state, age = dict(t.state), time.time() - t.state_t
    if age > 1.5:
        print(f"  WARNING: telemetry is stale ({age:.1f}s old) -- link may be down")
    if not args:
        print(format_telemetry(state, age))
    elif args[0] == "raw":
        print("  " + "  ".join(f"{k}={_fmt(v)}" for k, v in state.items()))
    else:
        for k in args:
            print(f"  {k} = {_fmt(state[k])}" if k in state else f"  {k}: not in telemetry "
                  f"(have: {', '.join(state)})")


def watch_telemetry(t: "Tello", seconds: float | None) -> None:
    """Live one-line readout, refreshed ~5x/s. Ctrl-C (or the time limit) stops it."""
    if not wait_for_state(t):
        print("  no telemetry yet")
        return
    print("  watching telemetry -- Ctrl-C to stop")
    end = time.monotonic() + seconds if seconds else None
    try:
        while end is None or time.monotonic() < end:
            s = t.state
            g = lambda k: _fmt(s.get(k, "?"))
            line = (f"  bat {g('bat')}%  h {g('h')}cm  tof {g('tof')}cm  "
                    f"p/r/y {g('pitch')}/{g('roll')}/{g('yaw')}  "
                    f"temp {g('templ')}-{g('temph')}C  age {time.time() - t.state_t:.1f}s")
            print("\r" + line.ljust(100), end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print()


# --------------------------------------------------------------------------- vision
VIDEO_PORT = 11111
# OpenCV ships these AprilTag families. Check which one your tags use (the repo's
# config/tags.yaml); 36h11 is the most common default.
TAG_FAMILIES = {"16h5": "DICT_APRILTAG_16h5", "25h9": "DICT_APRILTAG_25h9",
                "36h10": "DICT_APRILTAG_36h10", "36h11": "DICT_APRILTAG_36h11"}


class Vision:
    """Tello H.264 video receiver + AprilTag detector (needs: pip install opencv-python numpy)."""

    def __init__(self, tello: Tello, family: str = "36h11", target_id: int | None = None):
        self.t, self.family, self.target_id = tello, family, target_id
        self.cv2 = None
        self.frame = None                 # latest decoded BGR frame
        self.frame_t = 0.0                # time it arrived
        self.stream_on = False
        self.stop_evt = threading.Event()
        self.thread: threading.Thread | None = None
        self._detect = None

    # -- setup
    def _load(self) -> bool:
        if self.cv2:
            return True
        try:
            import cv2
        except ImportError:
            print("  video needs OpenCV:  pip install opencv-python numpy")
            return False
        if not hasattr(cv2, "aruco"):
            print("  this OpenCV has no aruco module: pip install --upgrade opencv-python")
            return False
        aruco = cv2.aruco
        d = aruco.getPredefinedDictionary(getattr(aruco, TAG_FAMILIES[self.family]))
        if hasattr(aruco, "ArucoDetector"):                       # OpenCV >= 4.7
            p = aruco.DetectorParameters()
            p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            det = aruco.ArucoDetector(d, p)
            self._detect = det.detectMarkers
        else:                                                     # older OpenCV
            p = aruco.DetectorParameters_create()
            p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            self._detect = lambda g: aruco.detectMarkers(g, d, parameters=p)
        self.cv2 = cv2
        return True

    def start(self, wait: float = 10.0) -> bool:
        """Turn the drone's stream on, start decoding, wait for a fresh frame."""
        if not self._load():
            return False
        reply = self.t.send("streamon", timeout=3.0, retries=1)
        if reply != "ok":
            print(f"  streamon -> {reply or 'NO REPLY'}")
            return False
        self.stream_on = True
        if self.thread is None or not self.thread.is_alive():
            self.stop_evt.clear()
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        t0, deadline = time.time(), time.monotonic() + wait
        while self.frame_t <= t0 and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.frame_t <= t0:
            print(f"  no video frames after {wait:.0f}s. Check: joined to the drone's Wi-Fi, "
                  f"firewall allows incoming UDP :{VIDEO_PORT}, no other app using the stream.")
            return False
        return True

    def _reader(self) -> None:
        # low-latency decode: drop buffering so we always see the newest frame
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")
        url = f"udp://0.0.0.0:{VIDEO_PORT}?overrun_nonfatal=1&fifo_size=5000000"
        cap = self.cv2.VideoCapture(url, self.cv2.CAP_FFMPEG)
        while not self.stop_evt.is_set():
            if not cap.isOpened():
                time.sleep(0.5)
                cap = self.cv2.VideoCapture(url, self.cv2.CAP_FFMPEG)
                continue
            ok, f = cap.read()
            if ok and f is not None:
                self.frame, self.frame_t = f, time.time()
            else:
                time.sleep(0.01)
        cap.release()

    def stop(self) -> None:
        self.stop_evt.set()
        if self.stream_on:
            self.t.send("streamoff", timeout=1.0)
            self.stream_on = False
        if self.thread:
            self.thread.join(timeout=1.0)

    # -- detection
    def detect(self, frame) -> list[dict]:
        """AprilTags in a frame: id, pixel center, mean edge length, offset from image center
        (dx, dy in -1..1; +dx = tag is right of center, +dy = below center), corners."""
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect(gray)
        if ids is None:
            return []
        h, w = gray.shape
        tags = []
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(4, 2)
            cx, cy = (float(v) for v in pts.mean(axis=0))
            size = sum(float(((pts[k] - pts[(k + 1) % 4]) ** 2).sum() ** 0.5)
                       for k in range(4)) / 4
            tags.append({"id": int(i), "center": (cx, cy), "size": size,
                         "offset": ((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2)),
                         "corners": pts})
        return sorted(tags, key=lambda tg: tg["id"])

    def is_target(self, tag: dict) -> bool:
        return self.target_id is not None and tag["id"] == self.target_id

    def annotate(self, frame, tags: list[dict]) -> None:
        cv2 = self.cv2
        h, w = frame.shape[:2]
        cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        for tg in tags:
            col = (0, 255, 0) if self.target_id is None or self.is_target(tg) else (0, 165, 255)
            cv2.polylines(frame, [tg["corners"].astype("int32")], True, col, 2)
            cx, cy = (int(v) for v in tg["center"])
            cv2.circle(frame, (cx, cy), 4, col, -1)
            cv2.putText(frame, f"id {tg['id']}  {tg['size']:.0f}px", (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        s = self.t.state
        hud = (f"bat {_fmt(s.get('bat', '?'))}%  h {_fmt(s.get('h', '?'))}cm  "
               f"tags: {[tg['id'] for tg in tags] or 'none'}"
               + (f"  (target {self.target_id})" if self.target_id is not None else ""))
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _tag_line(v: Vision, tg: dict) -> str:
    cx, cy = tg["center"]
    dx, dy = tg["offset"]
    return (f"  tag {tg['id']:>3}  center ({cx:4.0f},{cy:4.0f})  offset ({dx:+.2f},{dy:+.2f})"
            f"  size {tg['size']:.0f}px" + ("  <-- TARGET" if v.is_target(tg) else ""))


def video_window(v: Vision, seconds: float | None) -> None:
    """Live video window with AprilTag overlay. q / Esc closes it, s saves a snapshot."""
    if not v.start():
        return
    cv2 = v.cv2
    print("  live video -- q or Esc in the window (or Ctrl-C) to close, s = snapshot")
    end = time.monotonic() + seconds if seconds else None
    last_t, last_ids = 0.0, None
    try:
        while end is None or time.monotonic() < end:
            if v.frame is None or v.frame_t == last_t:
                if cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_t = v.frame_t
            img = v.frame.copy()
            tags = v.detect(img)
            v.annotate(img, tags)
            ids = tuple(tg["id"] for tg in tags)
            if ids != last_ids:
                print(f"  tags in view: {list(ids) or 'none'}")
                for tg in tags:
                    if v.is_target(tg):
                        print(_tag_line(v, tg))
                last_ids = ids
            cv2.imshow("tello", img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"snap_{int(time.time())}.png"
                cv2.imwrite(name, v.frame)
                print(f"  saved {name}")
    except KeyboardInterrupt:
        print()
    finally:
        cv2.destroyAllWindows()
        for _ in range(4):                    # lets macOS actually close the window
            cv2.waitKey(1)


def detect_loop(v: Vision, seconds: float) -> None:
    """Headless: print every AprilTag seen for `seconds`, then a summary."""
    if not v.start():
        return
    print(f"  looking for tag{v.family} for {seconds:g}s"
          + (f" (target id {v.target_id})" if v.target_id is not None else "") + " -- Ctrl-C to stop")
    seen: dict[int, int] = {}
    end, last_t, last_print = time.monotonic() + seconds, 0.0, 0.0
    try:
        while time.monotonic() < end:
            if v.frame is None or v.frame_t == last_t:
                time.sleep(0.02)
                continue
            last_t = v.frame_t
            tags = v.detect(v.frame)
            for tg in tags:
                seen[tg["id"]] = seen.get(tg["id"], 0) + 1
            if tags and time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                for tg in tags:
                    print(_tag_line(v, tg))
    except KeyboardInterrupt:
        print()
    if seen:
        print("  seen: " + ", ".join(f"id {i} x{n} frames" for i, n in sorted(seen.items())))
    else:
        print("  no tags seen")


def snapshot(v: Vision, path: str | None) -> None:
    if not v.start():
        return
    path = path or f"snap_{int(time.time())}.png"
    print(f"  saved {path}" if v.cv2.imwrite(path, v.frame) else f"  could not write {path}")


# --------------------------------------------------------------------------- commands
MOVES = {"up", "down", "left", "right", "forward", "back"}   # cm, 20..500
TURNS = {"cw", "ccw"}                                         # degrees, 1..360
ALIASES = {"fwd": "forward", "f": "forward", "b": "back", "l": "left", "r": "right",
           "u": "up", "d": "down", "to": "takeoff", "e": "emergency", "q": "quit",
           "exit": "quit", "h": "help", "?": "help",
           "state": "telemetry", "tel": "telemetry", "t": "telemetry", "w": "watch"}

HELP = """\
  connect / status
    battery            battery %            height     height (cm, from state)
    telemetry          grouped telemetry    wifi       signal-to-noise
      (alias: state, t)  telemetry raw  one line;  telemetry bat h tof  just those keys
    watch [seconds]    live readout (Ctrl-C stops, stays connected)
    sdk / sn           firmware / serial    raw <cmd>  send any SDK string as-is
  flight
    takeoff | land     stop (hover)         emergency  (or e) CUT MOTORS -- drone falls
    up|down|left|right|forward|back <cm>    20-500, e.g. `forward 100`
    cw|ccw <deg>       rotate 1-360         speed <cm/s>  10-100
    rc <lr> <fb> <ud> <yaw>   stick values, capped at +/-RC_LIMIT; `stop` zeroes them
    go <x> <y> <z> <speed>    fly to relative point (cm)
  video / AprilTags   (needs: pip install opencv-python numpy)
    video [seconds]    live window with AprilTag overlay (q/Esc closes, s = snapshot)
    detect [seconds]   headless: print each tag's id, pixel position, size (default 10 s)
    snap [file.png]    save one frame
    streamon | streamoff   raw stream control (video/detect/snap turn it on themselves)
  other
    sleep <s>          pause (useful in --do)    help    quit (lands if flying)"""


def run(t: Tello, line: str, rc_limit: int) -> bool:
    """Execute one command line. Returns False when the user asked to quit."""
    parts = line.strip().split()
    if not parts:
        return True
    op, args = ALIASES.get(parts[0].lower(), parts[0].lower()), parts[1:]

    def need(n: int) -> list[int] | None:
        try:
            nums = [int(a) for a in args]
        except ValueError:
            print(f"  {op}: expected {n} whole number(s)")
            return None
        if len(nums) != n:
            print(f"  {op}: expected {n} number(s), got {len(nums)}")
            return None
        return nums

    def show(reply: str | None) -> None:
        print(f"  {reply if reply is not None else 'NO REPLY (timeout)'}")

    if op == "quit":
        return False
    if op == "help":
        print(HELP.replace("RC_LIMIT", str(rc_limit)))
    elif op == "emergency":
        t.send_nowait("emergency")
        t.flying = False
        print("  EMERGENCY sent -- motors cut")
    elif op == "takeoff":
        r = t.send("takeoff", timeout=T_TAKEOFF)
        t.flying = t.flying or r == "ok"
        show(r)
    elif op == "land":
        t.send_nowait("rc 0 0 0 0")
        r = t.send("land", timeout=T_MOVE)
        if r == "ok":
            t.flying = False
        show(r)
    elif op == "stop":
        t.send_nowait("rc 0 0 0 0")
        show(t.send("stop"))
    elif op in MOVES:
        if (n := need(1)) is None:
            return True
        if not 20 <= n[0] <= 500:
            print("  distance must be 20-500 cm")
            return True
        show(t.send(f"{op} {n[0]}", timeout=T_MOVE))
    elif op in TURNS:
        if (n := need(1)) is None:
            return True
        if not 1 <= n[0] <= 360:
            print("  angle must be 1-360 degrees")
            return True
        show(t.send(f"{op} {n[0]}", timeout=T_MOVE))
    elif op == "speed":
        if (n := need(1)) is None:
            return True
        if not 10 <= n[0] <= 100:
            print("  speed must be 10-100 cm/s")
            return True
        show(t.send(f"speed {n[0]}"))
    elif op == "rc":
        if (n := need(4)) is None:
            return True
        if any(abs(v) > rc_limit for v in n):
            print(f"  rc values must be within +/-{rc_limit}")
            return True
        t.send_nowait("rc " + " ".join(map(str, n)))
        print("  sent (rc has no reply; `stop` to zero)")
    elif op == "go":
        if (n := need(4)) is None:
            return True
        show(t.send("go " + " ".join(map(str, n)), timeout=T_MOVE))
    elif op in {"battery", "battery?"}:
        show(t.send("battery?"))
    elif op in {"wifi", "wifi?"}:
        show(t.send("wifi?"))
    elif op in {"sdk", "sdk?"}:
        show(t.send("sdk?"))
    elif op in {"sn", "sn?"}:
        show(t.send("sn?"))
    elif op == "height":
        h = t.state.get("h")
        print(f"  {h:.0f} cm" if isinstance(h, float) else "  no telemetry yet")
    elif op == "telemetry":
        show_telemetry(t, args)
    elif op == "watch":
        try:
            watch_telemetry(t, float(args[0]) if args else None)
        except ValueError:
            print("  watch [seconds]")
    elif op in {"video", "detect", "snap"}:
        if t.vision is None:
            print("  vision not initialised")
        else:
            try:
                if op == "video":
                    video_window(t.vision, float(args[0]) if args else None)
                elif op == "detect":
                    detect_loop(t.vision, float(args[0]) if args else 10.0)
                else:
                    snapshot(t.vision, args[0] if args else None)
            except ValueError:
                print(f"  {op}: expected a number of seconds")
    elif op in {"streamon", "streamoff"}:
        show(t.send(op))
    elif op == "sleep":
        try:
            time.sleep(float(args[0]))
        except (IndexError, ValueError):
            print("  sleep <seconds>")
    elif op == "raw":
        if not args:
            print("  raw <sdk command>")
        else:
            cmd = " ".join(args)
            r = t.send(cmd, timeout=T_MOVE)
            if cmd == "takeoff" and r == "ok":
                t.flying = True
            if cmd == "land" and r == "ok":
                t.flying = False
            show(r)
    else:
        print(f"  unknown command {parts[0]!r} -- type `help`")
    return True


# --------------------------------------------------------------------------- setup
def iface_for_drone(name: str) -> str:
    """Look the drone's USB stick up in config/dongles.json and bring its nmcli profile up."""
    try:
        reg = json.loads(DONGLES.read_text())
    except FileNotFoundError:
        sys.exit(f"{DONGLES} not found")
    entry = next((d for d in reg.get("dongles", []) if d.get("drone") == name), None)
    if not entry or not entry.get("iface"):
        sys.exit(f"{name} has no dongle in {DONGLES}. Run core/tello_dongle_setup.py first, "
                 "or use --iface / no flag.")
    iface = entry["iface"]
    print(f"{name} -> {iface}; bringing up Wi-Fi profile...")
    try:
        r = subprocess.run(["nmcli", "connection", "up", name, "ifname", iface],
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"  nmcli: {r.stderr.strip() or r.stdout.strip()} (continuing anyway)")
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"  nmcli unavailable ({exc}); assuming the stick is already joined")
    return iface


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    how = ap.add_mutually_exclusive_group()
    how.add_argument("--drone", help="drone name in config/dongles.json, e.g. TELLO-1")
    how.add_argument("--iface", help="network interface to pin to, e.g. wlx6c4cbce345b3")
    ap.add_argument("--do", help='semicolon-separated commands, e.g. "takeoff; up 50; land"')
    ap.add_argument("--ip", default=TELLO_IP, help=argparse.SUPPRESS)          # testing
    ap.add_argument("--cmd-port", type=int, default=CMD_PORT, help=argparse.SUPPRESS)
    ap.add_argument("--tag-family", default="36h11", choices=sorted(TAG_FAMILIES),
                    help="AprilTag family to detect (default 36h11)")
    ap.add_argument("--tag-id", type=int, help="highlight/announce this tag id as the target")
    ap.add_argument("--rc-limit", type=int, default=30, help="max |rc| value (default 30)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print raw SDK traffic")
    a = ap.parse_args()

    iface = iface_for_drone(a.drone) if a.drone else a.iface
    try:
        t = Tello(a.ip, iface=iface, local_cmd_port=a.cmd_port, verbose=a.verbose)
    except OSError as exc:
        sys.exit(f"could not open UDP ports {a.cmd_port}/{STATE_PORT}: {exc} "
                 "(is another Tello script running? `pkill -f tello_`)")

    print(f"connecting to {a.ip}{' via ' + iface if iface else ''}...")
    if not t.connect():
        t.close()
        sys.exit("no reply to `command`. Check: drone on (yellow blinking), this computer "
                 "joined to its TELLO-XXXXXX Wi-Fi, no other app connected to it.")
    t.vision = Vision(t, a.tag_family, a.tag_id)
    t.start_keepalive()
    print(f"connected. battery {t.send('battery?') or '?'}%  sdk {t.send('sdk?') or '?'}")

    try:
        if a.do:
            for step in a.do.split(";"):
                if step.strip():
                    print(f"> {step.strip()}")
                    if not run(t, step, a.rc_limit):
                        break
        else:
            print("type `help` for commands, `quit` to exit (lands if flying)\n")
            while True:
                try:
                    line = input("tello> ")
                except EOFError:
                    break
                if not run(t, line, a.rc_limit):
                    break
    except KeyboardInterrupt:
        print("\nCtrl-C")
    finally:
        t.close()
        print("bye")


if __name__ == "__main__":
    main()