#!/usr/bin/env python3
"""Fly the gate policy on a real Tello, using the repo's existing link + detection.

Same plumbing as tools/tello_to_origin.py:
  core/test_link.connect_all()  -> DroneLink (dongle-pinned sockets, video, state)
  core/positional_kalman_filter.FilteredPositioner -> tag detection + map + Kalman
  core/tello_dual_video.sdk / send_rc, core/test_link.land -> commands
  race/lap_timer.LapTimer -> lap phase;  race/race_policy -> sticks

  python race/fly_race.py race --cruise 0.6
  python race/fly_race.py goto --tag 2 --standoff 1.0
  python race/fly_race.py race --no-fly          # connect + run the policy, never take off

Linux + USB dongle, like the rest of core/. Writes recordings/<session>_<drone>.race.jsonl
(rows carry "pos", so race/time_lap.py can re-time it). CATCHER READY; Ctrl-C lands.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "core"), str(ROOT / "tools")]

from positional_kalman_filter import FilteredPositioner  # noqa: E402
from race.lap_timer import LapTimer, load_course, pole_poses_from_map  # noqa: E402
from race.race_policy import Obs, PolicyCfg, RacePolicy, TagApproach  # noqa: E402
from tello_dual_video import RECORD_DIR, sdk, send_rc  # noqa: E402
from test_link import connect_all, land  # noqa: E402

TICK_S = 0.05
ARM_AFTER_S = 1.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="race", choices=["race", "goto"])
    ap.add_argument("--tag", type=int, help="goto: tag to approach (default Pole B)")
    ap.add_argument("--standoff", type=float, default=1.0)
    ap.add_argument("--cruise", type=float, help="top speed, m/s")
    ap.add_argument("--look-at", choices=["B", "target"])
    ap.add_argument("--seconds", type=float, default=45.0, help="safety cap")
    ap.add_argument("--no-fly", action="store_true", help="never take off; print the sticks it would send")
    args = ap.parse_args()

    cfg = PolicyCfg()
    if args.cruise is not None:
        cfg.cruise_mps = args.cruise
    if args.look_at:
        cfg.look_at = args.look_at
    course, poles = load_course(), pole_poses_from_map()

    stop = threading.Event()
    dl = connect_all(stop)[0]                       # first present quick_connect drone
    positioner = FilteredPositioner(dl.name.lower().replace("-", "_"))
    print(f"{dl.name} map tags {positioner.positioner.tag_map.mapped()}", flush=True)
    if args.mode == "race":
        timer = LapTimer(course, poles)
        policy = RacePolicy(poles, course.pole_a_tag, course.pole_b_tag, cfg)
    else:
        timer = None
        approach = TagApproach(poles[args.tag or course.pole_b_tag], args.standoff, cfg)

    log_path = Path(RECORD_DIR) / f"{dl.session}_{dl.name}.race.jsonl"
    log = log_path.open("w", encoding="utf-8")
    print(f"log {log_path}", flush=True)

    def send(rc):
        if not args.no_fly:
            send_rc(dl.cmd, rc)

    phase, settled, done_since, last_print = "IDLE", None, None, 0.0
    try:
        if not args.no_fly:
            print(f"{dl.name} takeoff {sdk(dl.cmd, 'takeoff', 15)}", flush=True)
        t0 = prev = time.monotonic()
        while not stop.is_set() and time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            dt, prev = now - prev, now
            if not dl.connected():
                send((0, 0, 0, 0))                  # hold; don't fly blind on a dropped link
                stop.wait(TICK_S)
                continue
            frame = dl.frames.get(dl.name)
            if frame is None:
                dl.ensure_video(stop)
                send((0, 0, 0, 0))
                stop.wait(TICK_S)
                continue
            est = positioner.locate(frame, now)
            raw = est["raw"]
            obs = Obs(t=now, pos=None if est["lost"] else est["xyz"], vel=est["vel"],
                      raw=raw if raw is not None and est["accepted"] else None,
                      lost=est["lost"], tags=raw["tags"] if raw else [])

            event = None
            if timer is not None:
                if obs.pos is not None:
                    status = timer.update(now, tuple(obs.pos))
                    phase, event = status["phase"], status["event"]
                    if event:
                        print(f"[{now - t0:5.1f}s] {event} lap={status['lap_time_s']}", flush=True)
                if phase == "AT_START" and not policy.armed and obs.pos is not None:
                    settled = settled or now
                    if now - settled >= ARM_AFTER_S:
                        policy.armed = True
                        print(f"[{now - t0:5.1f}s] armed -> GO", flush=True)
                elif not policy.armed:
                    settled = None
                rc, info = policy.step(obs, phase, dt)
                finished = phase == "FINISHED"
            else:
                rc, info = approach.step(obs, dt)
                finished = info["mode"] == "arrived"
            if finished:
                done_since = done_since or now
                if info["mode"] == "arrived" or now - done_since > 4.0:
                    if timer is not None:
                        print(f"LAP TIME {timer.lap_time_s:.3f} s", flush=True)
                    break
            if info["mode"] == "land":
                print("no tag after a full search -> landing", flush=True)
                break
            send(rc)
            log.write(json.dumps({"t": round(now - t0, 3), "pos": obs.pos, "vel": est["vel"],
                                  "tags": obs.tags, "rc": list(rc), "phase": phase, "event": event,
                                  **info, "h": dl.heights.get(dl.name)}) + "\n")
            if now - last_print > 0.5:
                where = "lost" if obs.pos is None else "[" + ", ".join(f"{v:+.2f}" for v in obs.pos) + "]"
                print(f"{now - t0:5.1f}s {phase:9s} {info['mode']:7s} pos={where} tags={obs.tags} rc={rc}",
                      flush=True)
                last_print = now
            stop.wait(max(0.0, TICK_S - (time.monotonic() - now)))
    except KeyboardInterrupt:
        print("Ctrl-C -> landing", flush=True)
    finally:
        send((0, 0, 0, 0))
        if not args.no_fly:
            print(f"{dl.name} land {land(dl.cmd)}", flush=True)
        log.close()
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
