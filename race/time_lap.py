#!/usr/bin/env python3
"""Time one lap using pose from a live.jsonl track (or stdin JSON lines).

Each input line needs ``t`` (seconds) and ``pos`` / ``tag`` with x,y,z in the map frame.
Example (from engine live tracks)::

  {"t": 12.4, "tag": {"x": 0.1, "y": 0.0, "z": 0.2}}

Usage::

  python race/time_lap.py recordings/session_TELLO-1.live.jsonl
  python race/time_lap.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from race.lap_timer import LapTimer, load_course, pole_poses_from_map  # noqa: E402


def _pos_from_row(row: dict) -> tuple[float, float, float] | None:
    for key in ("pos", "tag", "position"):
        block = row.get(key)
        if isinstance(block, dict) and all(k in block for k in ("x", "y", "z")):
            return float(block["x"]), float(block["y"]), float(block["z"])
        if isinstance(block, (list, tuple)) and len(block) >= 3:
            return float(block[0]), float(block[1]), float(block[2])
    return None


def run_track(path: Path) -> float | None:
    course = load_course()
    poles = pole_poses_from_map()
    timer = LapTimer(course, poles)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pos = _pos_from_row(row)
            if pos is None:
                continue
            t = float(row.get("t", row.get("time", 0.0)))
            st = timer.update(t, pos)
            if st["event"]:
                print(f"[{st['event']}] t={t:.3f} phase={st['phase']} elapsed={st['elapsed_s']}")
            if timer.phase.name == "FINISHED":
                break
    return timer.lap_time_s


def selftest() -> None:
    """Synthetic oval: leave A, hit B, return A."""
    course = load_course()
    poles = {1: (0.0, 0.0, 0.0), 2: (4.0, 0.0, 0.0)}
    timer = LapTimer(course, poles)
    # sit at A
    assert timer.update(0.0, (0.1, 0.0, 0.0))["phase"] == "AT_START"
    # leave A -> start
    st = timer.update(1.0, (1.5, 0.0, 0.0))
    assert st["event"] == "lap_start", st
    # miss B then hit B
    timer.update(2.0, (2.5, 0.0, 0.0))
    st = timer.update(3.0, (4.0, 0.0, 0.1))
    assert st["event"] == "turn", st
    # finish at A
    st = timer.update(5.5, (0.2, 0.0, 0.0))
    assert st["event"] == "lap_finish", st
    assert abs(timer.lap_time_s - 4.5) < 1e-9
    print("SELFTEST PASSED lap_time_s=", timer.lap_time_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("track", nargs="?", type=Path, help="live.jsonl path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.track:
        ap.error("track path required (or --selftest)")
    lap = run_track(args.track)
    if lap is None:
        raise SystemExit("no complete lap in track")
    print(f"LAP_TIME_S={lap:.3f}")


if __name__ == "__main__":
    main()
