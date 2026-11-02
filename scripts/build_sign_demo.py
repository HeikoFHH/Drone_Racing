"""Build the self-contained demo website of the sign mission: docs/sign_demo/index.html.

Runs the REAL policies in the simulator (virtual time, no browser needed), records them, and bakes the recordings
into one HTML file with the same control panel as the live interface (policy choice, Start, Stop) plus a camera view,
top-down and side views and a distance chart. Stop in the page switches to a recorded stop-and-land that was run from
that moment of the flight.

    .venv-sim/bin/python scripts/build_sign_demo.py [--out docs/sign_demo/index.html]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sign_mission.drone import VirtualClock
from sign_mission.policies import POLICIES
from sign_mission.runner import MissionRunner
from sign_mission.sim import Scenario, SimDrone
from sign_mission.spec import SIGN_WIDTH_M

DISTANCES = (4.0, 8.0)
BRANCH_EVERY_S = 1.5
SIGN_HEIGHT = 1.0


def record(policy: str, distance: float, stop_at: float | None = None, every: int = 2):
    clk = VirtualClock()
    sd = SimDrone(Scenario(distance_m=distance, sign_height_m=SIGN_HEIGHT, perception="analytic"), clock=clk, seed=3)
    r = MissionRunner(sd)
    rows, labels, events, tick = [], [], [], [0]
    last_label = [None]
    orig_send = sd.send_rc

    def send(*a):
        orig_send(*a)
        tick[0] += 1
        st = r.status()
        label = st["state"] if st["state"] else st["phase"]
        if label not in labels:
            labels.append(label)
        if label != last_label[0]:
            events.append([round(clk.now(), 1), label])
            last_label[0] = label
        if tick[0] % every == 0:
            tr = sd.truth()
            src = {"vision": 1, "dead_reckoning": 2, "near_field": 2}.get(st["source"], 0)
            rows.append([round(tr["t"], 2), round(tr["x"], 3), round(tr["y"], 3), round(tr["z"], 3), round(tr["theta_deg"], 1),
                         st["range_m"], int(st["detected"]), src, labels.index(label), *st["rc"]])
    sd.send_rc = send
    if stop_at is not None:
        orig_sleep = clk.sleep

        def sleep(dt):
            orig_sleep(dt)
            if clk.now() >= stop_at and not r.stop_evt.is_set():
                r.stop()
        clk.sleep = sleep
    r.start(policy)
    r.thread.join(600)
    s = r.status()
    if s["phase"] == "LANDED":                      # the touchdown happens inside the blocking land command: add it
        tr = sd.truth()
        if "LANDED" not in labels:
            labels.append("LANDED")
        rows.append([round(tr["t"], 2), round(tr["x"], 3), round(tr["y"], 3), round(tr["z"], 3), round(tr["theta_deg"], 1),
                     None, 0, 0, labels.index("LANDED"), 0, 0, 0, 0])
        events.append([round(clk.now(), 1), "LANDED"])
    return {"rows": rows, "labels": labels, "events": events, "phase": s["phase"], "reason": s["reason"], "result": {
        k: v for k, v in s["result"].items() if k in ("final_range_m", "out_distance_m", "returned_m")},
        "end": sd.truth()["plane_distance_m"]}


def build() -> dict:
    data = {"sign_w": SIGN_WIDTH_M, "sign_h": SIGN_HEIGHT, "stop_m": 0.5, "cols": ["t", "x", "y", "z", "th", "range", "det", "src", "label", "r", "f", "u", "yw"],
            "policies": {n: c.title for n, c in POLICIES.items()}, "runs": {}}
    for pol in POLICIES:
        for d in DISTANCES:
            main = record(pol, d)
            run = {"policy": pol, "distance": d, "labels": main["labels"], "rows": main["rows"], "events": main["events"],
                   "phase": main["phase"], "result": main["result"], "branches": []}
            t_end = main["rows"][-1][0]
            for k, T in enumerate([BRANCH_EVERY_S * i for i in range(1, int(t_end // BRANCH_EVERY_S) + 1)]):
                # only branch while the mission is still in the air and the policy is not done yet
                b = record(pol, d, stop_at=T, every=4)
                tail = [row for row in b["rows"] if row[0] >= T - 0.01]
                lab = b["labels"]
                run["branches"].append({"t": T, "labels": lab, "rows": tail, "events": [e for e in b["events"] if e[0] >= T - 0.1],
                                        "phase": b["phase"], "result": b["result"]})
            data["runs"][f"{pol}_{int(d)}"] = run
            print(pol, d, "m:", len(main["rows"]), "samples,", len(run["branches"]), "stop branches,", main["phase"], main["result"], flush=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/sign_demo/index.html"))
    a = ap.parse_args()
    data = build()
    html = (Path(__file__).parent / "sign_demo_template.html").read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")
    print("wrote", a.out, f"{a.out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
