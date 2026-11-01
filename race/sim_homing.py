#!/usr/bin/env python3
"""73 m course check for race/homing_policy.py, through the REAL tag detector.

Each 50 ms tick renders the Tello camera image (race/sim_race.py Renderer: tag36h11
markers projected with config/camera/default.yaml) and runs the repo's unchanged
tello_calibration TagDetector on it; range = TagObservation.distance_m, bearing from its
tvec. Telemetry h and yaw are simulated (optional gyro drift). Drone motion is the same
kinematic Tello as sim_race.py, with optional wind/gusts.

  python race/sim_homing.py                              # current 132 mm tags
  python race/sim_homing.py --tag-mm 1000 --wind 2 --gust 1
  python race/sim_homing.py --heading-error 5            # start pointed 5 deg off
  python race/sim_homing.py --b-offset 3                 # pole B 3 m sideways of where you aimed
  python race/sim_homing.py --decoy                      # a different-id tag near the course

Writes outputs/sim/homing_<name>.png / .json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "core"), str(ROOT / "tools")]

from race.homing_policy import HomingCfg, HomingObs, HomingPolicy, Sighting  # noqa: E402
from race.sim_race import OUT, Renderer, Wind, camera_R  # noqa: E402
from tello_calibration.calib.io import camera_from_dict, load_yaml  # noqa: E402
from tello_calibration.tags.detect import TagDetector, load_tag_config  # noqa: E402

DT = 0.05
TAG_HEIGHT = 1.2          # tag centres above the ground; ground is map y = -TAG_HEIGHT
FLIP = np.diag([-1.0, 1.0, -1.0])


def run(args) -> dict:
    rng = np.random.default_rng(args.seed)
    L = args.course
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([args.b_offset, 0.0, L])                 # where pole B really is
    tags = {1: (np.eye(3), A), 2: (FLIP, B)}              # A faces B, B faces A
    if args.hide_b:
        del tags[2]                                       # tag B missing / never visible
    if args.decoy:
        tags[7] = (FLIP, np.array([2.0, 0.0, L * 0.6]))   # a stray tag with another id, facing A
    size_m = {tid: args.tag_mm / 1000.0 for tid in tags}

    cfg_t = yaml.safe_load((ROOT / "config" / "tags.yaml").read_text())
    # what tags.yaml SAYS the tags are (may differ from the print: --config-tag-mm)
    cfg_t["sizes_mm"] = {tid: (args.config_tag_mm or args.tag_mm) for tid in (1, 2, 7)}
    tags_path = OUT / f"homing_tags_{os.getpid()}.yaml"
    OUT.mkdir(parents=True, exist_ok=True)
    tags_path.write_text(yaml.safe_dump(cfg_t))
    K, dist = camera_from_dict(load_yaml(ROOT / "config" / "camera" / "default.yaml"))
    detector = TagDetector(load_tag_config(tags_path), K, dist)
    tags_path.unlink()
    render = Renderer(tags, size_m, args.seed)

    hc = HomingCfg(course_m=L, cruise_mps=args.cruise, mps_per_rc=args.policy_mps_per_rc,
                   height_m=TAG_HEIGHT, start_offset_m=1.0, blind_mode=args.blind_mode,
                   tag_size_m=args.tag_mm / 1000.0)
    pol = HomingPolicy(hc)
    wind = Wind(args.wind, args.wind_dir, args.gust, 2.0, np.random.default_rng(args.seed + 1)) \
        if (args.wind or args.gust) else None

    # takes off 1 m in front of tag A, at 0.8 m, nose toward where we THINK B is
    p = np.array([0.0, 0.8 - TAG_HEIGHT, 1.0])
    heading = math.pi / 2 + math.radians(args.heading_error)
    vel = np.zeros(3)
    yaw0, drift = 23.0, 0.0
    rows, t = [], 0.0
    delayed = []                                          # sightings queue, for video latency
    h_stuck = None
    while t < args.seconds:
        frame, drawn = render.frame(p, heading)
        sightings = []
        for o in detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)):
            if o.accepted and o.tvec is not None:
                R_tc = cv2.Rodrigues(np.asarray(o.rvec, float))[0]
                sightings.append(Sighting(o.tag_id, o.distance_m,
                                          math.degrees(math.atan2(o.tvec[0], o.tvec[2])),
                                          tuple(float(v) for v in np.ravel(o.tvec)),
                                          tuple(float(v) for v in R_tc[:, 2])))
        # injected faults
        if args.occlude and args.occlude[0] <= t <= args.occlude[1]:
            sightings = [s for s in sightings if s.tag_id not in (1, 2)]   # someone stands in front
        if args.dropout and rng.random() < args.dropout:
            sightings = []                                                   # frame lost / not decoded
        delayed.append(sightings)
        sightings = delayed[max(0, len(delayed) - 1 - args.delay)]
        drift += args.gyro_drift * DT
        height = p[1] + TAG_HEIGHT + rng.normal(0, 0.02)
        if args.height_stuck_at is not None and t >= args.height_stuck_at:
            h_stuck = h_stuck if h_stuck is not None else height
            height = h_stuck                                                 # sensor frozen
        yaw_tel = ((math.degrees(heading) + yaw0 + drift + 180) % 360) - 180
        if args.yaw_dead_after is not None and t >= args.yaw_dead_after:
            yaw_tel = None
        obs = HomingObs(t=t, sightings=sightings, height_m=height, yaw_deg=yaw_tel)
        rc, info = pol.step(obs, DT)
        true_a, true_b = float(np.linalg.norm(p - A)), float(np.linalg.norm(p - B))
        rows.append({"t": round(t, 2), "p": p.round(3).tolist(), "heading": round(math.degrees(heading), 1),
                     "rc": list(rc), "seen": [s.tag_id for s in sightings], "true_a": round(true_a, 2),
                     "true_b": round(true_b, 2), **info})
        if info["mode"] in ("done", "land"):
            break
        right, fwd, up, yaw = rc
        cmd = camera_R(heading) @ np.array([right, 0.0, fwd]) * args.mps_per_rc
        cmd[1] = up * 0.006
        w = wind.step(DT) if wind is not None else np.zeros(3)
        vel += (cmd + args.coupling * w - vel) * min(1.0, DT / 0.35)
        p = p + vel * DT
        heading += math.radians(yaw * args.deg_per_rc * DT)
        t += DT
    return {"rows": rows, "A": A, "B": B, "L": L, "decoy": args.decoy, "ground_y": -TAG_HEIGHT}


def summarize(res: dict) -> dict:
    rows = res["rows"]
    out = {"end": rows[-1]["mode"], "end_state": rows[-1]["state"], "time_s": rows[-1]["t"]}
    for leg, key in (("to_B", "true_b"), ("to_A", "true_a")):
        r = [x for x in rows if x["leg"] == leg]
        if not r:
            continue
        seen = [x for x in r if x["target"] in x["seen"]]
        hold = [x for x in r if x["state"] == "HOLD"]
        out[leg] = {
            "first_seen_at_m": seen[0][key] if seen else None,
            "closest_m": min(x[key] for x in r),
            "stop_at_m": hold[-1][key] if hold else None,
            "range_est_at_stop": hold[-1]["range_m"] if hold else None,
            "time_s": round(r[-1]["t"] - r[0]["t"], 1),
            "validation": {s: sum(1 for x in r if x["validation"] == s) for s in
                           sorted({x["validation"] for x in r})},
        }
    side = [abs(x["p"][0]) for x in rows]
    out["max_sideways_m"] = round(max(side), 2)
    out["height_range_m"] = [round(min(x["p"][1] for x in rows if x["state"] not in ("CLIMB",)) + TAG_HEIGHT, 2),
                             round(max(x["p"][1] for x in rows) + TAG_HEIGHT, 2)]
    out["unexpected_tags"] = rows[-1]["unexpected_tags"]
    out["spins"] = rows[-1]["spins"]
    out["return_from"] = rows[-1]["return_from"]
    out["spin_s"] = round(sum(DT for x in rows if x["state"] == "SPIN"), 1)
    out["abort_reason"] = rows[-1].get("abort_reason")
    hz = lambda x: math.hypot(x["p"][0] - res["A"][0], x["p"][2] - res["A"][2])
    hb = lambda x: math.hypot(x["p"][0] - res["B"][0], x["p"][2] - res["B"][2])
    after = [x for x in rows if x["t"] > 3.0]                 # after takeoff (starts 1 m from pole A)
    out["min_ground_dist_pole_A_m"] = round(min(hz(x) for x in after), 2) if after else None
    out["min_ground_dist_pole_B_m"] = round(min(hb(x) for x in rows), 2)
    out["height_min_max_m"] = [round(min(x["p"][1] for x in after) + TAG_HEIGHT, 2) if after else None,
                               round(max(x["p"][1] for x in rows) + TAG_HEIGHT, 2)]
    out["max_from_start_m"] = round(max(math.hypot(x["p"][0], x["p"][2] - 1.0) for x in rows), 1)
    out["crash"] = bool((out["min_ground_dist_pole_A_m"] or 9) < 0.5 or out["min_ground_dist_pole_B_m"] < 0.5
                        or (out["height_min_max_m"][0] or 9) < 0.15)
    flying = [x for x in rows if x["state"] != "CLIMB"]   # the takeoff spot is 1 m from tag A
    out["closest_to_tag_after_takeoff_m"] = round(min(min(x["true_a"] if x["leg"] != "to_B" else 99, x["true_b"])
                                                      for x in flying), 2)
    return out


def plot(res: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, L = res["rows"], res["L"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1.2]})
    colors = {"CLIMB": "#999", "SPIN": "#c2185b", "BLIND": "#e08b2a", "TRACK": "#2a9d4b", "HOLD": "#2f6db3", "TURN": "#8e44ad",
              "SEARCH": "#d1453b", "ABORT": "#000", "DONE": "#2f6db3"}
    z = [x["p"][2] for x in rows]
    xs = [x["p"][0] for x in rows]
    for k in range(len(rows) - 1):
        ax1.plot(z[k:k + 2], xs[k:k + 2], color=colors.get(rows[k + 1]["state"], "k"), lw=1.6)
    for pos, lab in ((res["A"], "tag A"), (res["B"], "tag B")):
        ax1.plot(pos[2], pos[0], "ks", ms=8)
        ax1.add_patch(plt.Circle((pos[2], pos[0]), 5.0, fill=False, ls="--", color="#d1453b", lw=1))
        ax1.annotate(lab, (pos[2], pos[0] + 0.6), ha="center")
    if res["decoy"]:
        ax1.plot(L * 0.6, 2.0, "x", color="#d1453b", ms=8)
        ax1.annotate("decoy tag", (L * 0.6, 2.4), ha="center", fontsize=8)
    ax1.set_xlim(-6, L + 6)
    lim = max(6.0, max(abs(v) for v in xs) + 1.5)
    ax1.set_ylim(-lim, lim)
    ax1.set_ylabel("sideways (m)")
    ax1.set_xlabel("along course from A (m)   dashed = 5 m keep-out")
    ax1.set_title(title, fontsize=10)
    for s, c in colors.items():
        ax1.plot([], [], color=c, lw=3, label=s)
    ax1.legend(ncol=8, fontsize=8, loc="lower center")

    t = [x["t"] for x in rows]
    true_tgt = [x["true_b"] if x["target"] == 2 else x["true_a"] for x in rows]
    est = [x["range_m"] if x["state"] in ("TRACK", "HOLD") else None for x in rows]
    ax2.plot(t, true_tgt, color="#999", lw=1, label="true distance to target tag")
    ax2.plot(t, [e if e is not None else np.nan for e in est], color="#2a9d4b", lw=1.6, label="measured range (validated)")
    ax2.axhline(5.0, color="#d1453b", ls="--", lw=1, label="5 m stop")
    bad = [x["t"] for x in rows if x["validation"] in ("not_closing", "dr_mismatch", "abort")]
    if bad:
        ax2.plot(bad, [1.0] * len(bad), "|", color="#d1453b", ms=10, label="validation flag")
    ax2.set_ylim(0, max(true_tgt) + 3)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("distance (m)")
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=105)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course", type=float, default=73.0)
    ap.add_argument("--tag-mm", type=float, default=487.7, help="black square; 2 ft sheet -> 487.7, 2 ft square -> 609.6")
    ap.add_argument("--cruise", type=float, default=1.5, help="m/s, same toward both tags")
    ap.add_argument("--mps-per-rc", type=float, default=0.02, help="simulated drone: m/s per rc unit")
    ap.add_argument("--policy-mps-per-rc", type=float, default=0.02, help="what the policy assumes")
    ap.add_argument("--deg-per-rc", type=float, default=0.6)
    ap.add_argument("--heading-error", type=float, default=0.0, help="start this many deg off the B direction")
    ap.add_argument("--b-offset", type=float, default=0.0, help="pole B this many m sideways")
    ap.add_argument("--gyro-drift", type=float, default=0.0, help="telemetry yaw drift, deg/s")
    ap.add_argument("--decoy", action="store_true")
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--wind-dir", type=float, default=90.0)
    ap.add_argument("--gust", type=float, default=0.0)
    ap.add_argument("--coupling", type=float, default=0.35)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind-mode", choices=["auto", "spin", "fly"], default="auto")
    ap.add_argument("--config-tag-mm", type=float, help="tag size written in tags.yaml if it differs from the print")
    ap.add_argument("--occlude", type=float, nargs=2, metavar=("T0", "T1"), help="hide tags A/B between these times")
    ap.add_argument("--dropout", type=float, default=0.0, help="probability a frame gives no detections")
    ap.add_argument("--delay", type=int, default=0, help="detection latency in 50 ms ticks")
    ap.add_argument("--height-stuck-at", type=float, help="height sensor freezes at this time")
    ap.add_argument("--yaw-dead-after", type=float, help="yaw telemetry stops at this time")
    ap.add_argument("--hide-b", action="store_true", help="tag B missing")
    ap.add_argument("--export-rows", type=Path, help="write the run (10 Hz) for the 3-D replay")
    ap.add_argument("--label", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    res = run(args)
    s = summarize(res)
    name = args.name or f"tag{args.tag_mm:g}_cruise{args.cruise:g}"
    plot(res, OUT / f"homing_{name}.png", f"{name}: {s['end']} after {s['time_s']} s")
    (OUT / f"homing_{name}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print(f"plot: {OUT / f'homing_{name}.png'}")
    if args.export_rows:
        keep = res["rows"][::2] + ([res["rows"][-1]] if len(res["rows"]) % 2 == 0 else [])
        args.export_rows.write_text(json.dumps({
            "name": name, "label": args.label or name, "summary": s, "course": res["L"],
            "A": res["A"].tolist(), "B": res["B"].tolist(), "decoy": res["decoy"], "hide_b": args.hide_b,
            "tag_mm": args.tag_mm, "dt": 0.1,
            "rows": [[x["t"], x["p"][2], x["p"][0], round(x["p"][1] + TAG_HEIGHT, 3), x["heading"], x["state"],
                      x["leg"], x["range_m"], x["validation"], x["rc"], int(x["target"] in x["seen"]),
                      x.get("abort_reason")] for x in keep]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
