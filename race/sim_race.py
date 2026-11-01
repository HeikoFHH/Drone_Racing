#!/usr/bin/env python3
"""Best-case check of race/race_policy.py through the REAL detection pipeline.

Each tick it renders the camera image the Tello would see (tag36h11 markers drawn with
cv2.aruco.generateImageMarker -- the convention cal_and_map's detector is defined by --
projected with the calibrated camera in config/camera/default.yaml) and hands that
frame to the unchanged core/positional_kalman_filter.FilteredPositioner, i.e.
core/positional.Positioner -> tello_calibration TagDetector -> TagMap -> Kalman.
The lap is judged by race/lap_timer.LapTimer. Only the drone's motion is simulated
(a kinematic Tello: velocity follows rc * mps_per_rc with a 0.35 s lag).

  python race/sim_race.py                       # tag in view at the start
  python race/sim_race.py --scenario search     # starts facing away: find the tag first
  python race/sim_race.py --delay 3             # 3 ticks (150 ms) of video latency
  python race/sim_race.py --wind 3 --gust 1     # 3 m/s crosswind with 1 m/s gusts
  python race/sim_race.py --spacing 73 --tag-mm 1000 --relay-every 6 --relay-mm 1000
                                                # long course with relay tags along it

Writes outputs/sim/<scenario>.png (the path) and .jsonl (every tick).
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

from positional_kalman_filter import FilteredPositioner  # noqa: E402
from race.lap_timer import LapTimer, load_course, pole_poses_from_map  # noqa: E402
from race.race_policy import Obs, PolicyCfg, RacePolicy  # noqa: E402
from tello_calibration.calib.io import camera_from_dict, load_yaml  # noqa: E402
from tello_calibration.tags.detect import object_corners  # noqa: E402

DT = 0.05
W, H = 960, 720
OUT = ROOT / "outputs" / "sim"


# ----------------------------------------------------------------------------- world
RELAY_ID0 = 10        # relay tags are ids 10, 11, ...
RELAY_SIDE_M = 1.0    # relays stand this far to the side of the A-B line (so you don't hit them)


def write_course_map(path: Path, spacing: float, relay_every: float | None = None) -> list[int]:
    """Course: Pole A's tag (1) faces B; Pole B's tag (2) faces A. Optional relay tags
    every ``relay_every`` m along the course, beside the line, facing A (toward a camera
    that always looks at B). Returns the relay ids."""
    flip = np.diag([-1.0, 1.0, -1.0]).tolist()
    tags = {1: {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0], "seen": 1, "samples": 1},
            2: {"R": flip, "t": [0.0, 0.0, spacing], "seen": 1, "samples": 1}}
    relays = []
    if relay_every:
        z = relay_every
        while z < spacing - 1.0:
            tid = RELAY_ID0 + len(relays)
            tags[tid] = {"R": flip, "t": [RELAY_SIDE_M, 0.0, round(z, 3)], "seen": 1, "samples": 1}
            relays.append(tid)
            z += relay_every
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"origin_tag": 1, "frame": "origin tag: x right, y up, z out of "
                                    "the tag face; meters", "tags": tags}, sort_keys=False))
    return relays


class Wind:
    """Mean wind + gusts. Gusts are an Ornstein-Uhlenbeck process per ground axis -- the
    first-order (Dryden-style) turbulence model: random, but correlated over ``tau`` s,
    with standard deviation ``gust`` m/s. Map ground axes (x, z)."""

    def __init__(self, speed: float, dir_deg: float, gust: float, tau: float, rng) -> None:
        a = math.radians(dir_deg)                       # 0 = blowing from A to B, 90 = from the left
        self.mean = speed * np.array([math.sin(a), 0.0, math.cos(a)])
        self.gust, self.tau, self.rng = gust, tau, rng
        self.g = np.zeros(3)

    def step(self, dt: float) -> np.ndarray:
        if self.gust > 0:
            noise = self.rng.normal(0, 1, 3) * np.array([1.0, 0.0, 1.0])
            self.g += -self.g * dt / self.tau + self.gust * math.sqrt(2 * dt / self.tau) * noise
        return self.mean + self.g


def camera_R(heading: float) -> np.ndarray:
    """camera->map rotation, level flight, forward = (cos a, 0, sin a)."""
    s, c = math.sin(heading), math.cos(heading)
    return np.array([[-s, 0.0, c], [0.0, -1.0, 0.0], [c, 0.0, s]])


class Renderer:
    def __init__(self, tag_poses: dict, tag_size_m: dict, seed: int) -> None:
        self.K, self.dist = camera_from_dict(load_yaml(ROOT / "config" / "camera" / "default.yaml"))
        self.tags = tag_poses
        self.size = tag_size_m
        rng = np.random.default_rng(seed)
        bg = rng.integers(70, 190, (H // 8, W // 8), dtype=np.uint8)
        self.background = cv2.GaussianBlur(cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR), (0, 0), 6)
        self.rng = rng
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.markers = {}
        for tid in tag_poses:
            px = 400
            m = cv2.aruco.generateImageMarker(dictionary, tid, px, borderBits=1)
            pad = px // 8                                     # one cell of white quiet zone
            m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            src = np.float32([[pad, pad], [pad + px, pad], [pad + px, pad + px], [pad, pad + px]])
            self.markers[tid] = (m, src)

    def frame(self, cam_pos, heading) -> tuple[np.ndarray, list[int]]:
        img = self.background.copy()
        R_cm = camera_R(heading)
        drawn = []
        # painter's order: farthest first, so nearer tags cover farther ones
        order = sorted(self.tags.items(), key=lambda kv: -float(np.linalg.norm(kv[1][1] - cam_pos)))
        for tid, (R_mt, t_mt) in order:
            R_ct = R_cm.T @ R_mt
            t_ct = R_cm.T @ (t_mt - cam_pos)
            if t_ct[2] < 0.2 or float(R_ct[:, 2] @ t_ct) >= 0:   # behind camera / back of tag
                continue
            obj = object_corners(self.size[tid]) * 1.25              # include the quiet zone
            pts_cam = (R_ct @ obj.T).T + t_ct
            if (pts_cam[:, 2] < 0.1).any():
                continue
            dst, _ = cv2.projectPoints(object_corners(self.size[tid]), cv2.Rodrigues(R_ct)[0], t_ct,
                                       self.K, self.dist)
            dst = dst.reshape(4, 2).astype(np.float32)
            if np.abs(dst).max() > 5 * W:
                continue
            marker, src = self.markers[tid]
            Hm = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(marker, Hm, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full(marker.shape, 255, np.uint8), Hm, (W, H))
            img = np.where(mask > 127, warped, img)
            drawn.append(tid)
        noisy = np.clip(img.astype(np.int16) + self.rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR), drawn


# ----------------------------------------------------------------------------- run
def run(scenario: str, cfg: PolicyCfg, spacing: float, delay: int, mps_per_rc: float,
        deg_per_rc: float, seed: int, seconds: float, tag_mm: float | None = None,
        relay_every: float | None = None, relay_mm: float = 1000.0, wind: Wind | None = None,
        coupling: float = 0.35, min_side_px: float | None = None) -> dict:
    map_path = OUT / f"course_map_{os.getpid()}.yaml"      # per process: parallel runs don't collide
    relays = write_course_map(map_path, spacing, relay_every)
    course = load_course()
    poles = pole_poses_from_map(map_path)
    tags_cfg = yaml.safe_load((ROOT / "config" / "tags.yaml").read_text())
    tags_path = ROOT / "config" / "tags.yaml"
    if min_side_px is not None:                # existing tags.yaml gate: ignore tags drawn smaller
        tags_cfg["gate"]["min_side_px"] = min_side_px
    if tag_mm is not None or relays or min_side_px is not None:   # copy of tags.yaml, edited
        if tag_mm is not None:
            tags_cfg["sizes_mm"] = {k: tag_mm for k in tags_cfg["sizes_mm"]}
        tags_cfg["sizes_mm"].update({tid: relay_mm for tid in relays})
        tags_path = OUT / f"tags_{os.getpid()}.yaml"
        tags_path.write_text(yaml.safe_dump(tags_cfg, sort_keys=False))
    sizes = {int(k): float(v) / 1000.0 for k, v in tags_cfg["sizes_mm"].items()}
    raw_map = yaml.safe_load(map_path.read_text())["tags"]
    tag_poses = {int(k): (np.asarray(v["R"], float), np.asarray(v["t"], float)) for k, v in raw_map.items()}

    render = Renderer(tag_poses, sizes, seed)
    positioner = FilteredPositioner("tello_1", map_path=map_path, tags_path=tags_path)   # real detection + Kalman
    timer = LapTimer(course, poles)
    policy = RacePolicy(poles, course.pole_a_tag, course.pole_b_tag, cfg)

    A, B = np.asarray(poles[course.pole_a_tag]), np.asarray(poles[course.pole_b_tag])
    u = (B - A) / np.linalg.norm(B - A)
    p = A + u * 0.45 + np.array([0.15, -0.25, 0.0])        # took off inside A's gate, a bit low/off-line
    toward_b = math.atan2(u[2], u[0])
    heading = toward_b if scenario == "in-view" else toward_b + math.radians(110)
    vel = np.zeros(3)
    history = []                                            # rendered frames, for latency
    rows, phase, settled = [], "IDLE", None
    t = 0.0
    while t < seconds:
        frame, drawn = render.frame(p, heading)
        history.append((frame, drawn))
        frame, drawn = history[max(0, len(history) - 1 - delay)]
        est = positioner.locate(frame, t)
        raw = est["raw"]
        obs = Obs(t=t, pos=None if est["lost"] else est["xyz"], vel=est["vel"],
                  raw=raw if raw is not None and est["accepted"] else None,
                  lost=est["lost"], tags=raw["tags"] if raw else [])
        event = None
        if obs.pos is not None:
            st = timer.update(t, tuple(obs.pos))
            phase, event = st["phase"], st["event"]
        if phase == "AT_START" and not policy.armed and obs.pos is not None:
            settled = settled if settled is not None else t
            if t - settled >= 1.0:
                policy.armed, event = True, "armed"
        rc, info = policy.step(obs, phase, DT)
        rows.append({"t": round(t, 2), "true": p.round(3).tolist(), "est": obs.pos, "heading_deg":
                     round(math.degrees(heading), 1), "drawn": drawn, "detected": obs.tags,
                     "phase": phase, "event": event, "mode": info["mode"], "rc": list(rc),
                     "turb": info.get("turb"), "speed_scale": info.get("speed_scale")})
        if info["mode"] == "land" or (phase == "FINISHED" and (info["mode"] == "arrived" or
                                                               t - timer.t_finish > 3)):
            break
        right, fwd, up, yaw = rc
        cmd = camera_R(heading) @ np.array([right, 0.0, fwd]) * mps_per_rc
        cmd[1] = up * 0.006
        w = wind.step(DT) if wind is not None else np.zeros(3)
        # The Tello's own downward-camera hold fights the wind; ``coupling`` is the share
        # that still gets through (0 = perfect hold, 1 = drifts with the air). Fit it from
        # a hover log: drift speed / wind speed.
        vel += (cmd + coupling * w - vel) * min(1.0, DT / 0.35)
        p = p + vel * DT
        heading += math.radians(yaw * deg_per_rc * DT)
        t += DT
    return {"rows": rows, "lap_s": timer.lap_time_s, "A": A, "B": B, "policy": policy,
            "gate_r": course.gate_radius_m, "relays": {tid: tag_poses[tid][1] for tid in relays}}


# ----------------------------------------------------------------------------- report
def metrics(res: dict) -> dict:
    rows, A, B = res["rows"], res["A"], res["B"]
    u = (B - A) / np.linalg.norm(B - A)
    perp = np.array([-u[2], 0.0, u[0]])
    lap = [r for r in rows if r["phase"] in ("OUTBOUND", "TURNED")]
    err = [np.linalg.norm(np.subtract(r["est"], r["true"])) for r in rows if r["est"] is not None]
    out = {
        "lap_s": res["lap_s"],
        "first_fix_s": next((r["t"] for r in rows if r["est"] is not None), None),
        "events": {r["event"]: r["t"] for r in rows if r["event"]},
        "detect_err_cm": (round(100 * float(np.mean(err)), 1), round(100 * float(np.max(err)), 1)) if err else None,
        "frames_with_tag": f"{sum(1 for r in rows if r['detected'])}/{len(rows)}",
    }
    if lap:
        side = [float((np.asarray(r["true"]) - A) @ perp) for r in lap]
        pts = np.array([r["true"] for r in lap])
        length = float(np.linalg.norm(np.diff(pts[:, [0, 2]], axis=0), axis=1).sum())
        rc = np.array([r["rc"] for r in lap], float)
        d_rc = np.abs(np.diff(rc, axis=0)).mean(axis=0) if len(rc) > 1 else np.zeros(4)
        flips = int(np.sum(np.diff(np.sign(rc[:, 0][np.abs(rc[:, 0]) > 2])) != 0))
        hd = np.array([r["heading_deg"] for r in lap])
        out.update({
            "max_sideways_cm": round(100 * max(abs(s) for s in side), 1),
            "path_m": round(length, 2),
            "mean_step_change_rc": dict(zip(("right", "fwd", "up", "yaw"), d_rc.round(1).tolist())),
            "right_stick_sign_flips": flips,
            "heading_wobble_deg": round(float(hd.max() - hd.min()), 1),
        })
    along = [float((np.asarray(r["true"]) - A) @ u) for r in rows]
    out["furthest_along_m"] = round(max(along), 1)
    scales = [r["speed_scale"] for r in lap if r.get("speed_scale") is not None]
    if scales:
        out["speed_scale_mean_min"] = (round(float(np.mean(scales)), 2), round(float(min(scales)), 2))
    if res["relays"]:
        clear = min(math.hypot(r["true"][0] - q[0], r["true"][2] - q[2]) for r in rows for q in res["relays"].values())
        out["closest_to_relay_stand_m"] = round(clear, 2)
    out["last_mode"] = rows[-1]["mode"]
    return out


def plot(res: dict, path: Path, title: str) -> None:
    """Top-down path. Course axis to the right; sideways axis stretched so drift shows."""
    rows, A, B, pol = res["rows"], res["A"], res["B"], res["policy"]
    span = float(np.linalg.norm(B - A))
    sx = 820.0 / (span + 1.5)                         # px per metre along the course
    sy = 150.0                                        # px per metre sideways (fixed)
    ox, oy = 80, 260
    img = np.full((760, 980, 3), 255, np.uint8)

    def P(xyz):
        return int(ox + xyz[2] * sx), int(oy - xyz[0] * sy)

    step = 0.5 if span < 10 else 5.0
    for x in np.arange(-1.5, 1.51, 0.5):
        cv2.line(img, (ox - 60, int(oy - x * sy)), (ox + int((span + 1) * sx), int(oy - x * sy)), (238, 238, 238), 1)
    for z in np.arange(0, span + 1.01, step):
        cv2.line(img, (int(ox + z * sx), oy - 230), (int(ox + z * sx), oy + 230), (238, 238, 238), 1)
    cv2.line(img, P(pol.gate_a), P(pol.gate_b), (190, 190, 190), 1, cv2.LINE_AA)
    for pole, name in ((A, "A tag1"), (B, "B tag2")):
        axes = (max(2, int(res["gate_r"] * sx)), int(res["gate_r"] * sy))
        cv2.ellipse(img, P(pole), axes, 0, 0, 360, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.circle(img, P(pole), 7, (0, 0, 0), -1)
        cv2.putText(img, name, (P(pole)[0] - 25, P(pole)[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for q in res["relays"].values():
        cv2.drawMarker(img, P(q), (0, 120, 0), cv2.MARKER_SQUARE, 8, 1)
    colors = {"IDLE": (150, 150, 150), "AT_START": (180, 120, 0), "OUTBOUND": (40, 160, 40),
              "TURNED": (40, 40, 220), "FINISHED": (160, 0, 160)}
    for r0, r1 in zip(rows, rows[1:]):
        col = (0, 170, 255) if r1["mode"] == "search" else colors.get(r1["phase"], (0, 0, 0))
        cv2.line(img, P(r0["true"]), P(r1["true"]), col, 2, cv2.LINE_AA)
    for r in rows:
        if r["event"]:
            cv2.circle(img, P(r["true"]), 5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"{r['event']} {r['t']:.1f}s", (P(r["true"])[0] + 6, P(r["true"])[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(img, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    legend = [("search", (0, 170, 255)), ("at start", colors["AT_START"]), ("outbound", colors["OUTBOUND"]),
              ("return", colors["TURNED"]), ("finished", colors["FINISHED"])]
    for i, (name, col) in enumerate(legend):
        cv2.line(img, (12 + i * 110, 42), (40 + i * 110, 42), col, 3)
        cv2.putText(img, name, (44 + i * 110, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    cv2.putText(img, f"top-down; grid {step:g} m along, 0.5 m sideways (sideways stretched x{sy / sx:.0f}); "
                "squares = relay tags; ellipses = 0.8 m gates", (12, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (90, 90, 90), 1)

    # lower panel: sticks and the wind-adaptive speed scale vs time
    y0, hgt, t_end = 530, 210, max(rows[-1]["t"], 1e-3)
    cv2.rectangle(img, (60, y0), (960, y0 + hgt), (220, 220, 220), 1)
    cv2.line(img, (60, y0 + hgt // 2), (960, y0 + hgt // 2), (220, 220, 220), 1)
    names = (("right", (220, 120, 0)), ("forward", (40, 160, 40)), ("up", (150, 150, 150)), ("yaw", (40, 40, 220)))
    for k, (name, col) in enumerate(names):
        pts = [(int(60 + r["t"] / t_end * 900), int(y0 + hgt / 2 - r["rc"][k] / 100 * hgt / 2)) for r in rows]
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
        cv2.putText(img, name, (70 + k * 90, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    sc = [(int(60 + r["t"] / t_end * 900), int(y0 + hgt - (r["speed_scale"] or 0) * hgt)) for r in rows
          if r.get("speed_scale") is not None]
    if sc:
        cv2.polylines(img, [np.array(sc, np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "speed scale (black, 0..1)", (430, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "rc sticks -100..100", (760, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    cv2.putText(img, "0 s", (55, y0 + hgt + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.putText(img, f"{t_end:.1f} s", (920, y0 + hgt + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=["in-view", "search"], default="in-view")
    ap.add_argument("--cruise", type=float, default=PolicyCfg().cruise_mps)
    ap.add_argument("--look-at", choices=["B", "target"], default="B")
    ap.add_argument("--spacing", type=float, default=4.0, help="pole spacing, m")
    ap.add_argument("--delay", type=int, default=0, help="video latency in 50 ms ticks")
    ap.add_argument("--mps-per-rc", type=float, default=0.008, help="the simulated drone's real stick gain")
    ap.add_argument("--deg-per-rc", type=float, default=0.6)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag-mm", type=float, help="print size of both pole tags' black square (default tags.yaml)")
    ap.add_argument("--relay-every", type=float, help="relay tags every N m along the course")
    ap.add_argument("--relay-mm", type=float, default=1000.0, help="relay tag black-square size")
    ap.add_argument("--wind", type=float, default=0.0, help="mean wind speed, m/s")
    ap.add_argument("--wind-dir", type=float, default=90.0, help="deg: 0 = from A toward B, 90 = crosswind")
    ap.add_argument("--gust", type=float, default=0.0, help="gust std-dev, m/s")
    ap.add_argument("--gust-tau", type=float, default=2.0, help="gust correlation time, s")
    ap.add_argument("--coupling", type=float, default=0.35, help="share of wind the Tello's hold lets through")
    ap.add_argument("--min-side-px", type=float, help="tags.yaml gate.min_side_px (ignore far tags)")
    ap.add_argument("--adapt", action="store_true", help="slow down in gusts (PolicyCfg.adapt_to_wind)")
    args = ap.parse_args()

    cfg = PolicyCfg(cruise_mps=args.cruise, look_at=args.look_at, mps_per_rc=args.mps_per_rc,
                    adapt_to_wind=args.adapt)
    wind = Wind(args.wind, args.wind_dir, args.gust, args.gust_tau, np.random.default_rng(args.seed + 100)) \
        if (args.wind or args.gust) else None
    seconds = max(args.seconds, 4 * args.spacing / args.cruise)
    res = run(args.scenario, cfg, args.spacing, args.delay, args.mps_per_rc, args.deg_per_rc,
              args.seed, seconds, args.tag_mm, args.relay_every, args.relay_mm, wind, args.coupling,
              args.min_side_px)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{args.scenario}_cruise{args.cruise:g}_delay{args.delay}_{args.look_at}"
    if args.tag_mm:
        name += f"_tag{args.tag_mm:g}mm"
    if args.spacing != 4.0:
        name += f"_{args.spacing:g}m"
    if args.relay_every:
        name += f"_relay{args.relay_every:g}"
    if args.min_side_px:
        name += f"_minpx{args.min_side_px:g}"
    if wind is not None:
        name += f"_wind{args.wind:g}g{args.gust:g}" + ("_adapt" if args.adapt else "")
    with (OUT / f"{name}.jsonl").open("w") as f:
        for r in res["rows"]:
            f.write(json.dumps(r) + "\n")
    m = metrics(res)
    plot(res, OUT / f"{name}.png", f"{name}   lap {m['lap_s'] and round(m['lap_s'], 2)} s")
    (OUT / f"{name}.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    print(f"path plot: {OUT / (name + '.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
