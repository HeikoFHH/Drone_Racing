"""Run race/race_policy.RacePolicy closed loop against the torch sim, next to BaselinePolicy.

Same harness as baseline_policy_adapter.py: stress_sim's seeded start states, the same scenarios
and plants, truth lap timer. The sim state (course_flu_v1: X A->B, Y left, Z up) is converted to
the repo map frame (x right, y up, z out of tag 1) with frames.py, and packed the way
core/positional.FilteredPositioner hands it to the policy in flight:
    Obs.pos / Obs.vel   -> filtered position / velocity (map frame)
    Obs.raw["xyz","R"]  -> this frame's fix: position + camera->map rotation
Perfect state plus stress_sim's 3 mm noise: like the rest of the MVP sim, no camera/tag model
yet (M3), so this tests control, not perception. race/sim_race.py covers perception.

    .venv-sim/bin/python -m drone_race_sim.baselines.race_policy_adapter --seeds 100
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .. import _repo  # noqa: F401
from .. import frames
from ..config import EnvCfg
from ..course import map_poles_for_lap_timer
from ..env import CAUSE_NAMES, TelloRaceVecEnv
from ..lap import PHASE_NAMES
from .baseline_policy_adapter import initial_conditions, run_baseline, summarize
from policy_integration.policy import PolicyConfig
from policy_integration.stress_sim import SCENARIOS
from race.race_policy import Obs, PolicyCfg, RacePolicy

R_WM = frames.DEFAULT_R_WORLD_MAP          # map -> course rotation
ARM_AFTER_S = 1.0                          # settled in A's gate this long -> go (race/fly_race.py: 1.5)


def camera_to_map(yaw: float) -> np.ndarray:
    """camera->map rotation for a level drone with course yaw ``yaw`` (CCW from +X).
    Camera axes: x right, y down, z forward."""
    c, s = math.cos(yaw), math.sin(yaw)
    forward, left, up = np.array([c, s, 0.0]), np.array([-s, c, 0.0]), np.array([0.0, 0.0, 1.0])
    R_cam_course = np.column_stack([-left, -up, forward])
    return R_WM.T @ R_cam_course


def run_race_policy(scenario, seeds, *, plant: str = "placeholder", horizon_s: float = 60.0,
                    cfg: PolicyCfg | None = None, noise_m: float = 0.003, keep_track: bool = False) -> list[dict]:
    seeds = list(seeds)
    E, dt = len(seeds), 0.05
    env = TelloRaceVecEnv(EnvCfg(num_envs=E, device="cpu", dtype="float64", autoreset=False, dt=dt,
                                 episode_s=horizon_s))
    _, pos0, yaw0 = initial_conditions(seeds)
    opts = {"pos": pos0, "yaw": yaw0, "wind_xy": (0.0, scenario.wind_y_mps)}
    if plant == "stress":
        opts.update(preset="stress_sim", deadband=scenario.deadband_rc)
    elif scenario.deadband_rc > 0:
        opts["deadband"] = scenario.deadband_rc
    env.reset(options=opts)

    poles = map_poles_for_lap_timer(env.geom)                       # map frame, same course as the sim
    base = cfg or PolicyCfg()
    # the sim's poles sit at height 0; fly at the sim's reference height above them
    base = replace(base, rel_height_m=env.cfg.ref_height)
    policies = [RacePolicy(poles, env.geom.pole_a_tag, env.geom.pole_b_tag, replace(base)) for _ in range(E)]
    rngs = [random.Random(1000 + s) for s in seeds]
    settled = [None] * E
    result = [None] * E
    tracks = [[] for _ in range(E)]
    rc = torch.zeros(E, 4, dtype=env.dtype)
    for tick in range(int(horizon_s / dt) + 2):
        now = tick * dt
        pos_c, vel_c, yaw = env.dyn.pos.numpy(), env.dyn.vel.numpy(), env.dyn.yaw.numpy()
        phases = env.lap.phase.tolist()
        rc.zero_()
        for i in range(E):
            if result[i] is not None:
                continue
            p_map = frames.course_to_map(pos_c[i]) + np.array([rngs[i].gauss(0, noise_m) for _ in range(3)])
            v_map = R_WM.T @ vel_c[i]
            R = camera_to_map(float(yaw[i]))
            obs = Obs(t=now, pos=p_map.tolist(), vel=v_map.tolist(),
                      raw={"xyz": p_map.tolist(), "R": R.tolist(), "tags": []}, lost=False)
            phase = PHASE_NAMES[phases[i]]
            pol = policies[i]
            if phase == "AT_START" and not pol.armed:
                settled[i] = now if settled[i] is None else settled[i]
                if now - settled[i] >= ARM_AFTER_S:
                    pol.armed = True
            out, info = pol.step(obs, phase, dt)
            rc[i] = torch.tensor(out, dtype=env.dtype)
            if keep_track:
                tracks[i].append((now, *pos_c[i].tolist(), phase, info["mode"], *out, float(yaw[i])))
        _, _, _, _, info = env.step_rc(rc)
        for i in info["done"].nonzero(as_tuple=True)[0].tolist():
            if result[i] is None:
                lt = float(info["lap_time"][i])
                result[i] = {"seed": seeds[i], "cause": CAUSE_NAMES[int(info["cause"][i])],
                             "lap_time_s": None if lt != lt else round(lt, 2),
                             "track": tracks[i] if keep_track else None}
        if all(r is not None for r in result):
            break
    return [r or {"seed": s, "cause": "timeout", "lap_time_s": None, "track": tracks[i] if keep_track else None}
            for i, (r, s) in enumerate(zip(result, seeds))]


def summarize_ours(res: list[dict]) -> dict:
    ok = [r for r in res if r["cause"] == "success"]
    causes: dict[str, int] = {}
    for r in res:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    return {"success": len(ok), "n": len(res),
            "median_lap_s": statistics.median(r["lap_time_s"] for r in ok) if ok else None, "causes": causes}


def plot_tracks(rows: dict, geom, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.6 * len(rows)), squeeze=False)
    colors = {"IDLE": "#999999", "AT_START": "#2f6db3", "OUTBOUND": "#2a9d4b", "TURNED": "#d1453b", "FINISHED": "#8e44ad"}
    for ax, (name, res) in zip(axes[:, 0], rows.items()):
        for r in res:
            tr = r["track"]
            if not tr:
                continue
            xs, ys, ph = [t[1] for t in tr], [t[2] for t in tr], [t[4] for t in tr]
            for k in range(len(tr) - 1):
                ax.plot(xs[k:k + 2], ys[k:k + 2], color=colors.get(ph[k + 1], "k"), lw=1.2, alpha=0.8)
        for (px, py), lab in ((geom.pole_a_xy, "A"), (geom.pole_b_xy, "B")):
            ax.add_patch(plt.Circle((px, py), geom.gate_radius_m, fill=False, color="#bbbbbb"))
            ax.add_patch(plt.Circle((px, py), 0.25, color="black"))
            ax.annotate(lab, (px, py - 1.05), ha="center")
        s = summarize_ours(res)
        ax.set_title(f"{name}: {s['success']}/{s['n']} laps, median {s['median_lap_s']} s, causes {s['causes']}",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-1.2, geom.distance + 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel("Y left (m)")
    axes[-1, 0].set_xlabel("X from Pole A toward Pole B (m)   green = outbound, red = return")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def export_tracks(a, scen, cfg) -> None:
    """Tracks for the 3-D replay: course_flu_v1 metres, yaw CCW from +X, rc sticks per 50 ms tick."""
    import json
    geom = TelloRaceVecEnv(EnvCfg(num_envs=1)).geom
    env_cfg = EnvCfg()
    out = {"dt": 0.05, "policy": {"cruise_mps": cfg.cruise_mps, "max_rc": cfg.max_rc},
           "course": {"pole_a": geom.pole_a_xy, "pole_b": geom.pole_b_xy, "gate_radius": geom.gate_radius_m,
                      "pole_radius": env_cfg.pole_radius, "pole_height": env_cfg.pole_height,
                      "ref_height": env_cfg.ref_height},
           "scenarios": []}
    for sid in a.scenarios.split(","):
        sc = scen[sid]
        res = run_race_policy(sc, range(a.export_seeds), plant="placeholder", horizon_s=a.horizon, cfg=cfg,
                              keep_track=True)
        runs = []
        for r in res:
            tr = r["track"]
            runs.append({"seed": r["seed"], "cause": r["cause"], "lap_time_s": r["lap_time_s"],
                         "x": [round(t[1], 3) for t in tr], "y": [round(t[2], 3) for t in tr],
                         "z": [round(t[3], 3) for t in tr], "yaw": [round(t[10], 3) for t in tr],
                         "phase": [t[4] for t in tr], "mode": [t[5] for t in tr],
                         "rc": [[int(v) for v in t[6:10]] for t in tr]})
        out["scenarios"].append({"id": sid, "label": sc.label, "wind_y_mps": sc.wind_y_mps,
                                 "deadband_rc": sc.deadband_rc, "runs": runs})
        print(f"{sid}: {sum(r['cause'] == 'success' for r in res)}/{len(res)} laps", flush=True)
    a.export_json.parent.mkdir(parents=True, exist_ok=True)
    a.export_json.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.export_json} ({a.export_json.stat().st_size // 1024} KB)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--horizon", type=float, default=60.0)
    ap.add_argument("--scenarios", default="baseline,mild-wind,strong-wind,deadband")
    ap.add_argument("--plants", default="stress,placeholder")
    ap.add_argument("--max-rc", type=int, default=100, help="our stick cap (PPO in SIM_MVP_RESULTS used 35)")
    ap.add_argument("--cruise", type=float, default=PolicyCfg().cruise_mps)
    ap.add_argument("--no-baseline", action="store_true", help="skip Maude's BaselinePolicy columns")
    ap.add_argument("--plot", type=Path, help="save path plots (first 10 seeds per scenario, placeholder plant)")
    ap.add_argument("--export-json", type=Path, help="write 3-D tracks (placeholder plant) for the replay viewer")
    ap.add_argument("--export-seeds", type=int, default=8)
    a = ap.parse_args(argv)

    scen = {s.id: s for s in SCENARIOS}
    cfg = PolicyCfg(cruise_mps=a.cruise, max_rc=a.max_rc)
    if a.export_json:
        export_tracks(a, scen, cfg)
        return 0
    print(f"RacePolicy cruise {a.cruise} m/s, max_rc {a.max_rc}; {a.seeds} seeds, {a.horizon:g} s horizon")
    for plant in a.plants.split(","):
        print(f"\n== {plant} plant")
        print(f"{'scenario':<12} {'RacePolicy (ours)':<34} {'Baseline rc20':<22} {'Baseline rc35':<22}")
        plots = {}
        for sid in a.scenarios.split(","):
            sc = scen[sid]
            ours = run_race_policy(sc, range(a.seeds), plant=plant, horizon_s=a.horizon, cfg=cfg)
            s = summarize_ours(ours)
            cell = f"{s['success']}/{s['n']}, {s['median_lap_s']} s"
            if s["success"] < s["n"]:
                cell += f" {dict((k, v) for k, v in s['causes'].items() if k != 'success')}"
            cols = []
            if not a.no_baseline:
                for cap in (20, 35):
                    b = summarize([r for r in run_baseline(sc, range(a.seeds), plant=plant, horizon_s=a.horizon,
                                                           policy_cfg=PolicyConfig(rc_limit=cap))])
                    cols.append(f"{b['truth_laps']}/{a.seeds}, {b['median_lap_s']} s")
            print(f"{sid:<12} {cell:<34} " + " ".join(f"{c:<22}" for c in cols), flush=True)
            if a.plot and plant == "placeholder":
                plots[sid] = run_race_policy(sc, range(min(10, a.seeds)), plant=plant, horizon_s=a.horizon,
                                             cfg=cfg, keep_track=True)
        if plots:
            env = TelloRaceVecEnv(EnvCfg(num_envs=1))
            plot_tracks(plots, env.geom, a.plot)
            print(f"path plot: {a.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
