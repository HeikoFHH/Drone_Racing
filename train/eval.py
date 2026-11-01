"""Evaluate a checkpoint against BaselinePolicy on the dynamics-side stress scenarios (MVP table).

    .venv-sim/bin/python -m train.eval --ckpt outputs/sim/run1/last.pt --seeds 100

Scenarios are policy_integration/stress_sim.py's baseline, mild-wind, strong-wind, deadband
(delay/pose-loss/link-loss need the M3 sensor model). All runs use the same 60 s horizon and the
same start states (stress_sim's seeded initial conditions). "success" = the truth lap timer
finished the lap (A -> B -> A) with no crash; lap time is the truth timer's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from drone_race_sim import EnvCfg, TelloRaceVecEnv
from drone_race_sim.baselines.baseline_policy_adapter import initial_conditions, run_baseline
from drone_race_sim.env import CAUSE_NAMES
from policy_integration.policy import PolicyConfig
from policy_integration.stress_sim import SCENARIOS
from train.ppo import PPO, PPOCfg

SCENARIO_IDS = ("baseline", "mild-wind", "strong-wind", "deadband")
HORIZON_S = 60.0


def load_ppo(ckpt_path: Path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    env = TelloRaceVecEnv(EnvCfg(num_envs=2, device=device, rc_cap=ck["meta"]["rc_cap"]))
    ppo = PPO(env, PPOCfg(), device=device)
    ppo.load_state_dict(ck)
    return ppo, ck["meta"]


def run_ppo(ppo: PPO, meta: dict, scenario, seeds, plant: str, horizon_s=HORIZON_S, device="cpu") -> list[dict]:
    seeds = list(seeds)
    E = len(seeds)
    env = TelloRaceVecEnv(EnvCfg(num_envs=E, device=device, rc_cap=meta["rc_cap"], autoreset=False,
                                 episode_s=horizon_s))
    _, pos0, yaw0 = initial_conditions(seeds)
    opts = {"pos": pos0.float(), "yaw": yaw0.float(), "wind_xy": (0.0, scenario.wind_y_mps)}
    if plant == "stress":
        opts.update(preset="stress_sim", deadband=scenario.deadband_rc)
    elif scenario.deadband_rc > 0:
        opts["deadband"] = scenario.deadband_rc
    obs, _ = env.reset(options=opts)
    first = [None] * E
    for _ in range(int(horizon_s / env.cfg.dt) + 2):
        obs, _, te, tr, info = env.step(ppo.act_deterministic(obs))
        for i in info["done"].nonzero(as_tuple=True)[0].tolist():
            if first[i] is None:
                lt = float(info["lap_time"][i])
                first[i] = {"seed": seeds[i], "cause": CAUSE_NAMES[int(info["cause"][i])],
                            "lap_time_s": None if lt != lt else round(lt, 2)}
        if all(first):
            break
    return [f or {"seed": s, "cause": "timeout", "lap_time_s": None} for f, s in zip(first, seeds)]


def summarize_ppo(res):
    ok = [r for r in res if r["cause"] == "success"]
    causes = {}
    for r in res:
        causes[r["cause"]] = causes.get(r["cause"], 0) + 1
    return {"success": len(ok), "n": len(res), "median_lap_s": statistics.median(r["lap_time_s"] for r in ok) if ok else None,
            "causes": causes}


def summarize_baseline(res):
    ok = [r for r in res if r["truth_lap_finished"]]
    return {"success": len(ok), "n": len(res),
            "median_lap_s": statistics.median(r["truth_lap_time_s"] for r in ok) if ok else None,
            "outcomes": {o: sum(r["outcome"] == o for r in res) for o in ("complete", "abort", "timeout")}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    ppo, meta = load_ppo(a.ckpt, a.device)
    seeds = range(a.seeds)
    scen = {s.id: s for s in SCENARIOS}
    report = {"checkpoint": str(a.ckpt), "meta": meta, "horizon_s": HORIZON_S, "seeds": a.seeds, "plants": {}}
    for plant in ("stress", "placeholder"):
        rows = {}
        for sid in SCENARIO_IDS:
            sc = scen[sid]
            rows[sid] = {
                "ppo": summarize_ppo(run_ppo(ppo, meta, sc, seeds, plant, device=a.device)),
                "baseline_rc20": summarize_baseline(run_baseline(sc, seeds, plant=plant, dtype="float32", horizon_s=HORIZON_S)),
                "baseline_rc35": summarize_baseline(run_baseline(sc, seeds, plant=plant, dtype="float32", horizon_s=HORIZON_S,
                                                                 policy_cfg=PolicyConfig(rc_limit=35))),
            }
            r = rows[sid]
            fmt = lambda x: f"{x['success']:3d}/{x['n']} lap {x['median_lap_s'] if x['median_lap_s'] is not None else '  - '}s"
            print(f"[{plant:11s}] {sid:12s} PPO {fmt(r['ppo'])} | Baseline rc20 {fmt(r['baseline_rc20'])} | "
                  f"Baseline rc35 {fmt(r['baseline_rc35'])}", flush=True)
        report["plants"][plant] = rows
    out = a.out or a.ckpt.parent / "eval.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
