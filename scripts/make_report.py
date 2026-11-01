"""Run the sim test matrix and render all figures + the interactive replay.

    .venv-sim/bin/python scripts/make_report.py --ppo-perfect outputs/sim/run1/last.pt \
        --ppo-realistic outputs/sim/run_realistic/last.pt --seeds 30

Writes outputs/sim/report/{results.json, *.png, replay.html}; with --docs-dir also copies the figures there.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from drone_race_sim.controllers import (FaultyController, GuardedController, MaudeController, PolicyCfg, PolicyConfig,
                                        PPOController, RaceController)
from drone_race_sim.course import load_course_geometry
from drone_race_sim.rollout import rollout
from drone_race_sim.viz import plots
from drone_race_sim.viz.replay_html import pack_run, write_replay
from train.eval import load_ppo

SCENARIOS = ("baseline", "mild-wind", "strong-wind", "deadband")


def controllers(args):
    out = {"maude_baseline": lambda: MaudeController(),
           "race_policy": lambda: RaceController(PolicyCfg(max_rc=35), name="race_policy"),
           "guarded[race_policy]": lambda: GuardedController(RaceController(PolicyCfg(max_rc=35), name="race_policy"))}
    for key, path in (("ppo_perfect", args.ppo_perfect), ("ppo_realistic", args.ppo_realistic)):
        if path and Path(path).exists():
            ppo, meta = load_ppo(Path(path))
            out[key] = (lambda ppo=ppo, meta=meta, key=key: _named(PPOController(ppo, meta["rc_cap"]), key))
            if key == "ppo_realistic":
                out["guarded[ppo_realistic]"] = lambda ppo=ppo, meta=meta: GuardedController(_named(PPOController(ppo, meta["rc_cap"]), "ppo_realistic"))
    return out


def _named(c, name):
    c.name = name
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-perfect", type=Path, default=Path("outputs/sim/run1/last.pt"))
    ap.add_argument("--ppo-realistic", type=Path, default=Path("outputs/sim/run_realistic/last.pt"))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("outputs/sim/report"))
    ap.add_argument("--docs-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    geom = load_course_geometry()
    ctrl = controllers(a)
    seeds = range(a.seeds)
    results: dict = {"seeds": a.seeds, "matrix": {}, "faults": {}}

    # 1. controller x scenario x sensing -------------------------------------------------------------------
    for sensing in ("perfect", "realistic"):
        rows = {}
        for name, make in ctrl.items():
            if name.endswith("ppo_realistic]") or name == "ppo_realistic":
                if sensing == "perfect":
                    continue      # trained on estimator observations; not meaningful with truth observations
            rows[name] = {}
            for sc in SCENARIOS:
                ro = rollout(make(), sc, seeds, sensing=sensing, horizon_s=60.0)
                rows[name][sc] = ro.summary()
                print(f"[{sensing:9s}] {name:24s} {sc:12s} {ro.summary()['outcomes']} lap={ro.summary()['median_lap_s']}", flush=True)
        results["matrix"][sensing] = rows
        plots.plot_outcomes(rows, a.out / f"outcomes_{sensing}.png", f"Outcomes by scenario, sensing = {sensing} ({a.seeds} seeds, 60 s)")

    # 2. guard fault injection --------------------------------------------------------------------------------
    faults = ("nan", "stall", "runaway", "dive", "noise")
    prim = lambda: RaceController(PolicyCfg(max_rc=35), name="race_policy")
    for fault in faults:
        bare = rollout(FaultyController(prim(), fault, 4.0), "baseline", seeds, horizon_s=90)
        safe = rollout(GuardedController(FaultyController(prim(), fault, 4.0)), "baseline", seeds, horizon_s=90)
        results["faults"][fault] = {"unguarded": bare.summary()["outcomes"], "guarded": safe.summary()["outcomes"],
                                    "backup_reasons": sorted({t[1] for t in safe.takeover if t}), "land_reasons": sorted({r for r in safe.land_reason if r})}
        print("fault", fault, results["faults"][fault], flush=True)

    # 3. figures and replay (10 recorded seeds each) ------------------------------------------------------------
    rec = {}
    for sensing in ("perfect", "realistic"):
        ros = []
        for name, make in ctrl.items():
            if name.endswith("ppo_realistic]") or name == "ppo_realistic":
                if sensing == "perfect":
                    continue
            ro = rollout(make(), "baseline", range(10), sensing=sensing, record=True)
            ros.append(ro)
            rec[(name, sensing)] = ro
        plots.plot_comparison(ros, geom, a.out / f"paths_{sensing}.png", f"Baseline scenario, sensing = {sensing} (10 seeds)")
    runs = []
    for (name, sensing), ro in rec.items():
        plots.plot_rollout(ro, 0, geom, a.out / f"run_{name.replace('[', '_').replace(']', '')}_{sensing}.png")
        runs.append(pack_run(f"{name} | {sensing} sensing | baseline", ro, 0))
    for fault in ("nan", "stall", "runaway", "dive"):
        ro = rollout(GuardedController(FaultyController(prim(), fault, 4.0), ), "baseline", range(2), record=True, horizon_s=90)
        plots.plot_rollout(ro, 0, geom, a.out / f"guard_{fault}.png", title=f"guarded[race_policy] with a '{fault}' fault injected at t = 4 s: {ro.outcome[0]}"
                           + (f" ({ro.land_reason[0]})" if ro.land_reason[0] else "") + (f", backup: {ro.takeover[0][1]}" if ro.takeover[0] else ""))
        runs.append(pack_run(f"guard demo | race_policy + '{fault}' fault at 4 s", ro, 0))
    write_replay(runs, geom, a.out / "replay.html", "Tello race simulator replay")
    for key, path in (("run1", "outputs/sim/run1/log.csv"), ("run_realistic", "outputs/sim/run_realistic/log.csv")):
        if Path(path).exists():
            plots.plot_training(Path(path), a.out / f"training_{key}.png", f"PPO training: {key}")
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    if a.docs_dir:
        (a.docs_dir).mkdir(parents=True, exist_ok=True)
        for f in list(a.out.glob("*.png")) + [a.out / "replay.html", a.out / "results.json"]:
            shutil.copy(f, a.docs_dir / f.name)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
