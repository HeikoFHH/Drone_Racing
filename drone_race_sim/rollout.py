"""Closed-loop rollouts of any controller in the sim, with full per-tick recording for visualization."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
import torch

from . import _repo  # noqa: F401
from .baselines.baseline_policy_adapter import initial_conditions
from .config import EnvCfg
from .controllers import Controller
from .env import CAUSE_NAMES, TelloRaceVecEnv
from .lap import PHASE_NAMES
from policy_integration.stress_sim import SCENARIOS

SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}


@dataclass
class Rollout:
    controller: str
    scenario: str
    seeds: list
    outcome: list            # "lap" | "lap_est_only" | "landed" | "crash:<cause>" | "timeout"
    lap_time_s: list         # truth timer (None if no lap)
    lap_time_est_s: list     # estimate-driven timer (what flight would report)
    takeover: list           # (time, reason) of the guard's backup, or None
    land_reason: list
    dt: float
    tracks: list = field(default_factory=list)   # per env dict of numpy arrays (if recorded)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n = len(self.seeds)
        laps = [t for t in self.lap_time_s if t is not None]
        counts: dict[str, int] = {}
        for o in self.outcome:
            counts[o] = counts.get(o, 0) + 1
        return {"controller": self.controller, "scenario": self.scenario, "n": n, "laps": len(laps),
                "median_lap_s": statistics.median(laps) if laps else None, "outcomes": counts,
                "takeovers": sum(t is not None for t in self.takeover)}


def rollout(controller: Controller, scenario="baseline", seeds=range(20), *, plant="placeholder", sensing="perfect",
            horizon_s=60.0, rc_cap=35.0, record=False, start="stress", device="cpu", dtype="float32",
            sensor_noise_scale=1.0, env_seed=0, obs_fn=None) -> Rollout:
    """Run ``controller`` on ``len(seeds)`` parallel envs.

    plant: "stress" (policy_integration/stress_sim's plant) or "placeholder" (tello_placeholder.yaml).
    sensing: "perfect" or "realistic" (camera/tag/estimator model). start: "stress" (stress_sim's seeded start
    states near the route's first waypoint) or "spawn" (env's own random spawn inside the A gate).
    """
    sc = SCENARIO_BY_ID[scenario] if isinstance(scenario, str) else scenario
    seeds = list(seeds)
    E = len(seeds)
    dt = 0.05
    env = TelloRaceVecEnv(EnvCfg(num_envs=E, device=device, dtype=dtype, autoreset=False, rc_cap=rc_cap, episode_s=horizon_s,
                                 sensing=sensing, sensor_noise_scale=sensor_noise_scale, seed=env_seed))
    opts: dict = {"wind_xy": (0.0, sc.wind_y_mps)}
    if start == "stress":
        _, pos0, yaw0 = initial_conditions(seeds)
        opts.update(pos=pos0.to(env.dtype), yaw=yaw0.to(env.dtype))
    if plant == "stress":
        opts.update(preset="stress_sim", deadband=sc.deadband_rc)
    elif sc.deadband_rc > 0:
        opts["deadband"] = sc.deadband_rc
    obs, _ = env.reset(seed=env_seed, options=opts)
    controller.reset(env, seeds)
    finished = [False] * E
    outcome: list = [None] * E
    lap_t: list = [None] * E
    lap_est: list = [None] * E
    land_reason: list = [None] * E
    est_done: list = [None] * E          # tick at which the estimate-driven timer finished the lap
    GRACE_TICKS = 60                      # 3 s for the true timer to agree before calling it estimate-only
    rec = {k: [] for k in ("t", "pos", "yaw", "rc", "phase", "mode", "est", "lost", "seen", "fix_age")} if record else None
    n_ticks = int(horizon_s / dt) + 1
    for tick in range(n_ticks):
        now = round(tick * dt, 8)
        rc, infos = controller.act(env, obs, now)
        for i in range(E):
            if finished[i]:
                rc[i] = 0.0
            elif infos[i].get("land"):
                finished[i] = True
                outcome[i] = "landed"
                land_reason[i] = infos[i].get("reason") or infos[i].get("mode")
                rc[i] = 0.0
        if record:
            s = env.sensed
            rec["t"].append(env.t.cpu().numpy().copy())
            rec["pos"].append(env.dyn.pos.cpu().numpy().copy())
            rec["yaw"].append(env.dyn.yaw.cpu().numpy().copy())
            rec["rc"].append(torch.nan_to_num(rc).cpu().numpy().copy())
            rec["phase"].append(env.lap.phase.cpu().numpy().copy())
            rec["mode"].append([i.get("mode", "") for i in infos])
            rec["est"].append(s["pos"].cpu().numpy().copy())
            rec["lost"].append(s["lost"].cpu().numpy().copy())
            rec["seen"].append(s["seen"].cpu().numpy().copy())
            rec["fix_age"].append(s["fix_age"].cpu().numpy().copy())
        obs, _, te, tr, info = env.step_rc(torch.nan_to_num(rc))
        for i in info["done"].nonzero(as_tuple=True)[0].tolist():
            if outcome[i] is None:
                cause = CAUSE_NAMES[int(info["cause"][i])]
                outcome[i] = "lap" if cause == "success" else "timeout" if cause == "timeout" else f"crash:{cause}"
                finished[i] = True
        for i in range(E):  # truth lap time at the moment the env finished
            if lap_t[i] is None and bool(env.lap.phase[i] == 4):
                lap_t[i] = round(float(env.lap.lap_time[i]), 3)
            if lap_est[i] is None and bool(env.lap_est.phase[i] == 4):
                lap_est[i] = round(float(env.lap_est.lap_time[i]), 3)
        for i in range(E):
            if est_done[i] is None and not finished[i] and lap_est[i] is not None:
                est_done[i] = tick
            if est_done[i] is not None and not finished[i] and tick - est_done[i] >= GRACE_TICKS:
                finished[i] = True       # the official (estimate) lap ended the race; truth never agreed
                outcome[i] = "lap_est_only"
        if all(finished):
            break
    outcome = [o or "timeout" for o in outcome]
    # an env that finished the lap and then got a land request keeps its lap outcome
    outcome = ["lap" if lap_t[i] is not None and not outcome[i].startswith("crash") else outcome[i] for i in range(E)]
    takeover = [None] * E
    if hasattr(controller, "takeover_t"):
        takeover = [(controller.takeover_t[i], controller.reason[i]) if controller.takeover_t[i] is not None else None
                    for i in range(E)]
    tracks = []
    if record:
        T = {k: np.array(v) for k, v in rec.items() if k != "mode"}
        modes = rec["mode"]
        for i in range(E):
            tracks.append({"t": T["t"][:, i], "x": T["pos"][:, i, 0], "y": T["pos"][:, i, 1], "z": T["pos"][:, i, 2],
                           "yaw": T["yaw"][:, i], "rc": T["rc"][:, i], "phase": [PHASE_NAMES[p] for p in T["phase"][:, i]],
                           "mode": [m[i] for m in modes], "est_x": T["est"][:, i, 0], "est_y": T["est"][:, i, 1],
                           "lost": T["lost"][:, i], "seen": T["seen"][:, i], "fix_age": T["fix_age"][:, i]})
    name = controller.name
    return Rollout(name, sc.id, seeds, outcome, lap_t, lap_est, takeover, land_reason, dt, tracks,
                   {"plant": plant, "sensing": sensing, "horizon_s": horizon_s, "rc_cap": rc_cap,
                    "params_source": env.params_source, "sensor_source": env.sensor_source})
