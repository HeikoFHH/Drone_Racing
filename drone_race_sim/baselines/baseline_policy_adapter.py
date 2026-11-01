"""Run policy_integration.BaselinePolicy closed loop against the torch sim (baseline #1, D11).

The observation pipeline is a faithful port of policy_integration/stress_sim.py (delay history,
pose loss, link loss, 3 mm noise, same RNG order) so that, on the stress_sim plant preset in
float64, outcomes match stress_sim.run() exactly. Scalar per-env loop: evaluation only, not the
training hot path.
"""

from __future__ import annotations

import random
from collections import deque

import torch

from .. import _repo  # noqa: F401
from ..config import EnvCfg
from ..env import TelloRaceVecEnv
from policy_integration.policy import BaselinePolicy, Observation, PolicyConfig
from policy_integration.stress_sim import Scenario


def initial_conditions(seeds):
    """stress_sim's start states: near the A-offset waypoint, small random heading."""
    rngs, pos, yaw = [], [], []
    for s in seeds:
        rng = random.Random(s)
        pos.append([rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.08, 0.08), 1.0])
        yaw.append(rng.uniform(-0.1, 0.1))
        rngs.append(rng)
    return rngs, torch.tensor(pos, dtype=torch.float64), torch.tensor(yaw, dtype=torch.float64)


def run_baseline(scenario: Scenario, seeds, *, plant: str = "stress", dtype: str = "float64", device: str = "cpu",
                 horizon_s: float | None = None, policy_cfg: PolicyConfig | None = None) -> list[dict]:
    """plant "stress" = stress_sim's plant; "placeholder" = tello_placeholder.yaml (delay, deadband 8, lag)."""
    seeds = list(seeds)
    E, dt = len(seeds), 0.05
    horizon = scenario.timeout_s if horizon_s is None else horizon_s
    cfg = EnvCfg(num_envs=E, device=device, dtype=dtype, enable_terminations=False, autoreset=False, dt=dt)
    env = TelloRaceVecEnv(cfg)
    rngs, pos0, yaw0 = initial_conditions(seeds)
    opts = {"pos": pos0, "yaw": yaw0, "wind_xy": (0.0, scenario.wind_y_mps)}
    if plant == "stress":
        opts["preset"] = "stress_sim"
        opts["deadband"] = scenario.deadband_rc
    elif scenario.deadband_rc > 0:
        opts["deadband"] = scenario.deadband_rc
    env.reset(options=opts)
    pcfg = policy_cfg or PolicyConfig()
    policies = [BaselinePolicy(pcfg) for _ in range(E)]
    history = [deque() for _ in range(E)]
    last_frame: list = [None] * E
    decision: list = [None] * E
    finished = [False] * E
    elapsed = [0.0] * E
    peak = [0] * E
    max_age = [0.0] * E
    rc = torch.zeros(E, 4, dtype=env.dtype)
    for tick in range(int(horizon / dt) + 1):
        now = round(tick * dt, 8)
        rc.zero_()
        pos, vel, yaw = env.dyn.pos.tolist(), env.dyn.vel.tolist(), env.dyn.yaw.tolist()
        for i in range(E):
            if finished[i]:
                continue
            elapsed[i] = now
            history[i].append((now, tuple(pos[i]), tuple(vel[i]), yaw[i]))
            cutoff = now - scenario.delay_s
            h = history[i]
            while len(h) > 1 and h[1][0] <= cutoff + 1e-8:
                h.popleft()
            available = h[0] if h[0][0] <= cutoff + 1e-8 else None
            lost = scenario.pose_loss_at_s is not None and now >= scenario.pose_loss_at_s
            lf = last_frame[i]
            if available is not None and not lost and (lf is None or available[0] > lf[0]):
                captured, actual, v, heading = available
                measured = tuple(x + rngs[i].gauss(0, 0.003) for x in actual)
                last_frame[i] = lf = (captured, measured, v, heading)
            if lf is None:
                continue  # no pose yet: runner disabled, plant idles
            captured, measured, v, heading = lf
            link_ok = not (scenario.link_loss_at_s is not None and now >= scenario.link_loss_at_s)
            obs = Observation(captured, now, measured, v, heading, 0.03, 0.05, 75.0, link_ok, True, True)
            d = decision[i] = policies[i].step(obs, now)
            max_age[i] = max(max_age[i], now - captured)
            peak[i] = max(peak[i], *(abs(x) for x in d.rc))
            if d.request_land:
                finished[i] = True
                continue
            rc[i] = torch.tensor(d.rc, dtype=env.dtype)
        if all(finished):
            break
        env.step_rc(rc)
    out = []
    for i in range(E):
        d = decision[i]
        complete = d is not None and d.mode == "complete"
        aborted = d is not None and d.mode == "abort"
        out.append({"seed": seeds[i], "outcome": "complete" if complete else "abort" if aborted else "timeout",
                    "reason": d.reason if complete or aborted else "simulation_timeout",
                    "elapsed_s": round(elapsed[i], 3), "peak_abs_rc": peak[i], "max_pose_age_s": round(max_age[i], 3),
                    "truth_lap_finished": bool(env.lap.phase[i] == 4),
                    "truth_lap_time_s": None if torch.isnan(env.lap.lap_time[i]) else round(float(env.lap.lap_time[i]), 3)})
    return out


def summarize(results: list[dict]) -> dict:
    counts = {o: sum(r["outcome"] == o for r in results) for o in ("complete", "abort", "timeout")}
    laps = [r["truth_lap_time_s"] for r in results if r["truth_lap_finished"] and r["truth_lap_time_s"] is not None]
    return {"counts": counts, "truth_laps": len(laps), "median_lap_s": sorted(laps)[len(laps) // 2] if laps else None}
