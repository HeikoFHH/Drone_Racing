"""Deterministic toy-plant regression, NOT Isaac Sim or proof of physical flight."""

import json
import math
import random
from .policy import BaselinePolicy, Observation, PolicyConfig


def run(seed=0, fault=None):
    rng = random.Random(seed)
    cfg = PolicyConfig()
    policy = BaselinePolicy(cfg)
    dt = 0.05
    p = [rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.08, 0.08), 1.0]
    v = [0.0, 0.0, 0.0]
    yaw = rng.uniform(-0.1, 0.1)
    last_capture = 0.0
    last_measured = p[:]
    poses = []
    reason = None
    peak_rc = 0
    for tick in range(2400):
        now = tick * dt
        dropout = fault == "pose_loss" and now >= 5.0
        if not dropout:
            last_capture = now
            last_measured = [value + rng.gauss(0, 0.003) for value in p]
        obs = Observation(last_capture, now, tuple(last_measured), tuple(v), yaw,
                          0.03, 0.05, 75.0, not (fault == "link_loss" and now >= 5),
                          True, True)
        decision = policy.step(obs, now)
        peak_rc = max(peak_rc, *(abs(i) for i in decision.rc))
        poses.append((now, tuple(p), decision.waypoint_index))
        if decision.request_land:
            reason = decision.reason
            break
        right, forward, up, cw = decision.rc
        f = forward / 100 * cfg.horizontal_mps_at_100
        r = right / 100 * cfg.horizontal_mps_at_100
        desired = (math.cos(yaw) * f + math.sin(yaw) * r,
                   math.sin(yaw) * f - math.cos(yaw) * r,
                   up / 100 * cfg.vertical_mps_at_100)
        for axis in range(3):
            v[axis] += (desired[axis] - v[axis]) * dt / 0.18
            p[axis] += v[axis] * dt
        yaw -= cw / 100 * cfg.yaw_rps_at_100 * dt
    return {"seed": seed, "fault": fault, "complete": decision.mode == "complete",
            "elapsed_s": round(now, 2), "terminal_reason": reason or "timeout",
            "peak_abs_rc": peak_rc, "last_waypoint": decision.waypoint_index}, poses


def main():
    baseline = [run(i)[0] for i in range(20)]
    injected = [run(0, fault)[0] for fault in ("pose_loss", "link_loss")]
    summary = {"plant": "toy_first_order_no_wind_no_video_no_radio",
               "baseline": baseline, "fault_injection": injected,
               "completion_count": sum(x["complete"] for x in baseline),
               "scenario_count": len(baseline)}
    print(json.dumps(summary, indent=2))
    return 0 if all(x["complete"] for x in baseline) and all(not x["complete"] and x["terminal_reason"] != "timeout" for x in injected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
