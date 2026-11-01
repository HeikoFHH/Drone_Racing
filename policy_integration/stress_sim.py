"""CPU scenario runner using the real policy against a synthetic velocity plant.

This is not a rigid-body/Tello simulator. Wind is modelled as a world velocity
disturbance; delays, deadband and camera dropouts are synthetic test parameters.
No drone or Internet connection is opened. Full traces can drive visual playback.
"""

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import random

from .policy import BaselinePolicy, Observation, PolicyConfig


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    delay_s: float = 0.0
    wind_y_mps: float = 0.0
    deadband_rc: int = 0
    pose_loss_at_s: float | None = None
    link_loss_at_s: float | None = None
    timeout_s: float = 90.0


SCENARIOS = (
    Scenario("baseline", "Baseline"),
    Scenario("mild-wind", "Side drift: 0.06 m/s", wind_y_mps=0.06),
    Scenario("strong-wind", "Side drift: 0.12 m/s", wind_y_mps=0.12),
    Scenario("delay-150", "Camera delay: 150 ms", delay_s=0.15),
    Scenario("delay-300", "Camera delay: 300 ms", delay_s=0.3),
    Scenario("pose-loss", "Camera lost at 5 s", pose_loss_at_s=5),
    Scenario("link-loss", "Link lost at 5 s", link_loss_at_s=5),
    Scenario("deadband", "RC deadband: 12", deadband_rc=12),
)


def run(scenario: Scenario, seed=0, trace=False):
    rng = random.Random(seed)
    cfg = PolicyConfig()
    policy = BaselinePolicy(cfg)
    dt = 0.05
    p = [rng.uniform(-0.15, 0.15), 0.5 + rng.uniform(-0.08, 0.08), 1.0]
    v = [0.0, 0.0, 0.0]
    yaw = rng.uniform(-0.1, 0.1)
    history = deque()
    last_frame = None
    in_a_previous = False
    lap_phase = "IDLE"
    lap_started = None
    lap_time = None
    min_clearance = float("inf")
    max_age = 0.0
    peak_rc = 0
    trajectory = []
    decision = None
    for tick in range(int(scenario.timeout_s / dt) + 1):
        now = round(tick * dt, 8)
        snapshot = (now, tuple(p), tuple(v), yaw)
        history.append(snapshot)
        cutoff = now - scenario.delay_s
        # Keep newest snapshot whose capture time precedes the delayed delivery.
        while len(history) > 1 and history[1][0] <= cutoff + 1e-8:
            history.popleft()
        available = history[0] if history[0][0] <= cutoff + 1e-8 else None
        lost = scenario.pose_loss_at_s is not None and now >= scenario.pose_loss_at_s
        if available is not None and not lost and (last_frame is None or available[0] > last_frame[0]):
            captured, actual, vel, heading = available
            measured = tuple(value + rng.gauss(0, 0.003) for value in actual)
            last_frame = (captured, measured, vel, heading)
        if last_frame is None:
            # No pose has arrived. Runner stays disabled; no fabricated state is used.
            if trace:
                trajectory.append([now, *p, *p, 0, 0, 0, 0, "disabled", "awaiting_first_pose", 0.0, 0, lap_phase])
            continue
        captured, measured, vel, heading = last_frame
        link_ok = not (scenario.link_loss_at_s is not None and now >= scenario.link_loss_at_s)
        obs = Observation(captured, now, measured, vel, heading, 0.03, 0.05, 75.0,
                          link_ok, True, True)
        decision = policy.step(obs, now)
        age = now - captured
        max_age = max(max_age, age)
        peak_rc = max(peak_rc, *(abs(value) for value in decision.rc))
        # Independent truth-only race evaluator in the canonical world X/Y plane.
        in_a = math.hypot(p[0], p[1]) <= 0.8
        in_b = math.hypot(p[0] - 4, p[1]) <= 0.8
        if lap_phase == "IDLE" and in_a:
            lap_phase = "AT_START"
        elif lap_phase == "AT_START" and in_a_previous and not in_a:
            lap_phase, lap_started = "OUTBOUND", now
        elif lap_phase == "OUTBOUND":
            if in_b:
                lap_phase = "TURNED"
            elif in_a:
                lap_phase, lap_started = "AT_START", None
        elif lap_phase == "TURNED" and in_a:
            lap_phase, lap_time = "FINISHED", now - lap_started
        in_a_previous = in_a
        min_clearance = min(min_clearance, *(math.hypot(p[0]-x, p[1]-y) for x,y in cfg.poles_xy))
        if trace:
            trajectory.append([round(now, 3), *(round(value, 4) for value in p),
                               *(round(value, 4) for value in measured), *decision.rc,
                               decision.mode, decision.reason, round(age, 3), decision.waypoint_index, lap_phase])
        if decision.request_land:
            break  # Stop at request; do not simulate a landing that never happened.
        right, forward, up, clockwise = (value if abs(value) >= scenario.deadband_rc else 0 for value in decision.rc)
        f = forward / 100 * cfg.horizontal_mps_at_100
        r = right / 100 * cfg.horizontal_mps_at_100
        target_v = (math.cos(yaw) * f + math.sin(yaw) * r,
                    math.sin(yaw) * f - math.cos(yaw) * r + scenario.wind_y_mps,
                    up / 100 * cfg.vertical_mps_at_100)
        for axis in range(3):
            v[axis] += (target_v[axis] - v[axis]) * dt / 0.18
            p[axis] += v[axis] * dt
        yaw -= clockwise / 100 * cfg.yaw_rps_at_100 * dt
    complete = decision is not None and decision.mode == "complete"
    aborted = decision is not None and decision.mode == "abort"
    return {"scenario": asdict(scenario), "seed": seed,
            "outcome": "complete" if complete else "abort" if aborted else "timeout",
            "reason": decision.reason if complete or aborted else "simulation_timeout",
            "elapsed_s": round(now, 3), "truth_lap_finished": lap_phase == "FINISHED",
            "truth_lap_time_s": None if lap_time is None else round(lap_time, 3),
            "max_pose_age_s": round(max_age, 3), "min_truth_pole_distance_m": round(min_clearance, 3),
            "peak_abs_rc": peak_rc, "trace": trajectory if trace else None}


def suite(seeds=20, include_traces=False):
    scenarios = []
    for scenario in SCENARIOS:
        runs = [run(scenario, seed, trace=include_traces and seed == 0) for seed in range(seeds)]
        expected_abort = scenario.pose_loss_at_s is not None or scenario.link_loss_at_s is not None or scenario.delay_s > PolicyConfig().max_pose_age_s
        counts = {outcome: sum(r["outcome"] == outcome for r in runs) for outcome in ("complete", "abort", "timeout")}
        scenarios.append({"id": scenario.id, "label": scenario.label,
                          "expected_behavior": "abort" if expected_abort else "complete",
                          "counts": counts, "runs": runs})
    return {"model": "synthetic_velocity_plant_not_Tello_physics", "dt_s": 0.05,
            "seeds_per_scenario": seeds, "scenarios": scenarios}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--traces", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1 or args.seeds > 1000:
        parser.error("--seeds must be between 1 and 1000")
    report = suite(args.seeds, args.traces)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": report["model"], "scenarios": [
        {"id": s["id"], "counts": s["counts"], "expected_behavior": s["expected_behavior"]}
        for s in report["scenarios"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
