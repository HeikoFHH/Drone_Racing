"""The torch sim must reproduce policy_integration/stress_sim.py outcomes (SIMULATION_PLAN.md M1 exit)."""

import pytest

from drone_race_sim.baselines.baseline_policy_adapter import run_baseline
from policy_integration.stress_sim import SCENARIOS, run

EXPECTED = {"baseline": 20, "mild-wind": 20, "strong-wind": 0, "delay-150": 20, "delay-300": 0,
            "pose-loss": 0, "link-loss": 0, "deadband": 6}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario_matches_stress_sim(scenario):
    seeds = range(20)
    mine = run_baseline(scenario, seeds)
    for m, s in zip(mine, seeds):
        ref = run(scenario, s)
        assert (m["outcome"], m["reason"]) == (ref["outcome"], ref["reason"])
        assert abs(m["elapsed_s"] - ref["elapsed_s"]) < 1e-3 and m["peak_abs_rc"] == ref["peak_abs_rc"]
    assert sum(r["outcome"] == "complete" for r in mine) == EXPECTED[scenario.id]
