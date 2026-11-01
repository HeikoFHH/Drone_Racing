import torch

from drone_race_sim.controllers import (FaultyController, GuardCfg, GuardedController, MaudeController, PolicyConfig,
                                        PolicyCfg, RaceController)
from drone_race_sim.rollout import rollout

SEEDS = range(6)


def race(max_rc=35):
    return RaceController(PolicyCfg(max_rc=max_rc))


def test_guard_limits_come_from_maudes_policy_config():
    c = PolicyConfig()
    g = GuardCfg(profile="classic", policy=c)
    assert (g.pose_hold_s, g.pose_land_s) == (c.max_pose_age_s, c.land_after_pose_age_s)
    rl = GuardCfg(profile="rl", policy=c).backup_policy_cfg()
    assert rl.max_position_std_m == 0.5 and rl.land_after_pose_age_s == 4.0
    assert rl.rc_limit == c.rc_limit and rl.waypoints_w == c.waypoints_w     # same route and caps


def test_guard_is_transparent_when_primary_is_healthy():
    plain = rollout(race(), "baseline", SEEDS)
    guarded = rollout(GuardedController(race()), "baseline", SEEDS)
    assert plain.outcome == guarded.outcome == ["lap"] * len(SEEDS)
    assert guarded.takeover == [None] * len(SEEDS)
    assert guarded.lap_time_s == plain.lap_time_s


def test_nan_output_hands_over_to_maude_route_and_still_laps():
    r = rollout(GuardedController(FaultyController(race(), "nan", t_fault=4.0)), "baseline", SEEDS, horizon_s=90)
    assert all(t is not None and t[1] == "invalid_primary" for t in r.takeover)
    assert r.outcome.count("lap") == len(SEEDS)


def test_stalled_primary_is_replaced_after_the_watchdog():
    g = GuardedController(FaultyController(race(), "stall", t_fault=3.0), GuardCfg(stall_s=6.0))
    r = rollout(g, "baseline", SEEDS, horizon_s=120)
    assert all(t is not None and t[1] == "stalled" for t in r.takeover) and r.outcome.count("lap") == len(SEEDS)
    # without the guard the stalled primary never finishes
    assert rollout(FaultyController(race(), "stall", 3.0), "baseline", SEEDS, horizon_s=30).outcome == ["timeout"] * len(SEEDS)


def test_runaway_and_dive_end_in_landing_not_crash():
    for fault in ("runaway", "dive"):
        bare = rollout(FaultyController(race(), fault, t_fault=3.0), "baseline", SEEDS)
        assert all(o.startswith("crash") for o in bare.outcome), (fault, bare.outcome)
        safe = rollout(GuardedController(FaultyController(race(), fault, t_fault=3.0)), "baseline", SEEDS)
        assert safe.outcome == ["landed"] * len(SEEDS), (fault, safe.outcome)
        assert set(safe.land_reason) == {"geofence"}


def test_guard_with_realistic_sensing_never_crashes_a_healthy_policy():
    r = rollout(GuardedController(race()), "baseline", SEEDS, sensing="realistic")
    assert not any(o.startswith("crash") for o in r.outcome)
