"""Safety tests for race/homing_policy.py (synthetic inputs, no rendering, run in seconds).

    .venv-sim/bin/python -m pytest race/test_homing_policy.py -q
"""

from __future__ import annotations

import math
import random

import pytest

from race.homing_policy import HomingCfg, HomingObs, HomingPolicy, Sighting

DT = 0.05
H = 1.2  # height the policy holds


def make(**kw) -> HomingPolicy:
    return HomingPolicy(HomingCfg(height_m=H, **kw))


def tick(pol, t, sightings=(), h=H, yaw=0.0):
    return pol.step(HomingObs(t=t, sightings=list(sightings), height_m=h, yaw_deg=yaw), DT)


def b(r, bearing=0.0, tag=2):
    return Sighting(tag, r, bearing)


def check_contract(rc, info, cfg):
    assert len(rc) == 4 and all(isinstance(v, int) for v in rc)
    assert all(-100 <= v <= 100 for v in rc)
    assert rc[0] == 0, "never uses the sideways stick"
    assert abs(rc[2]) <= cfg.max_up_rc
    assert abs(rc[3]) <= max(cfg.max_yaw_rc, cfg.turn_yaw_rc, cfg.spin_yaw_rc)
    assert info["mode"] in ("go", "land", "done")
    if info["mode"] in ("land", "done"):
        assert rc[:2] == (0, 0) and rc[3] == 0


def test_output_contract_under_fuzz():
    rng = random.Random(0)
    bad_ranges = [float("nan"), float("inf"), -3.0, 0.0, 1e-6]
    for episode in range(30):
        pol = make()
        for k in range(600):
            sightings = []
            for _ in range(rng.randint(0, 3)):
                r = rng.choice(bad_ranges) if rng.random() < 0.2 else rng.uniform(0.2, 90)
                sightings.append(Sighting(rng.choice([1, 2, 7, 99]), r, rng.uniform(-60, 60)))
            h = rng.choice([None, float("nan"), rng.uniform(0, 3)]) if rng.random() < 0.1 else H + rng.gauss(0, 0.05)
            yaw = None if rng.random() < 0.05 else rng.uniform(-180, 180)
            rc, info = pol.step(HomingObs(k * DT, sightings, h, yaw), DT)
            check_contract(rc, info, pol.cfg)


def test_never_flies_forward_inside_5m_of_target():
    pol = make()
    tick(pol, 0.0)                               # climbed already -> looks for the tag
    r, t = 20.0, 0.05
    while r > 3.0:
        rc, info = tick(pol, t, [b(r)])
        if r < pol.cfg.stop_range_m:
            assert rc[1] <= 0, f"forward {rc[1]} at {r:.2f} m"
        r -= 0.08
        t += DT


def test_keep_out_applies_to_any_tag_not_just_the_target():
    pol = make()
    tick(pol, 0.0)
    for k in range(1, 60):
        rc, info = tick(pol, k * DT, [b(30.0 - 0.07 * k), Sighting(7, 3.0, 5.0)])
        assert rc[1] <= 0
    assert info["keep_out"] and 7 in info["unexpected_tags"]


def test_single_range_spike_does_not_stop_or_jump():
    pol = make()
    tick(pol, 0.0)
    for k in range(1, 200):
        r = 25.0 - 0.07 * k
        rc, info = tick(pol, k * DT, [b(5.2 if k == 100 else r)])
        assert info["state"] in ("TRACK", "SPIN", "BLIND"), info["state"]


@pytest.mark.parametrize("h", [None, float("nan")])
def test_missing_height_telemetry_aborts(h):
    pol = make()
    for k in range(80):
        rc, info = tick(pol, k * DT, h=h)
    assert info["mode"] == "land" and info["abort_reason"] == "no_height_telemetry"
    assert rc == (0, 0, 0, 0)


def test_height_reading_stuck_low_aborts_instead_of_climbing_forever():
    pol = make()
    for k in range(200):
        rc, info = tick(pol, k * DT, h=0.3)        # stuck reading: policy keeps asking to climb
        if info["mode"] == "land":
            break
    assert info["abort_reason"] == "height_not_responding"
    assert k * DT < 6.0


def test_too_high_aborts():
    pol = make()
    rc, info = tick(pol, 0.0, h=3.0)
    assert info["mode"] == "land" and info["abort_reason"] == "too_high"


def test_abort_is_sticky():
    pol = make()
    tick(pol, 0.0, h=3.0)
    for k in range(1, 50):
        rc, info = tick(pol, k * DT, [b(20.0)])
        assert rc == (0, 0, 0, 0) and info["mode"] == "land"


def test_no_tag_ever_stops_short_of_the_pole_then_lands():
    """Tag B never shows up: blind distance must stay short of the 5 m line, then land."""
    pol = make()
    t, flown = 0.0, 0.0
    while t < 400:
        rc, info = tick(pol, t)
        flown += rc[1] * pol.cfg.mps_per_rc * DT      # what the drone would actually fly
        if info["mode"] == "land":
            break
        t += DT
    assert info["mode"] == "land" and info["abort_reason"] in ("tag_not_found", "session_timeout")
    leg = pol.cfg.course_m - pol.cfg.start_offset_m
    # even if the drone is 30% faster than the policy thinks, it must stay outside the 5 m line
    assert flown * 1.3 <= leg - pol.cfg.stop_range_m, f"flew {flown:.1f} m (x1.3) of a {leg:.1f} m leg blind"


def test_dead_yaw_telemetry_aborts():
    pol = make()
    t = 0.0
    while t < 400:
        rc, info = tick(pol, t, yaw=None)
        if info["mode"] == "land":
            break
        t += DT
    assert info["mode"] == "land" and info["abort_reason"] == "no_yaw_telemetry" and t < 3.0


def test_session_timeout():
    pol = make(max_session_s=10.0)
    for k in range(400):
        rc, info = tick(pol, k * DT, [b(60.0 - 0.0 * k)])   # a tag that never gets closer
        if info["mode"] == "land":
            break
    assert info["abort_reason"] in ("session_timeout", "range_not_shrinking")
    assert k * DT <= 10.1


def test_range_that_stops_shrinking_aborts():
    pol = make()
    tick(pol, 0.0)
    for k in range(1, 400):
        rc, info = tick(pol, k * DT, [b(20.0)])             # flying at it, range never changes
        if info["mode"] == "land":
            break
    assert info["abort_reason"] == "range_not_shrinking"


# --- start / stop switch (race/homing_control.py) ----------------------------------------
from race.homing_control import FLOAT, PolicySwitch  # noqa: E402


def sw_tick(sw, t, sightings=(), h=H, yaw=0.0):
    return sw.step(HomingObs(t=t, sightings=list(sightings), height_m=h, yaw_deg=yaw), DT)


def test_switch_floats_until_started():
    sw = PolicySwitch(HomingCfg(height_m=H))
    for k in range(20):
        rc, info = sw_tick(sw, k * DT, [b(30.0)])
        assert rc == FLOAT and info["mode"] == "float" and not info["running"]


def test_switch_stop_floats_on_the_next_tick_and_resume_continues():
    sw = PolicySwitch(HomingCfg(height_m=H))
    sw.start()
    t = 0.0
    for _ in range(40):                                   # flying toward B
        rc, info = sw_tick(sw, t, [b(30.0 - 0.07 * t / DT)])
        t += DT
    assert info["mode"] == "go" and rc[1] > 0
    state_before = info["state"]
    sw.stop()
    rc, info = sw_tick(sw, t)
    assert rc == FLOAT and info["mode"] == "float"
    for _ in range(600):                                  # 30 s paused: no timeouts may fire
        t += DT
        rc, info = sw_tick(sw, t, h=None, yaw=None)       # even with telemetry gone while floating
        assert rc == FLOAT
    sw.start()                                            # resume, same mission
    rc, info = sw_tick(sw, t + DT, [b(27.0)])
    assert info["mode"] == "go" and info["state"] in (state_before, "TRACK")


def test_switch_new_mission_starts_over():
    sw = PolicySwitch(HomingCfg(height_m=H))
    sw.start()
    sw_tick(sw, 0.0, h=3.0)                               # trips "too_high" -> land
    rc, info = sw_tick(sw, DT)
    assert info["mode"] == "land"
    sw.start()                                            # plain start resumes: still aborted
    rc, info = sw_tick(sw, 2 * DT)
    assert info["mode"] == "land"
    sw.start(new_mission=True)
    rc, info = sw_tick(sw, 3 * DT)
    assert info["mode"] == "go"


def test_switch_buttons_from_another_thread():
    import threading
    sw = PolicySwitch(HomingCfg(height_m=H))
    presses = threading.Thread(target=lambda: [(sw.start(), sw.stop()) for _ in range(500)])
    presses.start()
    for k in range(500):
        rc, info = sw_tick(sw, k * DT, [b(30.0)])
        assert info["mode"] in ("float", "go")
    presses.join()
    sw.stop()
    assert sw_tick(sw, 1000.0)[0] == FLOAT


def test_start_works_with_the_tag_in_front_whichever_it_is():
    """policy start new facing tag 1 instead of tag 2: tag 1 becomes the first target, and the
    course length comes from what was measured, not the configured prior."""
    pol = make(course_m=73.0)
    tick(pol, 0.0)
    rc, info = tick(pol, DT, [Sighting(1, 12.0, 0.0)])
    assert info["first_target"] == 1 and info["target"] == 1 and info["state"] == "TRACK"
    assert abs(info["course_est_m"] - 13.0) < 0.5            # 12 m seen + 1 m start offset


def test_nothing_in_front_spins_to_look_first():
    pol = make()
    tick(pol, 0.0)
    rc, info = tick(pol, DT)
    assert info["state"] == "SPIN" and rc[1] == 0 and rc[3] != 0


def test_tag_we_took_off_next_to_is_not_the_first_target():
    """Spinning at takeoff sees the start tag 1 m away: ignore it, keep looking for the other."""
    pol = make()
    tick(pol, 0.0)
    rc, info = tick(pol, DT, [Sighting(1, 1.2, 10.0)])
    assert info["state"] == "SPIN" and info["first_target"] == 2
    rc, info = tick(pol, 2 * DT, [Sighting(2, 30.0, -5.0)])
    assert info["state"] == "TRACK" and info["target"] == 2


# --- stop zone: stop >= 5 m from the tag, and only turn once settled there --------------------
def _approach_to_hold(pol, t=0.0, r0=9.0):
    tick(pol, t)
    r = r0
    while True:
        t += DT
        rc, info = tick(pol, t, [b(r)])
        if info["state"] == "HOLD":
            return t, r
        r -= 0.05
        assert r > 4.0, "never stopped"


def test_stops_inside_the_stop_zone_not_closer_than_5m():
    pol = make()
    t, r = _approach_to_hold(pol)
    assert pol.cfg.stop_range_m <= r <= pol.cfg.stop_range_m + pol.cfg.stop_buffer_m + pol.cfg.stop_tol_m


def test_turns_only_after_settling_in_the_zone():
    pol = make()
    t, r = _approach_to_hold(pol)
    states = []
    aim = pol.cfg.stop_range_m + pol.cfg.stop_buffer_m
    for k in range(1, 40):                     # hold steady at the stop point for 2 s
        rc, info = tick(pol, t + k * DT, [b(aim)])
        states.append(info["state"])
    assert "TURN" in states
    first_turn = states.index("TURN")
    assert first_turn * DT >= pol.cfg.hold_s - DT, "turned before settling for hold_s"


def test_no_turn_if_the_tag_is_lost_while_holding():
    pol = make()
    t, r = _approach_to_hold(pol)
    for k in range(1, 80):                     # tag gone for 4 s
        rc, info = tick(pol, t + k * DT)
        assert info["state"] != "TURN"
        assert rc[1] <= 0                      # never creeps forward blind next to the pole
    assert info["state"] == "SPIN"             # looks for it in place instead


def test_no_turn_while_drifting_outside_the_zone():
    pol = make()
    t, r = _approach_to_hold(pol)
    for k in range(1, 60):                     # range wanders to 6.5 m: outside the zone
        rc, info = tick(pol, t + k * DT, [b(6.5)])
        assert info["state"] != "TURN"


def test_backs_off_if_inside_5m():
    pol = make()
    t, r = _approach_to_hold(pol)
    for k in range(1, 10):
        rc, info = tick(pol, t + k * DT, [b(4.6)])
    assert rc[1] < 0 and info["state"] == "HOLD"


def test_tag_a_already_in_view_at_the_turn_does_not_crash():
    pol = make()
    t, r = _approach_to_hold(pol)
    for k in range(1, 200):                    # hold at B, then turn with tag A in view the whole time
        rc, info = tick(pol, t + k * DT, [b(5.15), Sighting(1, 60.0, 0.0)], yaw=(k * 3.0) % 360 - 180)
        check_contract(rc, info, pol.cfg)


def test_stop_zone_is_5_to_5_3_m():
    c = HomingCfg()
    assert c.stop_range_m == 5.0 and abs(c.stop_range_m + c.stop_buffer_m + c.stop_tol_m - 5.3) < 1e-9


def test_turn_is_a_fixed_180_then_keeps_moving():
    pol = make()
    t, r = _approach_to_hold(pol, r0=30.0)    # 30 m course: tag A is ~25 m away after the stop
    aim = pol.cfg.stop_range_m + pol.cfg.stop_buffer_m
    yaw, k = 30.0, 0
    while True:                                # settle, then follow the turn with a simulated yaw
        k += 1
        rc, info = tick(pol, t + k * DT, [b(aim)] if info_state(pol) == "HOLD" else [], yaw=yaw)
        if info["state"] == "TURN":
            yaw = ((yaw + rc[3] * 0.6 * DT * 3 + 180) % 360) - 180   # yaw follows the stick
        if info["leg"] == "to_A":
            break
        assert k < 2000
    assert abs(abs(((pol.leg_heading - 30.0 + 180) % 360) - 180) - 180.0) < 1e-6   # exactly 180 from 30 deg
    assert info["state"] == "BLIND"            # no spin after the turn: fly on and look
    rc, info = tick(pol, t + (k + 1) * DT, [], yaw=yaw)
    assert rc[1] > 0


def info_state(pol):
    return pol.state


def test_default_flight_height_is_0_8_m():
    assert HomingCfg().height_m == 0.8
