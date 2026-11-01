import math

import numpy as np
import torch

from drone_race_sim import frames
from drone_race_sim.course import load_course_geometry, map_poles_for_lap_timer
from drone_race_sim.lap import VecLapTimer
from race.lap_timer import LapPhase, LapTimer, load_course


def test_default_fixture_is_proper_rotation():
    R = frames.check_rotation(frames.DEFAULT_R_WORLD_MAP)
    assert abs(np.linalg.det(R) - 1) < 1e-12


def test_map_course_round_trip_and_axes():
    p = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 0.0]])
    assert np.allclose(frames.course_to_map(frames.map_to_course(p)), p)
    # map x right -> course x; map y up -> course z up; map z out of tag -> course -y
    assert np.allclose(frames.map_to_course([1, 0, 0]), [1, 0, 0])
    assert np.allclose(frames.map_to_course([0, 1, 0]), [0, 0, 1])
    assert np.allclose(frames.map_to_course([0, 0, 1]), [0, -1, 0])


def test_rotation_rejects_mirror():
    try:
        frames.check_rotation(np.diag([1.0, 1.0, -1.0]))
    except ValueError:
        return
    raise AssertionError("mirror accepted")


def test_course_geometry_matches_repo_config():
    g = load_course_geometry()
    assert g.pole_a_xy == (0.0, 0.0) and abs(g.pole_b_xy[0] - 4.0) < 1e-9 and g.pole_b_xy[1] == 0.0
    assert g.gate_radius_m == 0.8 and g.require_turn


def test_lap_timer_parity_with_repo_timer():
    """VecLapTimer must reproduce race.lap_timer.LapTimer exactly on random and scripted paths."""
    geom = load_course_geometry()
    course = load_course()
    map_poles = map_poles_for_lap_timer(geom)
    rng = np.random.default_rng(0)
    n, steps = 300, 400
    # random walks that wander between and around both gates, plus a false start and a clean lap
    xy = np.cumsum(rng.normal(0, 0.18, (n, steps, 2)), axis=1) + np.array([0.0, 0.0])
    xy[: n // 3] += rng.uniform(-1, 5, (n // 3, 1, 2)) * np.array([1.0, 0.2])
    clean = np.concatenate([np.linspace([0, 0], [-1.2, 0], 40), np.linspace([-1.2, 0], [4, 0], 120),
                            np.linspace([4, 0], [0, 0], 120)])
    xy[0, : len(clean)] = clean
    xy[1, : len(clean)] = np.concatenate([np.linspace([0, 0], [-1.2, 0], 40), np.linspace([-1.2, 0], [0, 0], 40)] +
                                         [np.tile([0, 0], (len(clean) - 80, 1))])
    vec = VecLapTimer(n, geom.pole_a_xy, geom.pole_b_xy, geom.gate_radius_m, geom.require_turn, dtype=torch.float64)
    refs = [LapTimer(course, map_poles) for _ in range(n)]
    for k in range(steps):
        t = k * 0.05
        ev = vec.update(torch.full((n,), t, dtype=torch.float64), torch.tensor(xy[:, k], dtype=torch.float64))
        for i, ref in enumerate(refs):
            pos_map = tuple(frames.course_to_map([xy[i, k, 0], xy[i, k, 1], 1.0]))
            st = ref.update(t, pos_map)
            assert LapPhase[st["phase"]].value - 1 == int(vec.phase[i]), (i, k)
            assert (st["event"] == "lap_finish") == bool(ev["lap_finish"][i])
            assert (st["event"] == "lap_start") == bool(ev["lap_start"][i])
            assert (st["event"] == "turn") == bool(ev["turn"][i])
            assert (st["event"] == "false_start_reset") == bool(ev["false_start"][i])
    finished = [i for i, r in enumerate(refs) if r.lap_time_s is not None]
    assert 0 in finished and len(finished) >= 1
    for i in finished:
        assert abs(refs[i].lap_time_s - float(vec.lap_time[i])) < 1e-9
    assert not torch.isfinite(vec.lap_time[1])  # false start never produces a lap time
