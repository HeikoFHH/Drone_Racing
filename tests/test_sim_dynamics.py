import math
from dataclasses import replace

import torch

from drone_race_sim.config import DynamicsParams
from drone_race_sim.dynamics import TelloDynamics, segment_point_distance

DT = 0.05


def make(n=1, **kw):
    base = dict(a_max=math.inf, deadband=0.0, action_delay_s=0.0, tau_yaw=0.0)
    base.update(kw)
    d = TelloDynamics(n, replace(DynamicsParams(), **base), dtype=torch.float64)
    d.reset(torch.arange(n), torch.tensor([[0.0, 0.0, 1.0]] * n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64))
    return d


def rc(vals, n=1):
    return torch.tensor([vals] * n, dtype=torch.float64)


def dt_t(n=1, dt=DT):
    return torch.full((n,), dt, dtype=torch.float64)


def test_first_order_step_response_reaches_63_percent_at_tau():
    d = make(tau_xy=0.2)
    for _ in range(4):  # 4 * 0.05 = tau
        d.step(rc([0, 100, 0, 0]), dt_t())
    assert abs(float(d.vel[0, 0]) / 0.8 - (1 - math.exp(-1))) < 1e-9


def test_steady_state_gain_and_yaw_sign():
    d = make()
    for _ in range(200):
        d.step(rc([0, 50, 100, 0]), dt_t())
    assert abs(float(d.vel[0, 0]) - 0.4) < 1e-3 and abs(float(d.vel[0, 2]) - 0.6) < 1e-3
    y = make()
    for _ in range(40):
        y.step(rc([0, 0, 0, 100]), dt_t())
    assert abs(float(y.yaw_rate[0]) + math.pi / 3) < 1e-9 and float(y.yaw[0]) < 0  # clockwise rc -> yaw decreases


def test_right_strafe_is_negative_y_when_facing_plus_x():
    d = make()
    for _ in range(100):
        d.step(rc([100, 0, 0, 0]), dt_t())
    assert float(d.vel[0, 1]) < -0.7 and abs(float(d.vel[0, 0])) < 1e-6


def test_deadband_threshold():
    d = make(deadband=8.0)
    for _ in range(100):
        d.step(rc([0, 7.9, 0, 0]), dt_t())
    assert float(d.vel[0, 0]) == 0.0
    for _ in range(100):
        d.step(rc([0, 8.0, 0, 0]), dt_t())
    assert float(d.vel[0, 0]) > 0.05


def test_action_delay_shifts_command_exactly():
    d0, d2 = make(tau_xy=0.0), make(tau_xy=0.0, action_delay_s=2 * DT)
    seq = [0, 0, 100, 100, 100, 100]
    v0, v2 = [], []
    for u in seq:
        d0.step(rc([0, u, 0, 0]), dt_t()); v0.append(float(d0.vel[0, 0]))
        d2.step(rc([0, u, 0, 0]), dt_t()); v2.append(float(d2.vel[0, 0]))
    assert v2[:2] == [0.0, 0.0] and v2[2:] == v0[:-2]


def test_acceleration_clamp():
    d = make(tau_xy=0.0, a_max=1.0)
    d.step(rc([0, 100, 0, 0]), dt_t())
    assert abs(float(d.vel[0, 0]) - 1.0 * DT) < 1e-12


def test_exact_lag_semigroup_for_velocity():
    a, b = make(tau_xy=0.2), make(tau_xy=0.2)
    a.step(rc([0, 100, 0, 0]), dt_t(dt=0.1))
    for _ in range(2):
        b.step(rc([0, 100, 0, 0]), dt_t(dt=0.05))
    assert abs(float(a.vel[0, 0]) - float(b.vel[0, 0])) < 1e-12


def test_batch_independence_and_determinism():
    d = make(4)
    cmds = torch.tensor([[0, 0, 0, 0], [0, 100, 0, 0], [50, 0, 20, 0], [0, 0, 0, 80]], dtype=torch.float64)
    for _ in range(30):
        d.step(cmds, dt_t(4))
    for i in range(4):
        s = make(1)
        for _ in range(30):
            s.step(cmds[i : i + 1], dt_t(1))
        assert torch.allclose(s.pos[0], d.pos[i], atol=1e-12) and abs(float(s.yaw[0]) - float(d.yaw[i])) < 1e-12


def test_ground_clamp_and_segment_distance():
    d = make()
    for _ in range(100):
        d.step(rc([0, 0, -100, 0]), dt_t())
    assert float(d.pos[0, 2]) == 0.0
    p0, p1, q = torch.tensor([[0.0, 0.0]]), torch.tensor([[4.0, 0.0]]), torch.tensor([[2.0, 0.3]])
    assert abs(float(segment_point_distance(p0, p1, q)) - 0.3) < 1e-6
    assert abs(float(segment_point_distance(p0, p1, torch.tensor([[6.0, 0.0]]))) - 2.0) < 1e-6
