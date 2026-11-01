import math
from dataclasses import replace

import torch

from drone_race_sim import EnvCfg, TelloRaceVecEnv
from drone_race_sim.sensors import SensorModel, load_sensor_cfg

DT = 0.05
CFG, _ = load_sensor_cfg()


def model(n=1, **kw):
    return SensorModel(n, (0.0, 0.0), (4.0, 0.0), replace(CFG, **kw), DT)


def vis(m, pos, yaw, k):
    p = torch.tensor([pos], dtype=torch.float32)
    y = torch.tensor([yaw], dtype=torch.float32)
    return bool(m._tag_view(k, p, y)[0][0])


def test_fov_visibility_geometry():
    m = model()
    assert vis(m, [0.0, 0.0, 1.0], 0.0, 1)                      # tag B dead ahead at 4 m, facing the drone
    assert not vis(m, [0.0, 0.0, 1.0], math.pi, 1)              # facing away
    assert not vis(m, [3.0, 0.54, 1.0], 0.0, 1)                 # 28 deg off axis: outside the 55 deg HFOV
    assert vis(m, [3.0, 0.3, 1.0], 0.0, 1)                      # 17 deg off axis: inside
    assert not vis(m, [3.0, 0.0, 2.5], 0.0, 1)                  # too high: outside the vertical FOV


def test_tag_is_one_sided_and_range_limited():
    m = model()
    assert not vis(m, [5.0, 0.0, 1.0], math.pi, 1)              # behind tag B (it faces A), looking at it
    assert vis(m, [0.0, 0.0, 1.0], 0.0, 1)
    far = model(tag_size_m=0.13208)
    assert not vis(far, [-11.0, 0.0, 1.0], 0.0, 0)              # tag A is behind this view anyway
    assert not vis(far, [-6.5, 0.0, 1.0], 0.0, 1)               # 10.5 m from B: apparent side < 12 px


def test_steep_viewing_angle_rejected():
    m = model(max_view_deg=70.0)
    # tag B faces -X; a drone at (4.2, ...) looking toward B from the side sees it at a steep angle
    assert not vis(m, [4.5, 1.0, 1.0], math.atan2(-1.0, -0.5), 1)


def test_latency_is_exactly_k_ticks():
    m = model(video_delay_s=0.15, detect_prob=1.0, detector_hz=1000.0, depth_noise_per_m2=0.0, lateral_noise_rad=0.0,
              range_scale_sigma=0.0)
    idx = torch.arange(1)
    pos = torch.tensor([[0.0, 0.0, 1.0]])
    m.reset(idx, pos, torch.zeros(1, 3), torch.zeros(1))
    path = [torch.tensor([[0.04 * k, 0.0, 1.0]]) for k in range(1, 12)]
    for k, p in enumerate(path):
        m.step(p, torch.zeros(1, 3), torch.zeros(1), DT)
        if k >= 4 and bool(m.fix_now[0]):
            assert abs(float(m.fix_pos[0, 0]) - float(path[k - 3][0, 0])) < 1e-4   # 0.15 s / 0.05 s = 3 ticks


def test_fix_noise_grows_with_range_squared():
    stds = {}
    for x in (-1.0, -3.0):   # tag B at x=4: ranges 5 m and 7 m
        m = model(64, detect_prob=1.0, detector_hz=1000.0, range_scale_sigma=0.0)
        idx = torch.arange(64)
        pos = torch.tensor([[x, 0.0, 1.0]] * 64)
        m.reset(idx, pos, torch.zeros(64, 3), torch.zeros(64))
        errs = []
        for _ in range(25):
            m.step(pos, torch.zeros(64, 3), torch.zeros(64), DT)
            errs.append((m.fix_pos[:, 0] - pos[:, 0]))
        stds[x] = torch.cat(errs[8:]).std().item()
    assert stds[-3.0] > 1.5 * stds[-1.0]            # (7/5)^2 = 1.96 in depth


def test_estimator_tracks_then_dead_reckons_with_growing_std():
    env = TelloRaceVecEnv(EnvCfg(num_envs=32, sensing="realistic", autoreset=False, seed=1))
    env.reset(seed=1, options={"pos": torch.tensor([[1.0, 0.0, 1.0]] * 32), "yaw": torch.zeros(32)})
    for _ in range(60):
        env.step_rc(torch.zeros(32, 4))
    s = env.sensed
    assert (s["pos"][:, :2] - env.dyn.pos[:, :2]).norm(dim=-1).mean() < 0.15 and not s["lost"].any()
    env.reset(seed=1, options={"pos": torch.tensor([[1.0, 0.0, 1.0]] * 32), "yaw": torch.full((32,), math.pi)})
    for _ in range(80):                                 # facing away from both tags is impossible to see (A behind? no: A ahead)
        env.step_rc(torch.zeros(32, 4))
    # yaw pi: tag A (x=0, facing +X) is ahead and visible; rotate to look at the empty side instead
    env.reset(seed=1, options={"pos": torch.tensor([[2.0, 3.0, 1.0]] * 32), "yaw": torch.full((32,), math.pi / 2)})
    std0 = None
    for k in range(120):
        env.step_rc(torch.zeros(32, 4))
        if k == 20:
            std0 = env.sensed["pos_std"].mean().item()
    assert env.sensed["lost"].all() and env.sensed["pos_std"].mean() > std0
    assert (env.sensed["pos"][:, :2] - env.dyn.pos[:, :2]).norm(dim=-1).mean() < 0.6   # hovering: drift stays small


def test_realistic_obs_has_tag_and_estimator_slots():
    env = TelloRaceVecEnv(EnvCfg(num_envs=8, sensing="realistic", seed=0))
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(torch.zeros(8, 4))
    frame = obs[:, 75:100]     # newest of 4 stacked frames
    assert torch.isfinite(obs).all() and frame[:, 9:13].abs().sum() + frame[:, 13:17].abs().sum() > 0   # tag slots filled
