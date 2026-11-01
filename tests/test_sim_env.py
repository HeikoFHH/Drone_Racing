import dataclasses

import gymnasium as gym
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

import drone_race_sim  # noqa: F401  (registers DroneRace-v0)
from drone_race_sim import EnvCfg, TelloRaceVecEnv
from drone_race_sim.env import CAUSE_TIMEOUT
from policy_integration.policy import Observation


def test_shapes_and_reset_in_gate():
    env = TelloRaceVecEnv(EnvCfg(num_envs=64))
    obs, info = env.reset(seed=3)
    assert obs.shape == (64, env.obs_dim) == (64, 112) and info["priv"].shape == (64, 16)
    assert (env.dyn.pos[:, :2].norm(dim=-1) <= 0.6 + 1e-6).all() and (env.lap.phase == 1).all()  # AT_START


def test_seed_determinism():
    runs = []
    for _ in range(2):
        env = TelloRaceVecEnv(EnvCfg(num_envs=16, dr_level=1.0))
        env.reset(seed=7)
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            o, r, *_ = env.step(torch.rand(16, 4, generator=g) * 2 - 1)
        runs.append((o, r))
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])


def test_no_nan_random_actions_with_autoreset_and_final_obs():
    env = TelloRaceVecEnv(EnvCfg(num_envs=256, dr_level=1.0, seed=1))
    env.reset()
    g = torch.Generator().manual_seed(0)
    seen_done = 0
    for _ in range(600):
        o, r, te, tr, info = env.step(torch.rand(256, 4, generator=g) * 2 - 1)
        assert torch.isfinite(o).all() and torch.isfinite(r).all()
        d = te | tr
        seen_done += int(d.sum())
        assert torch.equal(d, info["done"])
        if d.any():  # final obs differs from the reset obs that was returned
            assert not torch.equal(info["final_obs"][d], o[d])
    assert seen_done > 0


def test_timeout_truncates_and_reward_components_finite():
    env = TelloRaceVecEnv(EnvCfg(num_envs=4, episode_s=1.0, enable_terminations=True))
    env.reset(seed=0)
    for k in range(20):
        o, r, te, tr, info = env.step(torch.zeros(4, 4))
    assert info["done"].all() and (info["cause"] == CAUSE_TIMEOUT).all() and tr.all()
    assert all(torch.isfinite(v).all() for v in info["reward_components"].values())


def test_obs_matches_policy_contract_semantics():
    """Position/velocity used to build observations are accepted by the repo's Observation schema."""
    env = TelloRaceVecEnv(EnvCfg(num_envs=2))
    env.reset(seed=0)
    p, v, y = env.dyn.pos[0].tolist(), env.dyn.vel[0].tolist(), float(env.dyn.yaw[0])
    obs = Observation(0.0, 0.0, tuple(p), tuple(v), y, 0.03, 0.05, 75.0, True, True, True)
    assert obs.frame == "course_flu_v1"


def test_single_env_passes_gymnasium_checker():
    env = gym.make("DroneRace-v0")
    check_env(env.unwrapped, skip_render_check=True)


def test_gym_make_vec_both_modes():
    v = gym.make_vec("DroneRace-v0", num_envs=6, vectorization_mode="vector_entry_point", sensing="realistic")
    o, _ = v.reset(seed=0)
    assert o.shape == (6, 112) and v.unwrapped.cfg.sensing == "realistic"
    s = gym.make_vec("DroneRace-v0", num_envs=3, vectorization_mode="sync")
    o, _ = s.reset(seed=0)
    o, r, te, tr, info = s.step(s.action_space.sample())
    assert o.shape == (3, 112) and isinstance(o, np.ndarray)


def test_stable_baselines3_can_train_on_it():
    sb3 = __import__("pytest").importorskip("stable_baselines3")
    from drone_race_sim.wrappers import SB3VecEnv

    venv = SB3VecEnv.create(TelloRaceVecEnv(EnvCfg(num_envs=16, seed=0)))
    model = sb3.PPO("MlpPolicy", venv, n_steps=16, batch_size=64, n_epochs=1, device="cpu", seed=0)
    model.learn(total_timesteps=16 * 16 * 2)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (16, 4) and np.isfinite(action).all()
