"""Numpy / single-env views of the torch vector env (gymnasium API)."""

from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np

from .config import EnvCfg
from .env import TelloRaceVecEnv


class SingleEnv(gym.Env):
    """num_envs=1 gym.Env on CPU; used by gym.make("DroneRace-v0") and gymnasium's env_checker."""

    metadata = {"render_modes": []}

    def __init__(self, **cfg_kwargs):
        cfg = dataclasses.replace(EnvCfg(), num_envs=1, device="cpu", autoreset=False, **cfg_kwargs)
        self.vec = TelloRaceVecEnv(cfg)
        self.observation_space = self.vec.single_observation_space
        self.action_space = self.vec.single_action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.vec.reset(seed=seed, options=options)
        return obs[0].numpy().astype(np.float32), {}

    def step(self, action):
        obs, r, te, tr, info = self.vec.step(np.asarray(action, dtype=np.float32)[None])
        return (obs[0].numpy().astype(np.float32), float(r[0]), bool(te[0]), bool(tr[0]),
                {"cause": int(info["cause"][0])})


class NumpyVecWrapper:
    """Same API as TelloRaceVecEnv but numpy in, numpy out (for SB3 adapters or logging)."""

    def __init__(self, env: TelloRaceVecEnv):
        self.env, self.num_envs = env, env.num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs.cpu().numpy(), info

    def step(self, actions):
        obs, r, te, tr, info = self.env.step(actions)
        return obs.cpu().numpy(), r.cpu().numpy(), te.cpu().numpy(), tr.cpu().numpy(), info
