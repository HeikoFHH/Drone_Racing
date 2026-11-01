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


class SB3VecEnv:
    """stable-baselines3 ``VecEnv`` view of TelloRaceVecEnv (numpy, same-step autoreset, ``terminal_observation``).

    Subclasses ``stable_baselines3.common.vec_env.VecEnv`` lazily so that SB3 stays an optional dependency:
    ``SB3VecEnv.create(env)`` returns the real VecEnv instance.
    """

    @staticmethod
    def create(env: TelloRaceVecEnv):
        from stable_baselines3.common.vec_env import VecEnv

        class _V(VecEnv):
            def __init__(self, inner):
                super().__init__(inner.num_envs, inner.single_observation_space, inner.single_action_space)
                self.inner, self._actions = inner, None

            def reset(self):
                obs, _ = self.inner.reset()
                return obs.cpu().numpy()

            def step_async(self, actions):
                self._actions = actions

            def step_wait(self):
                obs, rew, te, tr, info = self.inner.step(self._actions)
                done = info["done"].cpu().numpy()
                infos = [{} for _ in range(self.num_envs)]
                final = info["final_obs"].cpu().numpy()
                for i in np.nonzero(done)[0]:
                    infos[i]["terminal_observation"] = final[i]
                    infos[i]["TimeLimit.truncated"] = bool(tr[i]) and not bool(te[i])
                return obs.cpu().numpy(), rew.cpu().numpy(), done, infos

            def close(self):
                pass

            def get_attr(self, attr_name, indices=None):
                return [getattr(self.inner, attr_name)] * self.num_envs

            def set_attr(self, attr_name, value, indices=None):
                setattr(self.inner, attr_name, value)

            def env_method(self, method_name, *args, indices=None, **kwargs):
                return [getattr(self.inner, method_name)(*args, **kwargs)] * self.num_envs

            def env_is_wrapped(self, wrapper_class, indices=None):
                return [False] * self.num_envs

            def seed(self, seed=None):
                return [seed] * self.num_envs

        return _V(env)
