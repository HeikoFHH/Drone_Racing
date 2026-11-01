"""Mac-native vectorized simulator for the two-pole Tello race (docs/SIMULATION_PLAN.md)."""

import gymnasium as gym

from . import _repo  # noqa: F401  (makes race/ and policy_integration/ importable)
from .config import DynamicsParams, EnvCfg
from .env import TelloRaceVecEnv

gym.register(id="DroneRace-v0", entry_point="drone_race_sim.wrappers:SingleEnv",
             vector_entry_point="drone_race_sim.env:TelloRaceVecEnv")

__all__ = ["TelloRaceVecEnv", "EnvCfg", "DynamicsParams"]
