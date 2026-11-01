"""Two-pole one-lap outdoor race (dirt-field inspired)."""

import gymnasium as gym

from . import agents

gym.register(
    id="AS-Tello-Race-v0",
    entry_point=f"{__name__}.tello_race_env:TelloRaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tello_race_env_cfg:TelloRaceEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
