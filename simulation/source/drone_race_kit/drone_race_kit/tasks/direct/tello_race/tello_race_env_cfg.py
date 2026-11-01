"""Configuration for the outdoor two-pole Tello race (dirt-field inspired)."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.assets import TELLO_CFG
from action_space_kit.control import TelloCommandCfg


@configclass
class TelloRaceEnvCfg(DirectRLEnvCfg):
    """Single Tello; Pole A start/finish, Pole B turn; one-lap timing.

    Geometry mirrors ``config/race_course.yaml`` in the parent repo (default 4 m
    pole spacing, 0.8 m gate radius). Ground is a flat outdoor plane — stand-in
    for the Harvard i-labs dirt field / parking lot.
    """

    decimation = 2
    episode_length_s = 45.0

    # SDK rc channels: [vx, vy, vz, yaw_rate] in [-1, 1]
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
    # obs: rel A (3) + rel B (3) + lin vel (3) + ang vel (3) + gravity (3) + phase one-hot (5) + elapsed (1)
    observation_space = 21
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    robot_cfg: RigidObjectCfg = TELLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Wider spacing — outdoor course
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=12.0, replicate_physics=True)
    command: TelloCommandCfg = TelloCommandCfg()

    # Course (metres, env-local frame; Z up)
    pole_a_xy = (0.0, 0.0)
    pole_b_xy = (4.0, 0.0)
    pole_height = 1.2
    gate_radius_m = 0.8
    cruise_height = 1.0

    # Spawn: just outside Pole A, facing toward B
    spawn_offset_xy = (-1.2, 0.0)
    spawn_height = 0.9

    # Rewards
    rew_progress = 2.0          # toward next gate
    rew_phase_bonus = 50.0      # leaving A / hitting B / finishing
    rew_lap_complete = 200.0
    rew_time_penalty = -0.05    # per second while racing
    rew_crash = -100.0
    rew_upright = 0.2
    rew_action_rate = -0.01

    # Termination
    min_height = 0.05
    max_height = 3.5
    max_xy_from_course = 8.0
