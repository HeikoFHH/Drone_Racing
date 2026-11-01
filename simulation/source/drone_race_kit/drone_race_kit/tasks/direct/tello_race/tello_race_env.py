"""Outdoor two-pole one-lap race — dirt-field-inspired flat course.

Lap phases (same contract as ``race/lap_timer.py``):
  AT_START → leave Pole A (timer start / OUTBOUND) → enter Pole B (TURNED)
  → re-enter Pole A (FINISHED).
"""

from __future__ import annotations

import gymnasium as gym
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from action_space_kit.assets import TELLO_MASS
from action_space_kit.control import TelloVelocityController

from .tello_race_env_cfg import TelloRaceEnvCfg

# Lap phase indices (match one-hot in observations)
PHASE_AT_START = 0
PHASE_OUTBOUND = 1
PHASE_TURNED = 2
PHASE_FINISHED = 3
PHASE_IDLE = 4
N_PHASES = 5


class TelloRaceEnv(DirectRLEnv):
    cfg: TelloRaceEnvCfg

    def __init__(self, cfg: TelloRaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        inertia_zz = float(self._robot.root_physx_view.get_inertias()[0].reshape(3, 3)[2, 2])
        self._controller = TelloVelocityController(
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            mass=TELLO_MASS,
            inertia_zz=inertia_zz,
            device=self.device,
        )
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        origins = self.scene.env_origins
        self._pole_a = origins.clone()
        self._pole_a[:, 0] += self.cfg.pole_a_xy[0]
        self._pole_a[:, 1] += self.cfg.pole_a_xy[1]
        self._pole_a[:, 2] += self.cfg.pole_height
        self._pole_b = origins.clone()
        self._pole_b[:, 0] += self.cfg.pole_b_xy[0]
        self._pole_b[:, 1] += self.cfg.pole_b_xy[1]
        self._pole_b[:, 2] += self.cfg.pole_height

        self._phase = torch.full((self.num_envs,), PHASE_IDLE, dtype=torch.long, device=self.device)
        self._was_in_a = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._lap_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._lap_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_left_a = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_hit_b = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._just_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot_cfg)
        # Tan/dirt-ish ground as a stand-in for the outdoor lot
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(color=(0.55, 0.48, 0.35)),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["robot"] = self._robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.85, 0.85, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (0.15, 0.15, 1.2)
        marker_cfg.prim_path = "/Visuals/pole_a"
        self._pole_a_markers = VisualizationMarkers(marker_cfg)
        marker_b = CUBOID_MARKER_CFG.copy()
        marker_b.markers["cuboid"].size = (0.15, 0.15, 1.2)
        marker_b.prim_path = "/Visuals/pole_b"
        self._pole_b_markers = VisualizationMarkers(marker_b)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        force_w, torque_b = self._controller.compute(
            actions=self._actions,
            lin_vel_w=self._robot.data.root_lin_vel_w,
            ang_vel_b=self._robot.data.root_ang_vel_b,
            projected_gravity_b=self._robot.data.projected_gravity_b,
            dt=self.physics_dt,
        )
        self._robot.set_external_force_and_torque(forces=force_w, torques=torque_b, body_ids=[0])

    def _gate_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._robot.data.root_pos_w
        d_a = torch.linalg.norm(pos[:, :2] - self._pole_a[:, :2], dim=-1)
        d_b = torch.linalg.norm(pos[:, :2] - self._pole_b[:, :2], dim=-1)
        in_a = d_a <= self.cfg.gate_radius_m
        in_b = d_b <= self.cfg.gate_radius_m
        return in_a, in_b

    def _update_lap(self) -> None:
        in_a, in_b = self._gate_masks()
        self._just_left_a.zero_()
        self._just_hit_b.zero_()
        self._just_finished.zero_()

        # IDLE / fresh → AT_START when inside A
        idle = self._phase == PHASE_IDLE
        self._phase = torch.where(idle & in_a, torch.full_like(self._phase, PHASE_AT_START), self._phase)

        # Leave A → OUTBOUND (lap starts)
        at_start = self._phase == PHASE_AT_START
        left_a = at_start & self._was_in_a & ~in_a
        self._just_left_a = left_a
        self._phase = torch.where(left_a, torch.full_like(self._phase, PHASE_OUTBOUND), self._phase)
        self._lap_active = torch.where(left_a, torch.ones_like(self._lap_active), self._lap_active)
        self._lap_elapsed = torch.where(left_a, torch.zeros_like(self._lap_elapsed), self._lap_elapsed)

        # Hit B → TURNED
        outbound = self._phase == PHASE_OUTBOUND
        hit_b = outbound & in_b
        self._just_hit_b = hit_b
        self._phase = torch.where(hit_b, torch.full_like(self._phase, PHASE_TURNED), self._phase)

        # False start: back in A while outbound without B
        false_start = outbound & in_a
        self._phase = torch.where(false_start, torch.full_like(self._phase, PHASE_AT_START), self._phase)
        self._lap_active = torch.where(false_start, torch.zeros_like(self._lap_active), self._lap_active)

        # Finish: TURNED + enter A
        turned = self._phase == PHASE_TURNED
        finished = turned & in_a
        self._just_finished = finished
        self._phase = torch.where(finished, torch.full_like(self._phase, PHASE_FINISHED), self._phase)
        self._lap_active = torch.where(finished, torch.zeros_like(self._lap_active), self._lap_active)

        self._lap_elapsed = torch.where(self._lap_active, self._lap_elapsed + self.step_dt, self._lap_elapsed)
        self._was_in_a = in_a

    def _get_observations(self) -> dict:
        self._update_lap()
        pos = self._robot.data.root_pos_w
        phase_oh = torch.nn.functional.one_hot(self._phase.clamp(0, N_PHASES - 1), N_PHASES).float()
        obs = torch.cat(
            [
                self._pole_a - pos,
                self._pole_b - pos,
                self._robot.data.root_lin_vel_w,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                phase_oh,
                self._lap_elapsed.unsqueeze(-1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _target_xy(self) -> torch.Tensor:
        """Next gate center for shaping."""
        target = self._pole_b.clone()
        # AT_START / IDLE / OUTBOUND → B; TURNED / FINISHED → A
        use_a = (self._phase == PHASE_TURNED) | (self._phase == PHASE_FINISHED)
        target = torch.where(use_a.unsqueeze(-1), self._pole_a, target)
        return target

    def _get_rewards(self) -> torch.Tensor:
        pos = self._robot.data.root_pos_w
        target = self._target_xy()
        dist = torch.linalg.norm(pos[:, :2] - target[:, :2], dim=-1)
        progress = self.cfg.rew_progress * torch.exp(-dist)

        phase_bonus = (
            self.cfg.rew_phase_bonus * self._just_left_a.float()
            + self.cfg.rew_phase_bonus * self._just_hit_b.float()
            + self.cfg.rew_lap_complete * self._just_finished.float()
        )
        time_pen = self.cfg.rew_time_penalty * self._lap_active.float()
        upright = self.cfg.rew_upright * (-self._robot.data.projected_gravity_b[:, 2])
        action_rate = self.cfg.rew_action_rate * torch.sum(
            torch.square(self._actions - self._previous_actions), dim=-1
        )

        height = pos[:, 2] - self.scene.env_origins[:, 2]
        crash = (height < self.cfg.min_height) | (height > self.cfg.max_height)
        crash_pen = self.cfg.rew_crash * crash.float()

        return (progress + phase_bonus + time_pen + upright + action_rate + crash_pen) * self.step_dt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        pos = self._robot.data.root_pos_w
        height = pos[:, 2] - self.scene.env_origins[:, 2]
        mid = 0.5 * (self._pole_a[:, :2] + self._pole_b[:, :2])
        away = torch.linalg.norm(pos[:, :2] - mid, dim=-1) > self.cfg.max_xy_from_course
        crash = (height < self.cfg.min_height) | (height > self.cfg.max_height) | away
        finished = self._phase == PHASE_FINISHED
        terminated = crash | finished
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        state = self._robot.data.default_root_state[env_ids].clone()
        origins = self.scene.env_origins[env_ids]
        state[:, 0] = origins[:, 0] + self.cfg.spawn_offset_xy[0]
        state[:, 1] = origins[:, 1] + self.cfg.spawn_offset_xy[1]
        state[:, 2] = origins[:, 2] + self.cfg.spawn_height
        # Face +X (toward Pole B)
        state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(state[:, 7:], env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._controller.reset(env_ids)
        self._phase[env_ids] = PHASE_IDLE
        self._was_in_a[env_ids] = False
        self._lap_elapsed[env_ids] = 0.0
        self._lap_active[env_ids] = False

    def _debug_vis_callback(self, event):
        self._pole_a_markers.visualize(self._pole_a)
        self._pole_b_markers.visualize(self._pole_b)
