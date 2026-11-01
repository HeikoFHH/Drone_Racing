"""Vectorized two-pole Tello race environment in course_flu_v1 (SIMULATION_PLAN.md sections 5-8).

MVP (M1/M2): perfect-state observations laid out like the final actor frame (tag/estimator
fields reserved, set to zero until M3), dynamics with lag/delay/deadband/wind, repo lap rules,
same-step autoreset with ``info["final_obs"]``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import gymnasium as gym
import numpy as np
import torch

from . import _repo  # noqa: F401
from .config import (DT_NOMINAL, DynamicsParams, EnvCfg, load_dynamics, load_reward_cfg, params_hash)
from .course import CourseGeometry, load_course_geometry
from .dynamics import TelloDynamics, segment_point_distance
from .lap import AT_START, FINISHED, IDLE, OUTBOUND, TURNED, VecLapTimer
from .randomization import DynamicsSampler
from .rewards import compute_rewards

FRAME_DIM = 25
PRIV_DIM = 16
CAUSE_NONE, CAUSE_SUCCESS, CAUSE_POLE, CAUSE_GROUND, CAUSE_CEILING, CAUSE_OOB, CAUSE_TIMEOUT = range(7)
CAUSE_NAMES = ("none", "success", "pole", "ground", "ceiling", "out_of_bounds", "timeout")


class TelloRaceVecEnv(gym.vector.VectorEnv):
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(self, cfg: EnvCfg | None = None, dynamics: DynamicsParams | None = None,
                 geometry: CourseGeometry | None = None, params_source: str | None = None):
        self.cfg = cfg or EnvCfg()
        c = self.cfg
        self.dtype = torch.float64 if c.dtype == "float64" else torch.float32
        self.device = torch.device(c.device)
        if dynamics is None:
            dynamics, loaded_source = load_dynamics()
            params_source = params_source or loaded_source
        self.nominal, self.params_source = dynamics, params_source or "custom"
        self.geom = geometry or load_course_geometry()
        self.rw = load_reward_cfg()
        self.num_envs = E = c.num_envs
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(c.seed)
        self.dyn = TelloDynamics(E, dynamics, self.device, self.dtype)
        self.lap = VecLapTimer(E, self.geom.pole_a_xy, self.geom.pole_b_xy, self.geom.gate_radius_m,
                               self.geom.require_turn, self.device, self.dtype)
        self.sampler = DynamicsSampler(dynamics, device=self.device, dtype=self.dtype)
        self.dr_level = float(c.dr_level)
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=self.dtype)
        self.t, self.dt_last = z(E), torch.full((E,), c.dt, device=self.device, dtype=self.dtype)
        self.jitter = z(E)
        self.steps = torch.zeros(E, dtype=torch.long, device=self.device)
        self.hist = z(E, c.frame_stack, FRAME_DIM)
        self.act_hist = z(E, c.action_history, 4)
        self.ep_return, self.ep_len = z(E), torch.zeros(E, dtype=torch.long, device=self.device)
        self._override: dict = {}
        self.obs_dim = FRAME_DIM * c.frame_stack + 4 * c.action_history
        self.priv_dim = PRIV_DIM
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, E)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, E)
        self.spec = None
        self.render_mode = None
        self.closed = False
        self.params_hash = params_hash(c, dynamics)
        self.reset()

    # -- public API -------------------------------------------------------------------------------
    def set_dr_level(self, level: float) -> None:
        self.dr_level = float(level)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """options (persist until cleared with options={}): preset="stress_sim", wind_xy, deadband,
        pos (E,3), yaw (E,), params (dict of per-env dynamics overrides)."""
        if seed is not None:
            self.gen.manual_seed(seed)
        if options is not None:
            self._override = dict(options)
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        return self._stacked(), {"priv": self._priv()}

    def step(self, actions):
        a = torch.as_tensor(actions, device=self.device, dtype=self.dtype).clamp(-1.0, 1.0)
        rc = torch.round(a * self.cfg.rc_cap).clamp(-100.0, 100.0)
        return self._step(rc, a)

    def step_rc(self, rc):
        """Step with raw SDK rc integers (right, forward, up, yaw_cw), bypassing the action cap."""
        rc = torch.as_tensor(rc, device=self.device, dtype=self.dtype)
        return self._step(rc, rc / max(self.cfg.rc_cap, 1.0))

    # -- core -------------------------------------------------------------------------------------
    def _step(self, rc, a):
        c, rw = self.cfg, self.rw
        E = self.num_envs
        dt = c.dt * (1.0 + self.jitter * torch.rand(E, device=self.device, dtype=self.dtype, generator=self.gen))
        phase_prev = self.lap.phase.clone()
        target_prev = torch.where((phase_prev == TURNED)[:, None], self.lap.pole_a, self.lap.pole_b)
        d_prev = (self.dyn.pos[:, :2] - target_prev).norm(dim=-1)
        z_prev = self.dyn.pos[:, 2].clone()
        self.dyn.step(rc, dt, self.gen)
        self.t = self.t + dt
        self.dt_last = dt
        self.steps += 1
        pos = self.dyn.pos
        events = self.lap.update(self.t, pos[:, :2])
        d_curr = (pos[:, :2] - target_prev).norm(dim=-1)

        da = segment_point_distance(self.dyn.prev_pos[:, :2], pos[:, :2], self.lap.pole_a)
        db = segment_point_distance(self.dyn.prev_pos[:, :2], pos[:, :2], self.lap.pole_b)
        pole_dist = torch.minimum(da, db)
        low = torch.minimum(z_prev, pos[:, 2]) < c.pole_height + 0.1
        pole_hit = (pole_dist < c.pole_radius) & low
        ground = pos[:, 2] < c.ground_z
        ceiling = pos[:, 2] > c.ceiling_z
        mid = (self.lap.pole_a + self.lap.pole_b) / 2
        oob = (pos[:, :2] - mid).norm(dim=-1) > c.max_xy_from_course
        crash = (pole_hit | ground | ceiling | oob) & c.enable_terminations
        success = events["lap_finish"] & c.enable_terminations & c.terminate_on_finish
        terminated = crash | success
        truncated = (self.t >= c.episode_s) & ~terminated & c.enable_terminations

        comps = compute_rewards(rw, dt=dt, d_prev=d_prev, d_curr=d_curr, events=events, phase_new=self.lap.phase,
                                lap_time=self.lap.lap_time, act=a, act_prev=self.act_hist[:, 0],
                                act_prev2=self.act_hist[:, 1], z=pos[:, 2], pole_dist_min=pole_dist, crash=crash)
        reward = sum(comps.values())
        self.act_hist = torch.cat([a[:, None], self.act_hist[:, :-1]], dim=1)
        self.ep_return = self.ep_return + reward
        self.ep_len += 1
        self.hist = torch.cat([self.hist[:, 1:], self._frame()[:, None]], dim=1)

        cause = torch.zeros(E, dtype=torch.long, device=self.device)
        for code, mask in ((CAUSE_TIMEOUT, truncated), (CAUSE_OOB, crash & oob), (CAUSE_CEILING, crash & ceiling),
                           (CAUSE_GROUND, crash & ground), (CAUSE_POLE, crash & pole_hit), (CAUSE_SUCCESS, success)):
            cause[mask] = code
        done = terminated | truncated
        obs, priv = self._stacked(), self._priv()
        info = {"done": done, "cause": cause, "ep_return": self.ep_return.clone(), "ep_len": self.ep_len.clone(),
                "lap_finished": self.lap.phase == FINISHED, "lap_time": self.lap.lap_time.clone(),
                "phase": self.lap.phase.clone(), "final_obs": obs.clone(), "final_priv": priv.clone(),
                "reward_components": comps, "pole_dist": pole_dist}
        if c.autoreset and done.any():
            self._reset_idx(done.nonzero(as_tuple=True)[0])
            obs, priv = self._stacked(), self._priv()
        info["priv"] = priv
        return obs, reward, terminated, truncated, info

    # -- reset ------------------------------------------------------------------------------------
    def _reset_idx(self, idx: torch.Tensor) -> None:
        c, n = self.cfg, len(idx)
        u = lambda *s: torch.rand(*s, device=self.device, dtype=self.dtype, generator=self.gen)
        ov = self._override
        # dynamics parameters
        if ov.get("preset") == "stress_sim":
            from dataclasses import asdict
            values = {k: v for k, v in asdict(DynamicsParams.stress_sim_preset()).items()}
            self.dyn.set_params(idx, values)
            self.dyn.wind[idx] = 0.0
            self.jitter[idx] = 0.0
        else:
            s = self.sampler.sample(n, self.dr_level, self.gen, c.nominal_fraction if self.dr_level > 0 else 0.0)
            self.jitter[idx] = s.pop("dt_jitter_frac")
            s["lag_mode"] = self.nominal.lag_mode
            self.dyn.set_params(idx, s)
        if "params" in ov:
            self.dyn.set_params(idx, ov["params"])
        if "wind_xy" in ov:
            self.dyn.wind[idx] = torch.as_tensor(ov["wind_xy"], device=self.device, dtype=self.dtype)
        if "deadband" in ov:
            self.dyn.p["deadband"][idx] = float(ov["deadband"])
        # spawn inside the A gate, facing B
        ang, rad = 2 * math.pi * u(n), c.spawn_radius * u(n).sqrt()
        a_xy = self.lap.pole_a[idx]
        pos = torch.stack([a_xy[:, 0] + rad * torch.cos(ang), a_xy[:, 1] + rad * torch.sin(ang),
                           c.spawn_height[0] + (c.spawn_height[1] - c.spawn_height[0]) * u(n)], dim=-1)
        yaw = (2 * u(n) - 1) * c.spawn_yaw_range
        if "pos" in ov:
            pos = torch.as_tensor(ov["pos"], device=self.device, dtype=self.dtype)[idx]
        if "yaw" in ov:
            yaw = torch.as_tensor(ov["yaw"], device=self.device, dtype=self.dtype)[idx]
        self.dyn.reset(idx, pos, yaw)
        self.t[idx] = 0.0
        self.dt_last[idx] = c.dt
        self.steps[idx] = 0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        self.lap.reset(idx)
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[idx] = True
        self.lap.update(self.t, self.dyn.pos[:, :2], mask)
        self.act_hist[idx] = 0.0
        self.hist[idx] = self._frame()[idx][:, None].expand(-1, c.frame_stack, -1)

    # -- observations -----------------------------------------------------------------------------
    def _rel_body(self, target_xy):
        d = target_xy - self.dyn.pos[:, :2]
        co, si = torch.cos(self.dyn.yaw), torch.sin(self.dyn.yaw)
        return torch.stack([co * d[:, 0] + si * d[:, 1], -si * d[:, 0] + co * d[:, 1]], dim=-1)

    def _frame(self) -> torch.Tensor:
        """Current actor frame (25 dims). Tag/estimator slots are reserved for M3 and stay zero."""
        ph = self.lap.phase
        onehot = torch.stack([(ph == IDLE), (ph == AT_START), (ph == OUTBOUND), (ph == TURNED)], dim=-1).to(self.dtype)
        zeros = lambda k: torch.zeros(self.num_envs, k, device=self.device, dtype=self.dtype)
        return torch.cat([
            self._rel_body(self.lap.pole_a) / 4.0,                      # 2
            self._rel_body(self.lap.pole_b) / 4.0,                      # 2
            (self.dyn.pos[:, 2:3] - self.cfg.ref_height),               # 1
            self.dyn.body_velocity(),                                   # 3
            self.dyn.yaw_rate[:, None],                                 # 1
            zeros(4), zeros(4),                                         # tag A / tag B (M3)
            zeros(1), zeros(1), zeros(1),                               # fix_age, lost, std (M3)
            onehot,                                                     # 4
            (self.dt_last / DT_NOMINAL)[:, None],                       # 1
        ], dim=-1)

    def _stacked(self) -> torch.Tensor:
        return torch.cat([self.hist.flatten(1), self.act_hist.flatten(1)], dim=-1)

    def _priv(self) -> torch.Tensor:
        p, d = self.dyn.p, self.dyn
        return torch.cat([
            torch.stack([d.pos[:, 0] / 4.0, d.pos[:, 1] / 4.0, d.pos[:, 2]], dim=-1),
            torch.stack([torch.sin(d.yaw), torch.cos(d.yaw)], dim=-1),
            d.vel, d.wind,
            (self.lap.elapsed(self.t) / 30.0)[:, None],
            torch.stack([p["tau_xy"] / 0.2, p["deadband"] / 12.0, p["action_delay_s"] / 0.15,
                         p["gain_xy"] / 0.8, p["battery_scale"]], dim=-1),
        ], dim=-1)

    def close_extras(self, **kwargs):
        pass
