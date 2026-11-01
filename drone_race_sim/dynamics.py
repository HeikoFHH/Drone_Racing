"""Kinematic Tello model: rc sticks -> world velocity (SIMULATION_PLAN.md section 6).

Pipeline per control step: packet loss -> action delay -> deadband -> gains -> first-order
lag (world frame, as in policy_integration/stress_sim.py) -> integrate. The Tello closes its
own attitude and velocity loops, so rc is effectively a velocity setpoint.

All state is a tensor with leading dimension E (number of envs). Per-env parameters live in
``self.p`` so domain randomization is just a different value per env.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import torch

from .config import DynamicsParams

RC_BUF = 5  # applied rc history (current + 4 past); max delay = (RC_BUF - 1) * dt


def _full(n, value, device, dtype):
    return torch.full((n,), float(value), device=device, dtype=dtype)


class TelloDynamics:
    def __init__(self, num_envs: int, nominal: DynamicsParams, device="cpu", dtype=torch.float32):
        self.n, self.device, self.dtype = num_envs, torch.device(device), dtype
        self.nominal = nominal
        self.pos = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        self.prev_pos = self.pos.clone()
        self.vel = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)   # world frame
        self.yaw = torch.zeros(num_envs, device=self.device, dtype=dtype)       # CCW from +X
        self.yaw_rate = torch.zeros(num_envs, device=self.device, dtype=dtype)  # CCW positive
        self.rc_buf = torch.zeros(num_envs, RC_BUF, 4, device=self.device, dtype=dtype)
        self.wind = torch.zeros(num_envs, 2, device=self.device, dtype=dtype)   # world velocity disturbance
        self.gust = torch.zeros(num_envs, 2, device=self.device, dtype=dtype)   # OU component
        self.gust_sigma = 0.0
        self.gust_tau = 2.0
        self.p = {k: _full(num_envs, v, self.device, dtype) for k, v in asdict(nominal).items()
                  if k != "lag_mode"}
        self.p["lag_euler"] = _full(num_envs, 1.0 if nominal.lag_mode == "euler" else 0.0, self.device, dtype)

    # -- state management -------------------------------------------------------------------------
    def reset(self, idx: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor) -> None:
        self.pos[idx] = pos
        self.prev_pos[idx] = pos
        self.vel[idx] = 0.0
        self.yaw[idx] = yaw
        self.yaw_rate[idx] = 0.0
        self.rc_buf[idx] = 0.0
        self.gust[idx] = 0.0

    def set_params(self, idx: torch.Tensor, values: dict[str, torch.Tensor | float]) -> None:
        for k, v in values.items():
            if k == "wind":
                self.wind[idx] = torch.as_tensor(v, device=self.device, dtype=self.dtype)
            elif k == "lag_mode":
                self.p["lag_euler"][idx] = 1.0 if v == "euler" else 0.0
            else:
                self.p[k][idx] = torch.as_tensor(v, device=self.device, dtype=self.dtype)

    # -- stepping ---------------------------------------------------------------------------------
    @staticmethod
    def _alpha(dt, tau, euler):
        safe = tau.clamp_min(1e-6)
        a = torch.where(euler > 0.5, dt / safe, 1.0 - torch.exp(-dt / safe))
        return torch.where(tau <= 1e-6, torch.ones_like(a), a).clamp(max=1.0)

    def step(self, rc: torch.Tensor, dt: torch.Tensor, gen: torch.Generator | None = None) -> None:
        """rc: (E, 4) in SDK units (right, forward, up, yaw_clockwise); dt: (E,) seconds."""
        p, dev, dt_ = self.p, self.device, dt
        # 1. rc packet loss: the drone keeps the previously received command
        if (p["rc_loss_prob"] > 0).any():
            lost = torch.rand(self.n, device=dev, generator=gen) < p["rc_loss_prob"]
            rc = torch.where(lost[:, None], self.rc_buf[:, 0], rc)
        self.rc_buf = torch.cat([rc[:, None].to(self.dtype), self.rc_buf[:, :-1]], dim=1)
        # 2. action delay: linear interpolation between past applied commands
        pos_f = (p["action_delay_s"] / dt_).clamp(0.0, RC_BUF - 1.001)
        lo = pos_f.floor()
        frac = (pos_f - lo)[:, None]
        lo_i = lo.long()[:, None, None].expand(-1, 1, 4)
        a = torch.gather(self.rc_buf, 1, lo_i)[:, 0]
        b = torch.gather(self.rc_buf, 1, (lo_i + 1).clamp(max=RC_BUF - 1))[:, 0]
        rc_d = (1 - frac) * a + frac * b
        # 3. deadband
        rc_e = torch.where(rc_d.abs() >= p["deadband"][:, None], rc_d, torch.zeros_like(rc_d))
        # 4. gains
        right = rc_e[:, 0] / 100 * p["gain_xy"] * p["battery_scale"]
        fwd = rc_e[:, 1] / 100 * p["gain_xy"] * p["battery_scale"]
        up = rc_e[:, 2] / 100 * p["gain_up"] * p["battery_scale"]
        cw = rc_e[:, 3] / 100 * p["gain_yaw"] * p["battery_scale"]
        # 5. wind (constant + OU gust) enters the velocity target, like stress_sim's drift
        if self.gust_sigma > 0:
            noise = torch.randn(self.n, 2, device=dev, dtype=self.dtype, generator=gen)
            k = (2 * dt_ / self.gust_tau).sqrt()[:, None]
            self.gust = self.gust - self.gust / self.gust_tau * dt_[:, None] + self.gust_sigma * k * noise
        w = self.wind + self.gust
        co, si = torch.cos(self.yaw), torch.sin(self.yaw)
        target = torch.stack([co * fwd + si * right + w[:, 0], si * fwd - co * right + w[:, 1], up], dim=-1)
        # 6. first-order lag with acceleration clamp
        eul = p["lag_euler"]
        al = torch.stack([self._alpha(dt_, p["tau_xy"], eul)] * 2 + [self._alpha(dt_, p["tau_up"], eul)], dim=-1)
        dv = al * (target - self.vel)
        lim = (p["a_max"] * dt_)[:, None]
        dv = torch.maximum(torch.minimum(dv, lim), -lim)
        self.vel = self.vel + dv
        # 7. integrate (position first with updated velocity, then heading, as in stress_sim)
        self.prev_pos = self.pos
        self.pos = self.pos + self.vel * dt_[:, None]
        al_y = self._alpha(dt_, p["tau_yaw"], eul)
        self.yaw_rate = self.yaw_rate + al_y * (-cw - self.yaw_rate)
        self.yaw = self.yaw + self.yaw_rate * dt_
        # ground: the Tello never goes below the floor
        z = self.pos[:, 2].clamp_min(0.0)
        self.pos = torch.cat([self.pos[:, :2], z[:, None]], dim=-1)

    # -- helpers ----------------------------------------------------------------------------------
    def body_velocity(self) -> torch.Tensor:
        """(forward, left, up) velocity in the body frame."""
        co, si = torch.cos(self.yaw), torch.sin(self.yaw)
        vx, vy, vz = self.vel.unbind(-1)
        return torch.stack([co * vx + si * vy, -si * vx + co * vy, vz], dim=-1)


def segment_point_distance(p0: torch.Tensor, p1: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Distance from point q to segment p0->p1 (all (E, 2))."""
    d = p1 - p0
    denom = (d * d).sum(-1).clamp_min(1e-12)
    u = (((q - p0) * d).sum(-1) / denom).clamp(0.0, 1.0)
    closest = p0 + u[:, None] * d
    return (closest - q).norm(dim=-1)
