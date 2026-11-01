"""Per-episode domain randomization of the dynamics (SIMULATION_PLAN.md section 11)."""

from __future__ import annotations

import math

import torch

from .config import DynamicsParams, load_dr_ranges

_PLAIN = ("gain_xy", "gain_up", "gain_yaw", "tau_xy", "tau_up", "tau_yaw", "a_max", "deadband",
          "action_delay_s", "rc_loss_prob", "battery_scale")


class DynamicsSampler:
    """sample = nominal + level * (U(lo, hi) - nominal); level 0 is exactly nominal."""

    def __init__(self, nominal: DynamicsParams, ranges: dict | None = None, device="cpu", dtype=torch.float32):
        self.nominal, self.ranges = nominal, ranges or load_dr_ranges()
        self.device, self.dtype = torch.device(device), dtype

    def sample(self, n: int, level: float, gen: torch.Generator | None = None,
               nominal_fraction: float = 0.0) -> dict[str, torch.Tensor]:
        u = lambda *shape: torch.rand(*shape, device=self.device, dtype=self.dtype, generator=gen)
        lvl = torch.full((n,), float(level), device=self.device, dtype=self.dtype)
        if nominal_fraction > 0:
            lvl = torch.where(u(n) < nominal_fraction, torch.zeros_like(lvl), lvl)
        out: dict[str, torch.Tensor] = {}
        for name in _PLAIN:
            nom = getattr(self.nominal, name)
            lo, hi = self.ranges[name]
            if math.isinf(nom):  # disabled limit stays disabled at level 0
                nom = hi
            out[name] = nom + lvl * ((lo + (hi - lo) * u(n)) - nom)
        lo, hi = self.ranges["wind_speed"]
        speed = lvl * (lo + (hi - lo) * u(n))
        ang = 2 * math.pi * u(n)
        out["wind"] = torch.stack([speed * torch.cos(ang), speed * torch.sin(ang)], dim=-1)
        out["dt_jitter_frac"] = lvl * (self.ranges["dt_jitter_frac"][0]
                                       + (self.ranges["dt_jitter_frac"][1] - self.ranges["dt_jitter_frac"][0]) * u(n))
        return out
