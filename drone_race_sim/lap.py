"""Vectorized port of race/lap_timer.py (same rules, same transition order)."""

from __future__ import annotations

import torch

IDLE, AT_START, OUTBOUND, TURNED, FINISHED = range(5)
PHASE_NAMES = ("IDLE", "AT_START", "OUTBOUND", "TURNED", "FINISHED")


class VecLapTimer:
    """One timer per env. Gate distance is horizontal (course X/Y), like the repo timer's map X/Z."""

    def __init__(self, num_envs: int, pole_a, pole_b, gate_radius: float, require_turn: bool = True,
                 device="cpu", dtype=torch.float32):
        self.n, self.device, self.dtype = num_envs, torch.device(device), dtype
        self.gate_radius, self.require_turn = float(gate_radius), bool(require_turn)
        self.pole_a = torch.as_tensor(pole_a, device=self.device, dtype=dtype).expand(num_envs, 2).clone()
        self.pole_b = torch.as_tensor(pole_b, device=self.device, dtype=dtype).expand(num_envs, 2).clone()
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        nan = torch.full((num_envs,), float("nan"), device=self.device, dtype=dtype)
        self.t_start, self.t_finish = nan.clone(), nan.clone()
        self.was_in_a = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, idx: torch.Tensor | None = None, pole_b=None) -> None:
        idx = torch.arange(self.n, device=self.device) if idx is None else idx
        self.phase[idx] = IDLE
        self.t_start[idx] = float("nan")
        self.t_finish[idx] = float("nan")
        self.was_in_a[idx] = False
        if pole_b is not None:
            self.pole_b[idx] = pole_b

    @property
    def lap_time(self) -> torch.Tensor:
        return self.t_finish - self.t_start  # nan until finished

    def elapsed(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(self.t_start), torch.zeros_like(t), t - self.t_start)

    def in_gates(self, xy: torch.Tensor):
        r = self.gate_radius
        return ((xy - self.pole_a).norm(dim=-1) <= r), ((xy - self.pole_b).norm(dim=-1) <= r)

    def update(self, t: torch.Tensor, xy: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Advance the state machine for envs in ``mask`` (default all). Returns event masks."""
        in_a, in_b = self.in_gates(xy)
        m = torch.ones_like(in_a) if mask is None else mask
        ph, was = self.phase, self.was_in_a
        at_start = m & (ph == IDLE) & in_a
        lap_start = m & (ph == AT_START) & was & ~in_a
        turn = m & (ph == OUTBOUND) & in_b
        false_start = m & (ph == OUTBOUND) & ~in_b & in_a & self.require_turn
        finish = m & (ph == TURNED) & in_a
        new = ph.clone()
        new[at_start] = AT_START
        new[lap_start] = OUTBOUND
        new[turn] = TURNED
        new[false_start] = AT_START
        new[finish] = FINISHED
        self.t_start = torch.where(lap_start, t, torch.where(false_start, torch.full_like(t, float("nan")), self.t_start))
        self.t_finish = torch.where(lap_start, torch.full_like(t, float("nan")), torch.where(finish, t, self.t_finish))
        self.phase = new
        self.was_in_a = torch.where(m, in_a, was)
        return {"at_start": at_start, "lap_start": lap_start, "turn": turn, "false_start": false_start,
                "lap_finish": finish, "in_a": in_a, "in_b": in_b}
