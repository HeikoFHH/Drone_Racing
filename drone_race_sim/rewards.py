"""Reward components (SIMULATION_PLAN.md section 8). Returns a dict so each term can be logged."""

from __future__ import annotations

import torch

from .lap import OUTBOUND, TURNED


def compute_rewards(w: dict, *, dt, d_prev, d_curr, events, phase_new, lap_time, act, act_prev, act_prev2,
                    z, pole_dist_min, crash) -> dict[str, torch.Tensor]:
    zero = torch.zeros_like(dt)
    r: dict[str, torch.Tensor] = {}
    r["progress"] = w["progress"] * (d_prev - d_curr) / (w["v_ref"] * dt)
    r["leave_a"] = w["leave_a"] * events["lap_start"].to(dt.dtype)
    r["hit_b"] = w["hit_b"] * events["turn"].to(dt.dtype)
    bonus = w["finish_time_k"] * (w["finish_t_ref"] - torch.nan_to_num(lap_time, nan=w["finish_t_ref"])).clamp_min(0.0)
    r["finish"] = events["lap_finish"].to(dt.dtype) * (w["finish"] + bonus)
    racing = (phase_new == OUTBOUND) | (phase_new == TURNED)
    r["time"] = w["time_penalty"] * racing.to(dt.dtype)
    r["false_start"] = w["false_start"] * events["false_start"].to(dt.dtype)
    r["action_rate"] = w["action_rate"] * ((act - act_prev) ** 2).sum(-1)
    jerk = act - 2 * act_prev + act_prev2
    r["action_jerk"] = w["action_jerk"] * (jerk ** 2).sum(-1)
    r["height"] = w["height"] * ((w["height_lo"] - z).clamp_min(0.0) + (z - w["height_hi"]).clamp_min(0.0))
    near = (w["pole_proximity_dist"] - pole_dist_min).clamp_min(0.0) / w["pole_proximity_dist"]
    r["pole_proximity"] = w["pole_proximity"] * near
    r["crash"] = w["crash"] * crash.to(dt.dtype)
    return {k: torch.where(torch.isfinite(v), v, zero) for k, v in r.items()}
