"""Policies as plug-ins for the sim, plus the guard that makes Maude's policy the backup for all of them.

Every controller reads only ``env.sensed`` (the estimate in realistic mode, truth in perfect mode),
never the true state, so what is tested here is what could run in flight.

    act(env, obs, now) -> (rc (E, 4) tensor in SDK units, infos: list of dict with "mode" and "land")

Controllers
  PPOController      trained RL policy (batched)
  MaudeController    policy_integration.BaselinePolicy (classic PD route, +/-20 rc, full guard envelope)
  RaceController     race/race_policy.RacePolicy (Tvisha's gate racer, map frame, Positioner-style Obs)
  GuardedController  any primary controller supervised by Maude's guard limits; on a fault it hands over
                     to Maude's PD route (backup) or to hover-and-land
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import _repo  # noqa: F401
from . import frames
from .course import map_poles_for_lap_timer
from .lap import AT_START, FINISHED, IDLE, OUTBOUND, PHASE_NAMES, TURNED
from policy_integration.policy import BaselinePolicy, Observation, PolicyConfig
from race.race_policy import Obs, PolicyCfg, RacePolicy

R_WM = frames.DEFAULT_R_WORLD_MAP
ARM_AFTER_S = 1.0


class Controller:
    name = "controller"

    def reset(self, env, seeds) -> None:
        pass

    def act(self, env, obs, now: float):
        raise NotImplementedError


# ---------------------------------------------------------------------------------------- PPO
class PPOController(Controller):
    name = "ppo"

    def __init__(self, ppo, rc_cap: float):
        self.ppo, self.rc_cap = ppo, rc_cap

    def act(self, env, obs, now):
        a = self.ppo.act_deterministic(obs).to(env.device, env.dtype)
        rc = torch.round(a.clamp(-1, 1) * self.rc_cap)
        return rc, [{"mode": "ppo", "land": False}] * env.num_envs


# ----------------------------------------------------------------------------------- Maude (PD)
def maude_observation(env, i: int, now: float, rng: random.Random | None, pose_noise_m: float = 0.003,
                      bridged: bool = False) -> Observation:
    """Observation per policy_integration/CONTRACT.md from the sensed state of env i.

    Realistic sensing, default (contract v1): the held filtered pose with its ORIGINAL capture time, never
    refreshed between fixes. ``bridged=True`` (plan D4, contract v1.1 proposal): the estimator's dead-reckoned
    position, stamped now, with a std that grows while no tag is seen."""
    s = env.sensed
    if not env.realistic:
        pos = tuple(float(v) + rng.gauss(0, pose_noise_m) for v in s["pos"][i]) if rng else tuple(s["pos"][i].tolist())
        return Observation(now, now, pos, tuple(s["vel_w"][i].tolist()), float(s["yaw"][i]), 0.03, 0.05, 75.0, True, True, True)
    if bridged:
        return Observation(now, now, tuple(s["pos"][i].tolist()), tuple(s["vel_w"][i].tolist()), float(s["yaw"][i]),
                           max(0.03, float(s["pos_std"][i])), 0.1, 75.0, True, True, True)
    return Observation(float(s["fix_time"][i]), now, tuple(s["est_at_fix"][i].tolist()), tuple(s["vel_w"][i].tolist()),
                       float(s["yaw"][i]), max(0.03, float(s["pos_std"][i])), 0.1, 75.0, True, True, True)


class MaudeController(Controller):
    name = "maude_baseline"

    def __init__(self, cfg: PolicyConfig | None = None, name: str | None = None, bridged: bool = False):
        self.cfg, self.bridged = cfg or PolicyConfig(), bridged
        self.name = name or self.name

    def reset(self, env, seeds):
        self.policies = [BaselinePolicy(self.cfg) for _ in range(env.num_envs)]
        self.rngs = [random.Random(s) for s in seeds]

    def act(self, env, obs, now):
        rc = torch.zeros(env.num_envs, 4, dtype=env.dtype, device=env.device)
        infos = []
        for i, pol in enumerate(self.policies):
            d = pol.step(maude_observation(env, i, now, self.rngs[i] if not env.realistic else None, bridged=self.bridged), now)
            if not d.request_land:
                rc[i] = torch.tensor(d.rc, dtype=env.dtype)
            infos.append({"mode": d.mode, "reason": d.reason, "land": d.request_land})
        return rc, infos


# --------------------------------------------------------------------------------- Tvisha racer
def camera_to_map(yaw: float) -> np.ndarray:
    """camera->map rotation for a level drone with course yaw (camera: x right, y down, z forward)."""
    c, s = math.cos(yaw), math.sin(yaw)
    R_cam_course = np.column_stack([-np.array([-s, c, 0.0]), -np.array([0.0, 0.0, 1.0]), np.array([c, s, 0.0])])
    return R_WM.T @ R_cam_course


class RaceController(Controller):
    name = "race_policy"

    def __init__(self, cfg: PolicyCfg | None = None, name: str | None = None):
        self.cfg = cfg or PolicyCfg()
        self.name = name or self.name

    def reset(self, env, seeds):
        poles = map_poles_for_lap_timer(env.geom)
        base = replace(self.cfg, rel_height_m=env.cfg.ref_height)  # sim poles sit at map height 0
        self.policies = [RacePolicy(poles, env.geom.pole_a_tag, env.geom.pole_b_tag, replace(base)) for _ in seeds]
        self.settled = [None] * len(seeds)
        self.rngs = [random.Random(1000 + s) for s in seeds]
        self.prev_t = None

    def act(self, env, obs, now):
        s = env.sensed
        dt = 0.05 if self.prev_t is None else max(now - self.prev_t, 1e-3)
        self.prev_t = now
        lap = env.lap_est if env.realistic else env.lap
        phases = lap.phase.tolist()
        rc = torch.zeros(env.num_envs, 4, dtype=env.dtype, device=env.device)
        infos = []
        for i, pol in enumerate(self.policies):
            if env.realistic:
                lost = bool(s["lost"][i])
                kf_map = frames.course_to_map(s["kf_pos"][i].cpu().numpy())
                pos = None if lost else kf_map.tolist()
                vel = (R_WM.T @ s["vel_w"][i].cpu().numpy()).tolist()
                raw = None
                if bool(s["fix_now"][i]):
                    tags = [1 if bool(s["seen"][i, 0]) else None, 2 if bool(s["seen"][i, 1]) else None]
                    raw = {"xyz": frames.course_to_map(s["fix_pos"][i].cpu().numpy()).tolist(),
                           "R": camera_to_map(float(s["yaw"][i])).tolist(), "tags": [t for t in tags if t]}
                o = Obs(t=now, pos=pos, vel=vel, raw=raw, lost=lost, tags=raw["tags"] if raw else [])
            else:
                p_map = frames.course_to_map(s["pos"][i].cpu().numpy()) + np.array([self.rngs[i].gauss(0, 0.003) for _ in range(3)])
                o = Obs(t=now, pos=p_map.tolist(), vel=(R_WM.T @ s["vel_w"][i].cpu().numpy()).tolist(),
                        raw={"xyz": p_map.tolist(), "R": camera_to_map(float(s["yaw"][i])).tolist(), "tags": []}, lost=False)
            phase = PHASE_NAMES[phases[i]]
            if phase == "AT_START" and not pol.armed:
                self.settled[i] = now if self.settled[i] is None else self.settled[i]
                if now - self.settled[i] >= ARM_AFTER_S:
                    pol.armed = True
            out, info = pol.step(o, phase, dt)
            rc[i] = torch.tensor(out, dtype=env.dtype)
            infos.append({"mode": info.get("mode", "go"), "land": info.get("mode") == "land"})
        return rc, infos


# ------------------------------------------------------------------------------------------- guard
@dataclass
class GuardCfg:
    """Takeover limits. Defaults are Maude's PolicyConfig limits, except the pose-age profile.

    ``profile="classic"`` keeps Maude's pose limits (hold 0.25 s, land 0.75 s); ``"rl"`` relaxes them to
    what a forward-facing camera needs (plan D4: hold 2.5 s, land 4.0 s) while the estimator bridges.
    """

    profile: str = "rl"
    stall_s: float = 20.0               # no lap-phase progress for this long -> backup
    backup_cfg: PolicyConfig | None = None   # default: Maude's PolicyConfig, relaxed per ``profile`` below
    pole_takeover_m: float | None = None   # estimated pole-centre distance; default Maude's pole_clearance_m (0.3)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def backup_policy_cfg(self) -> PolicyConfig:
        if self.backup_cfg is not None:
            return self.backup_cfg
        if self.profile == "classic":
            return self.policy
        return replace(self.policy, max_position_std_m=0.5, max_velocity_std_mps=0.5, max_pose_age_s=self.pose_hold_s,
                       max_stale_hold_s=self.pose_land_s - 0.5, land_after_pose_age_s=self.pose_land_s)

    @property
    def pose_hold_s(self) -> float:
        return self.policy.max_pose_age_s if self.profile == "classic" else 2.5

    @property
    def pose_land_s(self) -> float:
        return self.policy.land_after_pose_age_s if self.profile == "classic" else 4.0


PRIMARY, BACKUP, LANDING = 0, 1, 2
_WAYPOINT_FOR_PHASE = {IDLE: 1, AT_START: 1, OUTBOUND: 1, TURNED: 2, FINISHED: 2}  # Maude's route: A, B, A


class GuardedController(Controller):
    """Maude's policy as the safety net for any primary controller (plan D3).

    Each tick, per env, in this order:
      1. LANDING is latched (Maude: an abort stays latched): zero sticks, land requested.
      2. Guard limits taken from ``PolicyConfig``: estimated position outside the geofence, measured
         speed over the limit, or no fix for ``pose_land_s``  ->  LANDING. No fix for ``pose_hold_s``  ->  hover.
      3. Primary output not finite / out of range, stalled lap progress, or the estimate inside a pole's
         takeover distance  ->  BACKUP: a fresh ``BaselinePolicy`` flies Maude's PD route from the waypoint
         that matches the lap phase, with all of its own guards, and lands when it aborts or completes.
    """

    def __init__(self, primary: Controller, cfg: GuardCfg | None = None):
        self.primary, self.cfg = primary, cfg or GuardCfg()
        self.name = f"guarded[{primary.name}]"

    def reset(self, env, seeds):
        E = env.num_envs
        self.primary.reset(env, seeds)
        self.state = [PRIMARY] * E
        self.backup = [None] * E
        self.reason = [None] * E
        self.takeover_t = [None] * E
        self.progress_t = [0.0] * E
        self.last_phase = [None] * E
        self.rngs = [random.Random(s) for s in seeds]

    def _guard_fault(self, env, i, s) -> str | None:
        c = self.cfg.policy
        if not c.inside(tuple(s["pos"][i].tolist())):
            return "geofence"
        if float(s["vel_w"][i].norm()) > c.max_measured_speed_mps:
            return "excess_speed"
        return None

    def act(self, env, obs, now):
        rc_p, infos_p = self.primary.act(env, obs, now)
        s = env.sensed
        lap = env.lap_est if env.realistic else env.lap
        phases = lap.phase.tolist()
        rc = torch.zeros_like(rc_p)
        infos = []
        pa, pb = env.lap.pole_a, env.lap.pole_b
        pole_m = self.cfg.pole_takeover_m or self.cfg.policy.pole_clearance_m
        for i in range(env.num_envs):
            phase = phases[i]
            if self.last_phase[i] != phase:
                self.last_phase[i], self.progress_t[i] = phase, now
            st = self.state[i]
            if st == LANDING:
                infos.append({"mode": "landing", "reason": self.reason[i], "land": True})
                continue
            fault = self._guard_fault(env, i, s)
            age = float(s["fix_age"][i]) if env.realistic else 0.0
            if fault is None and age > self.cfg.pose_land_s:
                fault = "sensors_timed_out"
            if fault:
                self.state[i], self.reason[i] = LANDING, fault
                infos.append({"mode": "landing", "reason": fault, "land": True})
                continue
            if st == PRIMARY:
                r = rc_p[i]
                bad = (not bool(torch.isfinite(r).all())) or bool((r.abs() > 100).any())
                trigger = "invalid_primary" if bad else None
                if trigger is None and now - self.progress_t[i] > self.cfg.stall_s and phase != FINISHED:
                    trigger = "stalled"
                pos_xy = s["pos"][i, :2]
                if trigger is None and min(float((pos_xy - pa[i]).norm()), float((pos_xy - pb[i]).norm())) < pole_m:
                    trigger = "pole_proximity"
                if trigger is not None:
                    self.state[i], self.reason[i], self.takeover_t[i] = BACKUP, trigger, now
                    pol = BaselinePolicy(self.cfg.backup_policy_cfg())
                    pol.waypoint_index = _WAYPOINT_FOR_PHASE[phase]
                    self.backup[i] = pol
                    st = BACKUP
                else:
                    if age > self.cfg.pose_hold_s:
                        infos.append({"mode": "hold", "reason": "pose_stale", "land": False})
                    else:
                        rc[i] = r
                        infos.append(dict(infos_p[i], reason=None))
                    continue
            # BACKUP: Maude's PD route
            d = self.backup[i].step(maude_observation(env, i, now, self.rngs[i] if not env.realistic else None,
                                                      bridged=self.cfg.profile != "classic"), now)
            if d.request_land or d.mode in ("abort", "complete"):
                self.state[i], self.reason[i] = LANDING, f"backup_{d.reason}"
                infos.append({"mode": "landing", "reason": self.reason[i], "land": True})
                continue
            rc[i] = torch.tensor(d.rc, dtype=env.dtype)
            infos.append({"mode": "backup", "reason": self.reason[i], "land": False})
        return rc, infos


# ------------------------------------------------------------------------- fault injection (tests)
class FaultyController(Controller):
    """Wrap a controller and corrupt its output from ``t_fault`` on, to exercise the guard."""

    def __init__(self, inner: Controller, fault: str, t_fault: float = 3.0, seed: int = 0):
        self.inner, self.fault, self.t_fault = inner, fault, t_fault
        self.name = f"{inner.name}+{fault}"
        self.rng = torch.Generator().manual_seed(seed)

    def reset(self, env, seeds):
        self.inner.reset(env, seeds)

    def act(self, env, obs, now):
        rc, infos = self.inner.act(env, obs, now)
        if now < self.t_fault:
            return rc, infos
        if self.fault == "nan":
            rc = torch.full_like(rc, float("nan"))
        elif self.fault == "stall":
            rc = torch.zeros_like(rc)
        elif self.fault == "runaway":
            rc = torch.tensor([60.0, 60.0, 0.0, 0.0], dtype=rc.dtype, device=rc.device).expand_as(rc).clone()
        elif self.fault == "noise":
            rc = (torch.rand(rc.shape, generator=self.rng, dtype=rc.dtype) * 200 - 100).to(rc.device)
        elif self.fault == "dive":
            rc = torch.tensor([0.0, 0.0, -80.0, 0.0], dtype=rc.dtype, device=rc.device).expand_as(rc).clone()
        else:
            raise ValueError(self.fault)
        return rc, [dict(i, mode=f"fault:{self.fault}") for i in infos]


class SensorFaultController(Controller):
    """Wrap a controller and inject a sensor fault on the env at ``t_start`` (tag blackout for ``duration_s``)."""

    def __init__(self, inner: Controller, duration_s: float, t_start: float = 6.0):
        self.inner, self.duration_s, self.t_start = inner, duration_s, t_start
        self.name = inner.name
        self._done = False

    def reset(self, env, seeds):
        self.inner.reset(env, seeds)
        self._done = False

    def act(self, env, obs, now):
        if not self._done and now >= self.t_start:
            env.sensors.inject_blackout(self.duration_s)
            self._done = True
        return self.inner.act(env, obs, now)
