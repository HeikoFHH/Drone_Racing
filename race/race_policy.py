"""Simple gate policy: find a tag, then fly through the gates A -> B -> A.

Built on what the repo already has -- nothing here detects tags or talks to the drone:
  * position + camera rotation come from core/positional.py (FilteredPositioner/Positioner)
  * "which way is that point from the camera" is Positioner.direction_to (same as
    tools/tello_to_origin.py, which steers on it)
  * yaw / height loops use the Pid class + yaw gains from tools/tello_policy.py
  * lap phases come from race/lap_timer.py

Map frame (config/map.yaml): origin = tag 1 on Pole A; x right, y UP, z out of the
tag face; metres. Ground plane = x-z. Output sticks: (right, forward, up, yaw), -100..100.

Default strategy ``look_at="B"``: the nose stays on Pole B's tag all lap -- forward to
B, then straight BACK to A -- so the drone never turns away from its position tag.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "core", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from positional import Positioner                       # noqa: E402  core/positional.py
from tello_policy import MAX_YAW, YAW_PID, Pid, _clamp  # noqa: E402  tools/tello_policy.py



def direction_in_camera(loc: dict, target) -> np.ndarray:
    """Unit vector from the drone to ``target`` in the CAMERA frame (x right, y down,
    z forward): Positioner.direction_to, exactly as tools/tello_to_origin.py steers.

    Pass the RAW Positioner result: its R and xyz come from the same tag sighting, so
    the direction stays right even when a far, small tag's pose flips between its two
    solutions (which moves both xyz and R, consistently)."""
    return np.asarray(Positioner.direction_to(None, loc, target), float)


@dataclass
class PolicyCfg:
    cruise_mps: float = 0.8        # top ground speed: THE speed knob
    accel_mps2: float = 1.0        # brake so that speed <= sqrt(2 * accel * distance_left)
    mps_per_rc: float = 0.008      # rc 100 ~ 0.8 m/s (tools/tello_policy.py estimate). MEASURE.
    k_vel: float = 25.0            # rc per m/s of speed error (feedback on top of feedforward)
    k_up: float = 60.0             # rc per metre of height error
    max_rc: int = 100              # stick cap; cruise_mps is the real speed limit
    max_up_rc: int = 30
    min_rc: int = 8                # the Tello ignores tiny sticks: push past that while far
    arrive_m: float = 0.15
    hold_R_s: float = 0.5          # reuse the last raw fix's rotation this long between fixes
    yaw_deadband_deg: float = 2.0  # ignore bearing jitter smaller than this
    search_yaw_rc: int = 40        # no tag: turn on the spot at this yaw stick until one shows up
    land_after_lost_s: float = 30.0 # still nothing after this long (> one full turn): land
    gate_inset_m: float = 0.6      # aim this far from a pole along A-B (gate radius 0.8)
    park_m: float = 1.2            # after the finish, stop this far from A toward B
    rel_height_m: float = 0.0      # fly this high relative to the tag centres (map y)
    look_at: str = "B"             # "B" (nose on B all lap) | "target" (face where you go)
    lookahead_m: float = 0.8       # follow the A-B line: aim this far ahead on it (pure pursuit)
    # --- wind ------------------------------------------------------------------------
    k_wind_i: float = 15.0         # integral on speed error: leans into a STEADY wind (Pid ki)
    adapt_to_wind: bool = False    # slow down in gusts. OFF: in sim it cost time AND drift (more exposure)
    response_tau_s: float = 0.35   # how long the drone takes to reach a commanded speed. MEASURE.
    turb_threshold_mps: float = 0.2  # gustiness below this is ignored
    turb_slowdown: float = 1.5     # lose this fraction of speed per m/s of gustiness above it
    min_speed_frac: float = 0.4    # never slow below this fraction of cruise


@dataclass
class Obs:
    t: float
    pos: list | None               # FilteredPositioner "xyz" (None while lost)
    vel: list | None               # FilteredPositioner "vel"
    raw: dict | None               # Positioner.locate() result for THIS frame (accepted), else None
    lost: bool
    tags: list = field(default_factory=list)


class GoTo:
    """Fly to a 3-D map point, nose on ``look_at``. Search (sweep) while lost."""

    def __init__(self, cfg: PolicyCfg | None = None) -> None:
        self.cfg = cfg or PolicyCfg()
        self.yaw_pid = Pid(*YAW_PID, out_abs=MAX_YAW)
        self.up_pid = Pid(self.cfg.k_up, 0.0, 10.0, out_abs=self.cfg.max_up_rc)
        # speed-feedback loops (existing Pid): P = k_vel, I = k_wind_i soaks up steady wind
        self.right_pid = Pid(self.cfg.k_vel, self.cfg.k_wind_i, 0.0, out_abs=100, i_abs=4.0)
        self.fwd_pid = Pid(self.cfg.k_vel, self.cfg.k_wind_i, 0.0, out_abs=100, i_abs=4.0)
        self._prev_want = None                  # body velocity we asked for last tick
        self._expect = np.zeros(2)              # where the drone's speed SHOULD be, given its lag
        self.wind_est = np.zeros(2)             # slow average of (measured - asked): steady push
        self.turb = 0.0                         # fast average of the rest: gustiness, m/s
        self._raw = None
        self._raw_t = -math.inf
        self._lost_since = None
        self._last_bearing = -1.0          # never seen a tag: search to the left first

    def _search(self, t: float):
        """Hold position and turn on the spot toward where the tag was last seen, until a
        mapped tag shows up. (core/tello_dual_video.py sweeps +-30 deg left/right instead:
        fine for a tag lost a moment ago, but it never finds one that starts behind you.)"""
        if self._lost_since is None:
            self._lost_since = t
            for pid in (self.yaw_pid, self.up_pid, self.right_pid, self.fwd_pid):
                pid.reset()
            self._prev_want = None
            self._expect = np.zeros(2)
        since = t - self._lost_since
        if since > self.cfg.land_after_lost_s:
            return (0, 0, 0, 0), {"mode": "land"}
        yaw = self.cfg.search_yaw_rc if self._last_bearing > 0 else -self.cfg.search_yaw_rc
        return (0, 0, 0, yaw), {"mode": "search"}

    def step(self, obs: Obs, target, look_at=None, speed_cap: float | None = None, dt: float = 0.05,
             line_from=None):
        """``line_from``: if given, stay on the straight line line_from -> target (aim at a
        point ``lookahead_m`` ahead on it) instead of heading straight for the target from
        wherever a gust left you."""
        c = self.cfg
        if obs.raw is not None:
            self._raw, self._raw_t = obs.raw, obs.t
        fresh = obs.raw is not None
        if obs.lost or obs.pos is None or obs.t - self._raw_t > c.hold_R_s:
            return self._search(obs.t)
        self._lost_since = None

        p = np.asarray(obs.pos, float)
        v = np.asarray(obs.vel if obs.vel is not None else (0, 0, 0), float)
        target = np.asarray(target, float)
        R = np.asarray(self._raw["R"], float)
        # directions from this frame's raw fix; between fixes, last rotation + filtered position
        loc = self._raw if fresh else {"R": self._raw["R"], "xyz": list(p)}

        # speed along the ground: cruise, braking near the target
        dist = math.hypot(target[0] - p[0], target[2] - p[2])
        speed = min(c.cruise_mps, math.sqrt(2.0 * c.accel_mps2 * dist))
        if speed_cap is not None:
            speed = min(speed, speed_cap)

        # wind: compare how the drone actually moved with what we asked last tick
        v_cam = R.T @ v                               # current velocity in body axes
        if self._prev_want is not None:
            # expected speed = what we asked, delayed by the drone's response time; anything
            # beyond that is the air pushing it
            self._expect += (self._prev_want - self._expect) * min(1.0, dt / c.response_tau_s)
            resid = np.array([v_cam[0], v_cam[2]]) - self._expect
            self.wind_est += (resid - self.wind_est) * min(1.0, dt / 3.0)
            self.turb += (float(np.linalg.norm(resid - self.wind_est)) - self.turb) * min(1.0, dt / 1.0)
        scale = 1.0
        if c.adapt_to_wind:
            scale = min(1.0, max(c.min_speed_frac,
                                 1.0 - c.turb_slowdown * max(0.0, self.turb - c.turb_threshold_mps)))
        speed *= scale

        # where to aim: the target, or a carrot on the line toward it
        aim = target
        if line_from is not None:
            a0 = np.asarray(line_from, float)
            seg = np.array([target[0] - a0[0], 0.0, target[2] - a0[2]])
            seg_len = float(np.linalg.norm(seg))
            if seg_len > 1e-6:
                u = seg / seg_len
                along = float(np.clip((p - a0) @ u, 0.0, seg_len))
                aim = a0 + u * min(seg_len, along + c.lookahead_m)
                if seg_len - along <= c.lookahead_m:
                    aim = target
        # which way is the aim point, in body axes (existing Positioner.direction_to)
        d = direction_in_camera(loc, [aim[0], loc["xyz"][1], aim[2]])
        flat = math.hypot(d[0], d[2]) or 1.0
        want_r, want_f = speed * d[0] / flat, speed * d[2] / flat
        self._prev_want = np.array([want_r, want_f])
        right = want_r / c.mps_per_rc + self.right_pid.step(want_r - v_cam[0], dt)
        fwd = want_f / c.mps_per_rc + self.fwd_pid.step(want_f - v_cam[2], dt)
        mag = math.hypot(right, fwd)
        if dist > c.arrive_m and 0 < mag < c.min_rc:
            right, fwd = right * c.min_rc / mag, fwd * c.min_rc / mag
        elif mag > c.max_rc:
            right, fwd = right * c.max_rc / mag, fwd * c.max_rc / mag

        # height (map y is up), with the existing Pid
        dy = target[1] - p[1]
        up = self.up_pid.step(dy, dt)

        # yaw: keep look_at centred -- bearing exactly as tools/tello_to_origin.py
        bearing = 0.0
        if look_at is not None and math.hypot(look_at[0] - p[0], look_at[2] - p[2]) > 0.3:
            b = direction_in_camera(loc, look_at)
            bearing = math.degrees(math.atan2(b[0], b[2]))
        if bearing:
            self._last_bearing = bearing
        if abs(bearing) < c.yaw_deadband_deg:
            bearing = 0.0
        yaw = self.yaw_pid.step(bearing, dt)

        rc = tuple(int(round(_clamp(x, -100, 100))) for x in (right, fwd, up, yaw))
        arrived = dist <= c.arrive_m and abs(dy) <= 0.15
        return rc, {"mode": "arrived" if arrived else "go", "dist_m": round(dist, 2),
                    "bearing_deg": round(bearing, 1), "speed_cmd": round(speed, 2),
                    "turb": round(self.turb, 2), "speed_scale": round(scale, 2)}


class TagApproach:
    """Get to one tag: stop ``standoff_m`` in front of it (on the side you started), nose on it."""

    def __init__(self, tag_xyz, standoff_m: float = 1.0, cfg: PolicyCfg | None = None) -> None:
        self.tag = np.asarray(tag_xyz, float)
        self.standoff_m = standoff_m
        self.goto = GoTo(cfg)
        self.goal = None

    def step(self, obs: Obs, dt: float = 0.05):
        if self.goal is None and obs.pos is not None and not obs.lost:
            p = np.asarray(obs.pos, float)
            dx, dz = p[0] - self.tag[0], p[2] - self.tag[2]
            n = math.hypot(dx, dz) or 1.0
            self.goal = self.tag + np.array([dx / n, 0, dz / n]) * self.standoff_m
            self.goal[1] = self.tag[1] + self.goto.cfg.rel_height_m
        return self.goto.step(obs, self.goal if self.goal is not None else self.tag, self.tag, dt=dt)


class RacePolicy:
    """A -> B -> A, driven by the lap phase from race/lap_timer.py."""

    def __init__(self, poles: dict, a_tag: int, b_tag: int, cfg: PolicyCfg | None = None):
        self.cfg = cfg or PolicyCfg()
        self.goto = GoTo(self.cfg)
        A, B = np.asarray(poles[a_tag], float), np.asarray(poles[b_tag], float)
        u = np.array([B[0] - A[0], 0.0, B[2] - A[2]])
        u /= np.linalg.norm(u)
        y = 0.5 * (A[1] + B[1]) + self.cfg.rel_height_m
        self.pole_a, self.pole_b = A, B
        self.gate_a = A + u * self.cfg.gate_inset_m
        self.gate_b = B - u * self.cfg.gate_inset_m
        self.park = A + u * self.cfg.park_m
        for point in (self.gate_a, self.gate_b, self.park):
            point[1] = y
        self.armed = False

    def plan(self, phase: str):
        """(target, look_at, speed_cap, line_from) for this lap phase."""
        line = None
        if phase == "FINISHED":
            target, cap = self.park, 0.4
        elif phase == "TURNED":
            target, cap, line = self.gate_a, None, self.gate_b
        elif phase == "OUTBOUND" or (phase == "AT_START" and self.armed):
            target, cap, line = self.gate_b, None, self.gate_a
        else:                       # IDLE / waiting at the start: settle in A's gate
            target, cap = self.gate_a, 0.4
        if self.cfg.look_at == "B":
            look = self.pole_b
        else:
            look = self.pole_b if target is self.gate_b else self.pole_a
        return target, look, cap, line

    def step(self, obs: Obs, phase: str, dt: float = 0.05):
        target, look, cap, line = self.plan(phase)
        rc, info = self.goto.step(obs, target, look, cap, dt, line_from=line)
        info["phase"] = phase
        return rc, info
