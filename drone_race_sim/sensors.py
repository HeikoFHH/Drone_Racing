"""Sensing and state estimation for the race sim (SIMULATION_PLAN.md section 7).

What the real stack provides, modelled per env:
  * AprilTag fixes from the forward camera: field-of-view, range (apparent size) and viewing-angle
    gating, per-frame detection probability, range-dependent pose noise, a per-episode range scale
    error (the engine's f=871 vs 914.5), and video latency. A fix is consumed as if it were current,
    exactly like core/positional_kalman_filter.FilteredPositioner does.
  * The repo's own constant-velocity Kalman filter (``KalmanCV``, torch backend), with the same
    lost / restart-from-fix logic as ``FilteredPositioner.locate``.
  * Tello state packet: body velocity with noise, bias and scale error; height; drifting yaw.
  * A bridged position estimate: KF position while tags are tracked, dead reckoning on packet
    velocity while they are not (this is what an RL policy can use; ``kf_pos``/``lost`` is what the
    laptop wrapper reports).

Not modelled yet: rendered images/blur, ambiguity pose flips, dropout bursts, pole map error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from . import _repo  # noqa: F401
from .config import PARAMS_DIR
from core.positional_kalman_filter import KalmanCfg, KalmanCV  # noqa: E402

import yaml

BUF = 8  # true-state history length (ticks) for latency


@dataclass
class SensorCfg:
    image_width: float = 960.0
    image_height: float = 720.0
    fx: float = 914.54
    fy: float = 914.41
    tag_size_m: float = 0.13208
    min_side_px: float = 12.0
    max_view_deg: float = 70.0
    fov_margin: float = 0.92
    camera_pitch_deg: float = 0.0
    tag_z: float = 1.0
    video_delay_s: float = 0.15
    packet_delay_s: float = 0.07
    detector_hz: float = 12.0
    detect_prob: float = 0.95
    depth_noise_per_m2: float = 0.0058
    lateral_noise_rad: float = 0.015
    range_scale_sigma: float = 0.03
    fix_yaw_sigma_deg: float = 2.0
    vel_noise: float = 0.03
    vel_bias_sigma: float = 0.03
    vel_scale_sigma: float = 0.05
    height_noise: float = 0.02
    yaw_drift_deg_s: float = 0.2
    yaw_noise_deg: float = 0.5
    kf_lost_after_s: float = 0.4
    dead_reckoning_std_per_s: float = 0.05   # growth of the position std while only packet velocity is used
    noise_scale: float = 1.0   # global multiplier on fix noise (domain randomization hook)

    @property
    def hfov(self) -> float:
        return 2 * math.atan(self.image_width / 2 / self.fx)

    @property
    def vfov(self) -> float:
        return 2 * math.atan(self.image_height / 2 / self.fy)


def load_sensor_cfg(path=None) -> tuple[SensorCfg, str]:
    raw = yaml.safe_load(open(path or PARAMS_DIR / "sensors_placeholder.yaml")) or {}
    source = str(raw.pop("params_source", "unknown"))
    names = {f.name for f in fields(SensorCfg)}
    flat = {k: (v["value"] if isinstance(v, dict) else v) for k, v in raw.items() if k in names}
    return SensorCfg(**flat), source


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SensorModel:
    """Vectorized over E envs. Call ``reset`` for new episodes and ``step`` once per control tick."""

    def __init__(self, num_envs: int, tag_a_xy, tag_b_xy, cfg: SensorCfg, dt: float, device="cpu", dtype=torch.float32):
        self.n, self.cfg, self.dt = num_envs, cfg, dt
        self.device, self.dtype = torch.device(device), dtype
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)
        a = torch.as_tensor(tag_a_xy, device=self.device, dtype=dtype)
        b = torch.as_tensor(tag_b_xy, device=self.device, dtype=dtype)
        self.tag_pos = torch.stack([torch.cat([a, a.new_tensor([cfg.tag_z])]),
                                    torch.cat([b, b.new_tensor([cfg.tag_z])])])        # (2, 3)
        n_a = (b - a) / (b - a).norm()                                                  # tag A faces B, tag B faces A
        self.tag_n = torch.stack([n_a, -n_a])                                           # (2, 2)
        self.delay_steps = max(0, min(BUF - 1, round(cfg.video_delay_s / dt)))
        self.pkt_steps = max(0, min(BUF - 1, round(cfg.packet_delay_s / dt)))
        self.hist = z(num_envs, BUF, 7)           # x y z vx vy vz yaw  (index 0 = newest)
        self.kf = KalmanCV(KalmanCfg(lost_after=cfg.kf_lost_after_s), batch=(num_envs,), xp=torch, device=self.device)
        self.kf.p = self.kf.p.to(dtype)
        self.kf.v, self.kf.a, self.kf.b, self.kf.c = (x.to(dtype) for x in (self.kf.v, self.kf.a, self.kf.b, self.kf.c))
        self.kf.rejects = self.kf.rejects.to(dtype)
        # per-episode systematic errors
        self.range_scale = z(num_envs)
        self.vel_bias = z(num_envs, 3)
        self.vel_scale = z(num_envs)
        self.yaw_drift = z(num_envs)
        # outputs
        self.bridge_pos = z(num_envs, 3)
        self.vel_w = z(num_envs, 3)
        self.vel_body = z(num_envs, 3)
        self.height = z(num_envs)
        self.yaw_est = z(num_envs)
        self.yaw_rate = z(num_envs)
        self.fix_age = z(num_envs)
        self.lost = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.fix_now = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.visible = torch.zeros(num_envs, 2, dtype=torch.bool, device=self.device)
        self.seen = torch.zeros(num_envs, 2, dtype=torch.bool, device=self.device)
        self.tag_img = z(num_envs, 2, 3)          # u, v in [-1, 1], log range
        self.pos_std = z(num_envs)
        self.fix_pos = z(num_envs, 3)             # last accepted raw fix (course frame)
        self.age = z(num_envs)
        self.blackout = z(num_envs)               # seconds of forced tag blackout left (fault injection)
        self.est_at_fix = z(num_envs, 3)          # KF position at the last accepted fix (held between fixes)
        self.fix_time = z(num_envs)               # capture time of that fix (set by the env clock)

    # -- helpers ----------------------------------------------------------------------------------
    def _randn(self, *shape, gen=None):
        return torch.randn(*shape, device=self.device, dtype=self.dtype, generator=gen)

    def reset(self, idx, pos, vel, yaw, gen=None):
        n, c = len(idx), self.cfg
        cfg_delay = c.video_delay_s
        self.hist[idx] = torch.cat([pos, vel, yaw[:, None]], dim=-1)[:, None].expand(-1, BUF, -1)
        self.range_scale[idx] = c.range_scale_sigma * self._randn(n, gen=gen)
        self.vel_bias[idx] = c.vel_bias_sigma * self._randn(n, 3, gen=gen)
        self.vel_scale[idx] = c.vel_scale_sigma * self._randn(n, gen=gen)
        self.yaw_drift[idx] = math.radians(c.yaw_drift_deg_s) * self._randn(n, gen=gen)
        self.age[idx] = 0.0
        self.blackout[idx] = 0.0
        mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        mask[idx] = True
        self.kf.reset(mask)
        # armed in the A gate: the first frame localises the drone (a noisy fix seeds the filter)
        seed = pos + 0.1 * self._randn(n, 3, gen=gen)
        full = self.bridge_pos.clone()
        full[idx] = seed
        self.kf.update(full, mask)
        self.bridge_pos[idx] = seed
        self.fix_pos[idx] = seed
        self.yaw_est[idx] = yaw + math.radians(2.0) * self._randn(n, gen=gen)
        self.yaw_rate[idx] = 0.0
        self.vel_w[idx] = 0.0
        self.vel_body[idx] = 0.0
        self.height[idx] = pos[:, 2]
        self.fix_age[idx] = 0.0
        self.lost[idx] = False
        self.fix_now[idx] = False
        self.visible[idx] = False
        self.seen[idx] = False
        self.tag_img[idx] = 0.0
        self.pos_std[idx] = 0.1
        self.est_at_fix[idx] = seed
        self.fix_time[idx] = -cfg_delay

    def inject_blackout(self, duration_s: float, mask=None) -> None:
        """Fault injection: no tag is detected for ``duration_s`` (occlusion, glare, blur, a dropped video stream)."""
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.blackout = torch.where(m, torch.full_like(self.blackout, float(duration_s)), self.blackout)

    def _tag_view(self, k: int, pos_d, yaw_d):
        """Geometry of tag k as seen from the (delayed) true pose: visibility mask and image coords."""
        c = self.cfg
        rel = self.tag_pos[k] - pos_d                                   # drone -> tag, course frame
        co, si = torch.cos(yaw_d), torch.sin(yaw_d)
        f = co * rel[:, 0] + si * rel[:, 1]
        l = -si * rel[:, 0] + co * rel[:, 1]
        rng = rel.norm(dim=-1).clamp_min(1e-3)
        horiz = torch.hypot(f, l).clamp_min(1e-6)
        az = torch.atan2(l, f)
        el = torch.atan2(rel[:, 2], horiz) - math.radians(c.camera_pitch_deg)
        in_fov = (f > 0.2) & (az.abs() < c.hfov / 2 * c.fov_margin) & (el.abs() < c.vfov / 2 * c.fov_margin)
        hd = torch.hypot(rel[:, 0], rel[:, 1]).clamp_min(1e-6)
        cosv = (-(self.tag_n[k, 0] * rel[:, 0] + self.tag_n[k, 1] * rel[:, 1]) / hd)  # tag normal . (tag -> camera)
        side_px = c.fx * c.tag_size_m / rng * cosv.clamp_min(0.0)
        vis = in_fov & (cosv > math.cos(math.radians(c.max_view_deg))) & (side_px >= c.min_side_px)
        u = -torch.tan(az) * c.fx / (c.image_width / 2)
        v = -torch.tan(el) * c.fy / (c.image_height / 2)
        return vis, rel, rng, torch.stack([u.clamp(-2, 2), v.clamp(-2, 2), torch.log(rng)], dim=-1)

    # -- one control tick -------------------------------------------------------------------------
    def step(self, pos, vel, yaw, dt, gen=None, t=None):
        c, E = self.cfg, self.n
        self.hist = torch.cat([torch.cat([pos, vel, yaw[:, None]], dim=-1)[:, None], self.hist[:, :-1]], dim=1)
        d = self.hist[:, self.delay_steps]
        pos_d, yaw_d = d[:, :3], d[:, 6]
        pk = self.hist[:, self.pkt_steps]

        # --- tag fixes -----------------------------------------------------------------------
        frame_ok = torch.rand(E, device=self.device, dtype=self.dtype, generator=gen) < min(1.0, c.detector_hz * float(self.dt))
        fix_sum = torch.zeros(E, 3, device=self.device, dtype=self.dtype)
        n_fix = torch.zeros(E, device=self.device, dtype=self.dtype)
        vis_all = []
        det_all = []
        for k in range(2):
            vis, rel, rng, img = self._tag_view(k, pos_d, yaw_d)
            det = vis & frame_ok & (torch.rand(E, device=self.device, dtype=self.dtype, generator=gen) < c.detect_prob) & (self.blackout <= 0)
            sd = (c.depth_noise_per_m2 * rng ** 2 * c.noise_scale)[:, None]
            sl = (c.lateral_noise_rad * rng * c.noise_scale)[:, None]
            unit = rel / rng[:, None]
            hperp = torch.stack([-rel[:, 1], rel[:, 0]], dim=-1) / torch.hypot(rel[:, 0], rel[:, 1]).clamp_min(1e-6)[:, None]
            h3 = torch.cat([hperp, torch.zeros(E, 1, device=self.device, dtype=self.dtype)], dim=-1)
            vperp = torch.cross(unit, h3, dim=-1)
            noise = (sd * self._randn(E, 1, gen=gen) * unit + sl * self._randn(E, 1, gen=gen) * h3
                     + sl * self._randn(E, 1, gen=gen) * vperp)
            p_meas = self.tag_pos[k] - rel * (1 + self.range_scale)[:, None] + noise
            fix_sum = fix_sum + torch.where(det[:, None], p_meas, torch.zeros_like(p_meas))
            n_fix = n_fix + det.to(self.dtype)
            vis_all.append(vis)
            det_all.append(det)
            self.tag_img[:, k] = torch.where(vis[:, None], img, torch.zeros_like(img))
        got = n_fix > 0
        self.blackout = (self.blackout - dt).clamp_min(0.0)
        z_fix = fix_sum / n_fix.clamp_min(1.0)[:, None]
        self.visible = torch.stack(vis_all, dim=-1)
        self.seen = torch.stack(det_all, dim=-1)
        self.fix_now = got

        # --- Kalman filter, as FilteredPositioner.locate -----------------------------------------
        self.kf.predict(float(dt) if not torch.is_tensor(dt) else float(dt.mean()), active=~self.lost)
        back = got & (self.fix_age > c.kf_lost_after_s)
        if back.any():
            self.kf.reset(back)
        ok = self.kf.update(z_fix, got)
        self.fix_age = torch.where(got & ok, torch.zeros_like(self.fix_age), self.fix_age + dt)
        self.lost = (self.fix_age > c.kf_lost_after_s) | ~self.kf.ready
        self.kf.freeze(self.lost)
        self.fix_pos = torch.where((got & ok)[:, None], z_fix, self.fix_pos)
        self.est_at_fix = torch.where((got & ok)[:, None], self.kf.p, self.est_at_fix)
        if t is not None:
            self.fix_time = torch.where(got & ok, t - c.video_delay_s, self.fix_time)  # capture time, not arrival

        # --- state packet --------------------------------------------------------------------
        v_meas = pk[:, 3:6] * (1 + self.vel_scale)[:, None] + self.vel_bias + c.vel_noise * self._randn(E, 3, gen=gen)
        yaw_prev = self.yaw_est
        self.age = self.age + dt
        yaw_pkt = pk[:, 6] + self.yaw_drift * self.age + math.radians(c.yaw_noise_deg) * self._randn(E, gen=gen)
        # complementary heading: packet yaw, re-anchored by tag fixes (which carry absolute heading)
        offset_fix = wrap(yaw_d + math.radians(c.fix_yaw_sigma_deg) * self._randn(E, gen=gen) - yaw_pkt)
        self.yaw_off = getattr(self, "yaw_off", torch.zeros(E, device=self.device, dtype=self.dtype))
        self.yaw_off = torch.where(got & ok, self.yaw_off + 0.3 * wrap(offset_fix - self.yaw_off), self.yaw_off)
        self.yaw_est = wrap(yaw_pkt + self.yaw_off)
        self.yaw_rate = wrap(self.yaw_est - yaw_prev) / dt
        self.vel_w = v_meas
        co, si = torch.cos(self.yaw_est), torch.sin(self.yaw_est)
        self.vel_body = torch.stack([co * v_meas[:, 0] + si * v_meas[:, 1], -si * v_meas[:, 0] + co * v_meas[:, 1],
                                     v_meas[:, 2]], dim=-1)
        self.height = pk[:, 2] + c.height_noise * self._randn(E, gen=gen)

        # --- bridged estimate: KF while tracked, dead reckoning on packet velocity when lost -----
        dead = self.bridge_pos + v_meas * dt
        self.bridge_pos = torch.where(self.lost[:, None], dead, self.kf.p)
        self.bridge_pos = torch.cat([self.bridge_pos[:, :2], self.height[:, None]], dim=-1)
        std_kf = self.kf.a.mean(-1).clamp_min(0).sqrt()
        self.pos_std = torch.where(self.lost, std_kf + c.dead_reckoning_std_per_s * (self.fix_age - c.kf_lost_after_s).clamp_min(0), std_kf)
