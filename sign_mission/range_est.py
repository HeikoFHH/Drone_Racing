"""Distance to the sign: three independent cues, one Kalman filter, a near-field fallback.

Why not just use the pose solution's distance
  A single AprilTag pose (IPPE) gives ``tvec``, but its depth is the noisiest part of it: it wobbles with
  corner noise, and at long range or near-frontal views the solution can flip between two poses. The sign is
  a known size (22.5 in), so the apparent size itself is a second and third estimate:

    z_pnp     depth from the full pose solution
    z_width   fx * W * cos(yaw of sign)   / width in px     (tag-horizontal edges)
    z_height  fy * W * cos(pitch of sign) / height in px    (tag-vertical edges)

  The cues share the same corner noise, so averaging them does not make a single frame more accurate (measured:
  scripts/sign_range_accuracy.py). They are kept as a consistency check: while they agree the pose depth is used,
  when the pose solution disagrees with the other two (a flipped pose, a bad corner) the median is used, and the
  disagreement widens the measurement noise. What does reduce the error is the 1-D Kalman filter, driven by the
  drone's own forward speed: it smooths the frame noise (about 2x lower RMSE, 3-4x lower worst case), bridges
  missed frames, and rejects single-frame outliers (4-sigma gate).

Near field
  A 22.5 in sign no longer fits completely in the 960x720 image closer than about 0.75 m, and the detector
  needs all four corners. Below ``near_start_m`` the filter keeps running on the drone's speed alone
  (``source="near_field"``); the standard deviation grows with the distance travelled blind.

Range means depth: distance from the camera to the sign plane along the camera's viewing axis. That is the
distance "in front of" the sign when the sign is centred in the image.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .common import RangeEstimate, SignObs
from .spec import CameraModel, SignSpec, load_camera


def observation_from_corners(corners_ippe, margin: float, tag_id: int, cam: CameraModel, width_m: float,
                             edge_margin_px: float = 8.0) -> SignObs | None:
    """Derive all geometry of one detection. ``corners_ippe``: (4, 2) raw px, order TL, TR, BR, BL of the tag."""
    c = np.asarray(corners_ippe, np.float64).reshape(4, 2)
    und = cv2.undistortPoints(c.reshape(-1, 1, 2), cam.K, cam.dist, P=cam.K).reshape(-1, 2)
    h = width_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, und, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not len(rvecs):
        return None
    rvec, tvec = np.ravel(rvecs[0]), np.ravel(tvecs[0])
    if tvec[2] <= 0:
        if len(rvecs) > 1 and np.ravel(tvecs[1])[2] > 0:
            rvec, tvec = np.ravel(rvecs[1]), np.ravel(tvecs[1])
        else:
            return None
    R = cv2.Rodrigues(rvec)[0]
    n = R[:, 2]                                           # sign normal in camera axes
    n = n if n[2] < 0 else -n                             # pointing back toward the camera
    cos_h = abs(n[2]) / max(math.hypot(n[0], n[2]), 1e-9)  # sign yaw relative to the image plane
    cos_v = abs(n[2]) / max(math.hypot(n[1], n[2]), 1e-9)  # sign pitch relative to the image plane
    w_px = 0.5 * (np.linalg.norm(und[1] - und[0]) + np.linalg.norm(und[2] - und[3]))
    h_px = 0.5 * (np.linalg.norm(und[3] - und[0]) + np.linalg.norm(und[2] - und[1]))
    centre = und.mean(axis=0)
    to_cam = -tvec / np.linalg.norm(tvec)
    view = math.degrees(math.acos(float(np.clip(n @ to_cam, -1, 1))))
    inside = bool(((c[:, 0] > edge_margin_px) & (c[:, 0] < cam.width - edge_margin_px)
                   & (c[:, 1] > edge_margin_px) & (c[:, 1] < cam.height - edge_margin_px)).all())
    return SignObs(
        tag_id=int(tag_id), corners=c, margin=float(margin), z_pnp=float(tvec[2]),
        z_width=float(cam.fx * width_m * max(cos_h, 0.85) / max(w_px, 1e-6)),
        z_height=float(cam.fy * width_m * max(cos_v, 0.85) / max(h_px, 1e-6)),
        tan_x=float((centre[0] - cam.cx) / cam.fx), tan_y=float((centre[1] - cam.cy) / cam.fy),
        width_px=float(w_px), height_px=float(h_px), view_deg=view, fully_visible=inside,
        normal_cam=(float(n[0]), float(n[1]), float(n[2])))


class RangeCfg:
    min_side_px = 12.0           # smaller detections are ignored (repo gate)
    max_view_deg = 75.0          # sign seen at a steeper angle than this: pose and width cues unreliable
    min_margin = 8.0             # detector decision margin
    corner_sigma_px = 0.4        # corner noise, px
    floor_sigma_m = 0.015        # measurement noise floor
    gate_sigma = 4.0             # innovation gate
    reinit_rejects = 4           # this many gated measurements in a row: the estimate was wrong, restart from the data
    speed_sigma_abs = 0.08       # m/s uncertainty of the speed used to predict
    speed_sigma_frac = 0.30      # plus this fraction of the speed (rc -> m/s scale is not known exactly)
    q_range = 0.01               # extra process noise, m^2 per s
    near_margin = 1.25           # near field starts at this multiple of the fully-visible range
    far_loss_dr_s = 1.5          # lost far away: dead-reckon at most this long, then report "not tracking"
    near_dr_s = 12.0             # lost in the near field (expected): dead-reckon this long at most
    min_range_m = 0.10
    max_range_m = 80.0


class SignRangeEstimator:
    def __init__(self, sign: SignSpec | None = None, cam: CameraModel | None = None, cfg: RangeCfg | None = None):
        self.sign = sign or SignSpec()
        self.cam = cam or load_camera()
        self.cfg = cfg or RangeCfg()
        self.fit_range_m = self.cam.min_range_fully_visible(self.sign.width_m)
        self.near_start_m = self.cfg.near_margin * self.fit_range_m
        self.reset()

    def reset(self) -> None:
        self.t_last: float | None = None
        self.r = None            # filtered range
        self.var = 1.0
        self.last_det_t: float | None = None
        self.last_vis_r: float | None = None      # range at the last accepted measurement
        self.rejects = 0
        self.bearing = 0.0
        self.lateral = 0.0
        self.vertical = 0.0
        self.tracking = False
        self.det_count = 0
        self.last_fully_visible = False

    # -- cues ---------------------------------------------------------------------------------
    def _usable(self, o: SignObs) -> bool:
        c = self.cfg
        if self.sign.tag_id is not None and o.tag_id != self.sign.tag_id:
            return False
        return o.side_px >= c.min_side_px and o.view_deg <= c.max_view_deg and o.margin >= c.min_margin

    @staticmethod
    def fuse(o: SignObs) -> tuple[float, float]:
        """(depth, relative disagreement of the three cues)."""
        cues = sorted((o.z_pnp, o.z_width, o.z_height))
        mid = cues[1]
        z = o.z_pnp if abs(o.z_pnp - mid) <= 0.05 * mid else mid       # trust the pose unless the other cues contradict it
        return z, (cues[2] - cues[0]) / max(mid, 1e-6)

    def _sigma(self, o: SignObs, z: float, spread: float) -> float:
        c = self.cfg
        rel = max(0.003, 1.5 * c.corner_sigma_px / max(o.side_px, 1.0))
        return z * (rel + 0.5 * spread) + c.floor_sigma_m

    # -- one tick ----------------------------------------------------------------------------
    def step(self, t: float, detections: list[SignObs], closing_mps: float = 0.0) -> RangeEstimate:
        """``closing_mps``: how fast the drone is moving toward the sign (commanded speed x calibration, or telemetry)."""
        c = self.cfg
        dt = 0.0 if self.t_last is None else max(0.0, t - self.t_last)
        self.t_last = t
        # predict
        if self.r is not None and dt > 0:
            self.r -= closing_mps * dt
            sv = c.speed_sigma_abs + c.speed_sigma_frac * abs(closing_mps)
            self.var += (sv * dt) ** 2 + c.q_range * dt
        cand = [o for o in detections if self._usable(o)]
        est = RangeEstimate(t=t, closing_mps=closing_mps)
        if cand:
            o = max(cand, key=lambda d: d.side_px)             # the closest / largest = the sign
            z, spread = self.fuse(o)
            est.detected, est.naive_m, est.raw_m, est.spread = True, o.z_pnp, z, spread
            est.fully_visible = o.fully_visible
            self.last_fully_visible = o.fully_visible
            self.bearing = math.degrees(math.atan(o.tan_x))
            self.lateral = z * o.tan_x
            self.vertical = -z * o.tan_y                       # + = sign above the camera
            if not o.fully_visible:                            # corners at the image border: shape is cut, do not trust
                est.note = "sign cut by the image border"
            elif c.min_range_m < z < c.max_range_m:
                s = self._sigma(o, z, spread)
                lost = self.last_det_t is None or not self.tracking
                if self.r is None or lost:
                    self.r, self.var, self.rejects = z, s * s, 0
                else:
                    innov, S = z - self.r, self.var + s * s
                    if innov * innov > (c.gate_sigma ** 2) * S:
                        self.rejects += 1
                        est.note = "outlier rejected"
                        if self.rejects >= c.reinit_rejects:
                            self.r, self.var, self.rejects = z, s * s, 0
                            est.note = "restarted from data"
                    else:
                        k = self.var / S
                        self.r += k * innov
                        self.var = (1 - k) * self.var
                        self.rejects = 0
                self.last_det_t, self.last_vis_r = t, z
                self.det_count += 1
        if self.r is None:
            return est
        age = math.inf if self.last_det_t is None else t - self.last_det_t
        est.age_s = age
        est.fully_visible = est.fully_visible or (self.last_fully_visible and age <= 0.35)
        near = self.last_vis_r is not None and (self.last_vis_r <= self.near_start_m * 1.2 or self.r <= self.near_start_m)
        if age <= 0.35:
            self.tracking, est.source = True, "vision"
        elif near and age <= c.near_dr_s:
            self.tracking, est.source, est.near_field = True, "near_field", True
        elif age <= c.far_loss_dr_s:
            self.tracking, est.source = True, "dead_reckoning"
        else:
            self.tracking, est.source = False, "none"
        est.tracking = self.tracking
        if self.tracking:
            est.range_m = max(c.min_range_m, self.r)
            est.std_m = math.sqrt(max(self.var, 0.0))
            est.near_field = est.near_field or est.range_m <= self.near_start_m
        est.bearing_deg, est.lateral_m, est.vertical_m = self.bearing, self.lateral, self.vertical
        return est
