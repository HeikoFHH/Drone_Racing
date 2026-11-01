"""A simulated Tello flying toward a sign, behind the same interface as the real drone.

World frame: x from the take-off spot toward the sign, y to the left, z up. The sign hangs at
(distance, lateral, sign_height) facing back toward the take-off spot. The drone's heading theta is
counter-clockwise from +x; telemetry yaw is clockwise-positive with an arbitrary offset (like the real
state packet). rc sticks drive velocity through a first-order lag. Two perception modes:

  analytic   corners of the sign are projected with the real camera model, noised and gated (fast; tests)
  render     a 960x720 image is rendered and the real AprilTag detector runs on it (the web demo)

Everything the sim "knows" but the drone must not ( true position ) is only reachable through ``truth()``.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from .common import SignObs
from .drone import Clock, Drone, FrameResult, RealClock, Telemetry
from .perception import SignDetector
from .range_est import observation_from_corners
from .spec import CameraModel, SignSpec, load_camera

CAMERA_FORWARD_OFFSET_M = 0.03


@dataclass
class Scenario:
    distance_m: float = 8.0           # take-off spot to the sign plane
    lateral_m: float = 0.0            # sign to the left (+) / right (-) of the take-off line
    heading_error_deg: float = 0.0    # start heading error, counter-clockwise +
    sign_height_m: float = 1.0        # sign centre above the ground
    takeoff_height_m: float = 0.9
    # the true drone (a policy only knows its own estimate of these)
    mps_per_rc: float = 0.008
    deg_per_rc: float = 0.6
    up_mps_per_rc: float = 0.006
    tau_s: float = 0.35
    rc_deadband: int = 0
    # environment
    wind_mps: tuple[float, float] = (0.0, 0.0)
    wind_coupling: float = 0.35       # share of the wind the drone's own stabilisation lets through
    gust_sigma_mps: float = 0.0
    # sensing
    perception: str = "analytic"      # analytic | render
    video_delay_s: float = 0.10
    frame_hz: float = 15.0
    detect_prob: float = 0.97
    corner_noise_px: float = 0.35
    # telemetry
    yaw_offset_deg: float = 23.0
    gyro_drift_dps: float = 0.0
    height_noise_cm: float = 1.0
    speed_scale: float = 1.0          # what telemetry reports vs the truth
    battery_pct: float = 90.0


def object_corners(size: float) -> np.ndarray:
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def sign_pose_in_camera(cam_pos: np.ndarray, theta: float, sign_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R_ct, t_ct): sign frame -> camera frame (camera: x right, y down, z forward). Sign faces -x."""
    fwd = np.array([math.cos(theta), math.sin(theta), 0.0])
    left = np.array([-math.sin(theta), math.cos(theta), 0.0])
    right, down = -left, np.array([0.0, 0.0, -1.0])
    t_ct = np.array([(sign_pos - cam_pos) @ right, (sign_pos - cam_pos) @ down, (sign_pos - cam_pos) @ fwd])
    R_cw = np.vstack([right, down, fwd])
    R_wt = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # columns: tag x, y, normal in world axes
    return R_cw @ R_wt, t_ct


class SignRenderer:
    """The printed sign (tag36h11 with a white quiet zone) seen through the calibrated camera."""

    def __init__(self, cam: CameraModel, sign: SignSpec, tag_id: int = 1, seed: int = 0):
        self.cam, self.sign = cam, sign
        self.rng = np.random.default_rng(seed)
        W, H = cam.width, cam.height
        bg = self.rng.integers(95, 165, (H // 12, W // 12), dtype=np.uint8)
        bg = cv2.GaussianBlur(cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 9)
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        vignette = 1.0 - 0.18 * (xx ** 2 + yy ** 2)
        self.background = np.clip(bg * vignette, 0, 255).astype(np.uint8)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        px = 400
        m = cv2.aruco.generateImageMarker(d, tag_id, px, borderBits=1)
        pad = px // 8
        self.marker = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        self.src = np.float32([[pad, pad], [pad + px, pad], [pad + px, pad + px], [pad, pad + px]])

    def project(self, R_ct: np.ndarray, t_ct: np.ndarray) -> np.ndarray | None:
        obj = object_corners(self.sign.width_m)
        if t_ct[2] < 0.15 or float(R_ct[:, 2] @ t_ct) >= 0:       # behind the camera / seen from the back
            return None
        if ((R_ct @ obj.T).T + t_ct)[:, 2].min() < 0.05:
            return None
        pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R_ct)[0], t_ct, self.cam.K, self.cam.dist)
        return pts.reshape(4, 2)

    def render(self, R_ct: np.ndarray, t_ct: np.ndarray, noise: float = 3.0) -> np.ndarray:
        W, H = self.cam.width, self.cam.height
        img = self.background.copy()
        dst = self.project(R_ct, t_ct)
        if dst is not None and np.abs(dst).max() < 6 * W:
            Hm = cv2.getPerspectiveTransform(self.src, dst.astype(np.float32))
            warped = cv2.warpPerspective(self.marker, Hm, (W, H), flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full(self.marker.shape, 255, np.uint8), Hm, (W, H))
            img = np.where(mask > 127, warped, img)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        img = np.clip(img.astype(np.int16) + self.rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class SimDrone(Drone):
    live = False
    name = "simulated Tello"

    def __init__(self, scenario: Scenario | None = None, clock: Clock | None = None, sign: SignSpec | None = None,
                 cam: CameraModel | None = None, seed: int = 0):
        self.clock = clock or RealClock()
        self.sign = sign or SignSpec()
        self.cam = cam or load_camera()
        self.seed = seed
        self.lock = threading.RLock()
        self._detector: SignDetector | None = None
        self._renderer: SignRenderer | None = None
        self.reset(scenario or Scenario())

    # -- setup -----------------------------------------------------------------------------
    def reset(self, scenario: Scenario | None = None) -> None:
        with self.lock:
            self.sc = scenario or self.sc
            self.rng = np.random.default_rng(self.seed)
            self.sign_pos = np.array([self.sc.distance_m, self.sc.lateral_m, self.sc.sign_height_m])
            self.pos = np.zeros(3)
            self.vel = np.zeros(3)
            self.theta = math.radians(self.sc.heading_error_deg)
            self.yaw_rate = 0.0
            self.rc = (0, 0, 0, 0)
            self.phase = "ground"          # ground | takeoff | flying | landing | landed
            self.collided = False
            self.gust = np.zeros(2)
            self._t = self.clock.now()
            self._t0 = self._t
            self.hist: deque = deque(maxlen=400)
            self._hist_push()
            self._last_frame_t = -1e9
            self._last_result: FrameResult | None = None
            self._preview: np.ndarray | None = None
            self._renderer = None
            self.events: list[str] = []

    @property
    def renderer(self) -> SignRenderer:
        if self._renderer is None:
            self._renderer = SignRenderer(self.cam, self.sign, tag_id=self.sign.tag_id or 1, seed=self.seed)
        return self._renderer

    @property
    def detector(self) -> SignDetector:
        if self._detector is None:
            self._detector = SignDetector(self.sign, self.cam)
        return self._detector

    # -- physics ---------------------------------------------------------------------------
    def _hist_push(self) -> None:
        self.hist.append((self._t, self.pos.copy(), self.theta))

    def _advance(self) -> None:
        now = self.clock.now()
        total = now - self._t
        if total <= 0:
            return
        n = max(1, int(math.ceil(total / 0.02)))
        h = total / n
        for _ in range(n):
            self._t += h
            self._substep(h)
            self._hist_push()
        self._t = now

    def _substep(self, h: float) -> None:
        sc = self.sc
        if self.phase == "takeoff":
            gap = sc.takeoff_height_m - self.pos[2]
            self.pos[2] += max(min(0.55 * h, gap), 0.0) if gap > 0.03 else gap
            if gap <= 0.03:
                self.phase = "flying"
            return
        if self.phase == "landing":
            self.vel *= math.exp(-h / 0.3)
            self.pos[:2] += self.vel[:2] * h
            self.pos[2] = max(0.0, self.pos[2] - 0.28 * h)
            if self.pos[2] <= 0.0:
                self.phase, self.vel[:] = "landed", 0.0
                self.events.append("landed")
            return
        if self.phase != "flying":
            return
        r, f, u, y = (0 if abs(v) < sc.rc_deadband else v for v in self.rc)
        co, si = math.cos(self.theta), math.sin(self.theta)
        body = np.array([f * sc.mps_per_rc * co + r * sc.mps_per_rc * si,
                         f * sc.mps_per_rc * si - r * sc.mps_per_rc * co])
        if sc.gust_sigma_mps > 0:
            self.gust += -self.gust * h / 2.0 + sc.gust_sigma_mps * math.sqrt(2 * h / 2.0) * self.rng.normal(0, 1, 2)
        wind = sc.wind_coupling * (np.array(sc.wind_mps) + self.gust)
        target = np.array([body[0] + wind[0], body[1] + wind[1], u * sc.up_mps_per_rc])
        a = 1.0 - math.exp(-h / sc.tau_s)
        self.vel += (target - self.vel) * a
        self.pos += self.vel * h
        self.pos[2] = max(0.0, self.pos[2])
        target_rate = -y * sc.deg_per_rc * math.pi / 180.0
        self.yaw_rate += (target_rate - self.yaw_rate) * (1.0 - math.exp(-h / 0.15))
        self.theta += self.yaw_rate * h
        wall = self.sign_pos[0] - 0.02
        if self.pos[0] >= wall and abs(self.pos[1] - self.sign_pos[1]) < 0.45 and abs(self.pos[2] - self.sign_pos[2]) < 0.45:
            if not self.collided:
                self.events.append("COLLISION with the sign")
            self.collided = True
            self.pos[0], self.vel[:] = wall, 0.0

    def truth(self) -> dict:
        with self.lock:
            self._advance()
            cam = self.pos + CAMERA_FORWARD_OFFSET_M * np.array([math.cos(self.theta), math.sin(self.theta), 0.0])
            dx = self.sign_pos[0] - cam[0]
            return {"t": self._t, "x": float(self.pos[0]), "y": float(self.pos[1]), "z": float(self.pos[2]),
                    "theta_deg": math.degrees(self.theta), "depth_to_sign_m": float(
                        (self.sign_pos - cam) @ np.array([math.cos(self.theta), math.sin(self.theta), 0.0])),
                    "plane_distance_m": float(dx), "lateral_m": float(self.pos[1] - self.sign_pos[1]),
                    "collided": self.collided, "phase": self.phase,
                    "speed_mps": float(np.linalg.norm(self.vel[:2]))}

    # -- Drone interface -------------------------------------------------------------------
    def prepare(self) -> None:
        """Before a new mission: a landed (or crashed) sim drone goes back to the take-off spot."""
        with self.lock:
            if self.phase in ("landed", "ground"):
                self.reset(self.sc)

    def connect(self) -> bool:
        return True

    def start_video(self) -> bool:
        return True

    def telemetry(self) -> Telemetry:
        with self.lock:
            self._advance()
            sc = self.sc
            yaw = ((-math.degrees(self.theta) + sc.yaw_offset_deg + sc.gyro_drift_dps * (self._t - self._t0) + 180) % 360) - 180
            h = None if self.phase in ("ground",) and self.pos[2] == 0 else \
                round((self.pos[2] * 100 + self.rng.normal(0, sc.height_noise_cm))) / 100.0
            flying = self.phase in ("takeoff", "flying", "landing")
            bat = sc.battery_pct - 0.01 * (self._t - self._t0) if flying or self.phase == "landed" else sc.battery_pct
            speed = float(np.linalg.norm(self.vel[:2])) * sc.speed_scale + abs(self.rng.normal(0, 0.01))
            co, si = math.cos(self.theta), math.sin(self.theta)
            vb = (float((self.vel[0] * co + self.vel[1] * si) * sc.speed_scale + self.rng.normal(0, 0.02)),
                  float((self.vel[0] * si - self.vel[1] * co) * sc.speed_scale + self.rng.normal(0, 0.02)))
            return Telemetry(t=self._t, height_m=0.0 if h is None else h, yaw_deg=round(yaw), speed_mps=speed,
                             vel_body_mps=vb, battery_pct=bat, link_age_s=0.0, flying=flying)

    def _pose_at(self, t: float) -> tuple[np.ndarray, float]:
        best = min(self.hist, key=lambda e: abs(e[0] - t))
        return best[1], best[2]

    def _camera_pose(self, pos: np.ndarray, theta: float):
        cam = pos + CAMERA_FORWARD_OFFSET_M * np.array([math.cos(theta), math.sin(theta), 0.0])
        return sign_pose_in_camera(cam, theta, self.sign_pos)

    def _analytic_detections(self, R_ct, t_ct) -> list[SignObs]:
        sc = self.sc
        dst = self.renderer.project(R_ct, t_ct)
        if dst is None:
            return []
        W, H = self.cam.width, self.cam.height
        if not ((dst[:, 0] > 2) & (dst[:, 0] < W - 2) & (dst[:, 1] > 2) & (dst[:, 1] < H - 2)).all():
            return []                                            # a cut-off tag cannot be decoded
        side = min(np.linalg.norm(dst[(k + 1) % 4] - dst[k]) for k in range(4))
        if side < 8:
            return []
        p = sc.detect_prob * float(np.clip((side - 8.0) / 12.0, 0.0, 1.0))
        if self.rng.random() > p:
            return []
        noisy = dst + self.rng.normal(0, sc.corner_noise_px + 0.02 * abs(math.degrees(self.yaw_rate)), dst.shape)
        o = observation_from_corners(noisy, min(120.0, 30.0 + 0.8 * side), self.sign.tag_id or 1, self.cam, self.sign.width_m)
        return [] if o is None else [o]

    def poll_frame(self) -> FrameResult | None:
        with self.lock:
            self._advance()
            if self._t - self._last_frame_t < 1.0 / self.sc.frame_hz:
                return None
            self._last_frame_t = self._t
            pos, theta = self._pose_at(self._t - self.sc.video_delay_s)
            R_ct, t_ct = self._camera_pose(pos, theta)
            if self.sc.perception == "render":
                frame = self.renderer.render(R_ct, t_ct)
                dets = self.detector.detect(frame)
                self._preview = frame
            else:
                frame, dets = None, self._analytic_detections(R_ct, t_ct)
            self._last_result = FrameResult(self._t, frame, dets)
            return self._last_result

    def preview(self) -> np.ndarray | None:
        with self.lock:
            self._advance()
            R_ct, t_ct = self._camera_pose(self.pos, self.theta)
            return self.renderer.render(R_ct, t_ct)

    def takeoff(self) -> bool:
        with self.lock:
            if self.phase != "ground":
                return False
            self._advance()
            self.phase = "takeoff"
            self.events.append("takeoff")
        t0 = self.clock.now()
        while self.clock.now() - t0 < 15.0:
            self.clock.sleep(0.05)
            with self.lock:
                self._advance()
                if self.phase == "flying":
                    return True
        return False

    def land(self) -> bool:
        with self.lock:
            self._advance()
            if self.phase in ("ground", "landed"):
                return True
            self.phase = "landing"
            self.rc = (0, 0, 0, 0)
        t0 = self.clock.now()
        while self.clock.now() - t0 < 20.0:
            self.clock.sleep(0.05)
            with self.lock:
                self._advance()
                if self.phase == "landed":
                    return True
        return False

    def send_rc(self, right: int, forward: int, up: int, yaw: int) -> None:
        with self.lock:
            self._advance()
            c = lambda v: max(-100, min(100, int(v)))
            self.rc = (c(right), c(forward), c(up), c(yaw))
