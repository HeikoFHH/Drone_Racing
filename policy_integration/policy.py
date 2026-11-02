"""Bounded waypoint policy and shared guard for classic / learned controllers.

No networking, arming, takeoff, or SDK calls. A separate flight runner must consume
Decisions; request_land is advisory until that runner sends and acknowledges land.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

FRAME = "course_flu_v1"  # X A->B, Y left, Z up; yaw CCW from +X
ACTION_ORDER = ("right", "forward", "up", "yaw_clockwise")
ZERO = (0, 0, 0, 0)
Vec3 = tuple[float, float, float]


def finite(values):
    try:
        return all(type(x) in (int, float) and math.isfinite(x) for x in values)
    except (TypeError, OverflowError):
        return False


def clamp(x, limit):
    return max(-limit, min(limit, x))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


@dataclass(frozen=True)
class Observation:
    pose_time_s: float
    telemetry_time_s: float
    position_w: Vec3
    velocity_w: Vec3
    yaw_rad: float
    position_std_m: float
    velocity_std_mps: float
    battery_pct: float
    link_ok: bool
    airborne: bool
    enabled: bool
    frame: str = FRAME
    schema_version: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    rc: tuple[int, int, int, int]
    mode: str
    reason: str
    waypoint_index: int
    request_land: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ActionSample:
    """Latest output from a SEPARATE inference worker, not a blocking model call."""
    values: tuple[float, float, float, float]
    generated_at_s: float
    observation_pose_time_s: float
    model_id: str
    waypoint_index: int = 0
    action_order: tuple[str, ...] = ACTION_ORDER
    frame: str = FRAME


@dataclass(frozen=True)
class PolicyConfig:
    # Example route only: measure the actual course before selecting a route.
    waypoints_w: tuple[Vec3, ...] = ((0.0, 0.5, 1.0), (4.0, 0.5, 1.0), (0.0, 0.5, 1.0))
    # Optional simulation/review candidate: face a landmark for each route leg.
    # None preserves the existing fixed +X / backward-return baseline.
    heading_targets_w: tuple[Vec3, ...] | None = None
    poles_xy: tuple[tuple[float, float], ...] = ((0.0, 0.0), (4.0, 0.0))
    x_bounds: tuple[float, float] = (-1.0, 5.0)
    y_bounds: tuple[float, float] = (-1.0, 1.5)
    z_bounds: tuple[float, float] = (0.4, 2.0)
    pole_clearance_m: float = 0.3  # center distance; must include drone radius + margin
    max_pose_age_s: float = 0.25
    max_stale_hold_s: float = 0.5
    max_telemetry_age_s: float = 0.25
    land_after_pose_age_s: float = 0.75
    max_control_gap_s: float = 0.2
    max_session_s: float = 90.0
    max_action_age_s: float = 0.1
    max_position_std_m: float = 0.15
    max_velocity_std_mps: float = 0.25
    min_battery_pct: float = 25.0
    max_measured_speed_mps: float = 1.5
    waypoint_tolerance_m: float = 0.15
    arrival_speed_mps: float = 0.12
    arrival_dwell_s: float = 0.3
    kp: float = 0.7
    kd: float = 0.5
    yaw_kp: float = 1.2
    max_yaw_error_rad: float = 0.35
    turn_entry_speed_mps: float = 0.05
    rc_limit: int = 20
    rc_slew_per_s: float = 60.0
    # Approximate upstream action->speed model. Not a hardware calibration.
    horizontal_mps_at_100: float = 0.8
    vertical_mps_at_100: float = 0.6
    yaw_rps_at_100: float = math.pi / 3

    def __post_init__(self):
        scalar = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not finite(scalar) or any(v <= 0 for v in scalar):
            raise ValueError("policy limits and gains must be finite and positive")
        if type(self.rc_limit) is not int or not 1 <= self.rc_limit <= 100:
            raise ValueError("rc_limit must be an integer in [1, 100]")
        if self.land_after_pose_age_s <= self.max_pose_age_s:
            raise ValueError("landing timeout must exceed hold timeout")
        if self.min_battery_pct > 100:
            raise ValueError("battery threshold exceeds 100")
        for bounds in (self.x_bounds, self.y_bounds, self.z_bounds):
            if len(bounds) != 2 or not finite(bounds) or bounds[0] >= bounds[1]:
                raise ValueError("invalid geofence bounds")
        if not self.waypoints_w:
            raise ValueError("route is empty")
        if self.heading_targets_w is not None:
            if len(self.heading_targets_w) != len(self.waypoints_w):
                raise ValueError("one heading landmark is required per waypoint")
            for target in self.heading_targets_w:
                if len(target) != 3 or not finite(target):
                    raise ValueError("heading landmark must be a finite 3-vector")
        for p in self.poles_xy:
            if len(p) != 2 or not finite(p):
                raise ValueError("invalid pole")
        for goal in self.waypoints_w:
            if len(goal) != 3 or not finite(goal) or not self.inside(goal):
                raise ValueError("waypoint outside geofence or invalid")
            if any(math.hypot(goal[0] - p[0], goal[1] - p[1]) < self.pole_clearance_m for p in self.poles_xy):
                raise ValueError("waypoint inside pole clearance")
        for a, b in zip(self.waypoints_w, self.waypoints_w[1:]):
            if not self.segment_clear(a, b):
                raise ValueError("route segment crosses a pole clearance zone")

    def inside(self, p):
        return all(lo <= v <= hi for v, (lo, hi) in zip(p, (self.x_bounds, self.y_bounds, self.z_bounds)))

    def segment_clear(self, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = dx * dx + dy * dy
        for px, py in self.poles_xy:
            u = 0 if denom == 0 else max(0, min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / denom))
            if math.hypot(a[0] + u * dx - px, a[1] + u * dy - py) < self.pole_clearance_m:
                return False
        return True


class BaselinePolicy:
    """PD waypoint controller or guard for an async normalized-action candidate.

    ActionSamples share guards and route sequencing; they cannot bypass the envelope.
    This is NOT a loader for the repository's 21-feature PPO checkpoint.
    """

    def __init__(self, cfg: PolicyConfig | None = None, mode="classic", expected_model_id=None):
        self.cfg = cfg or PolicyConfig()
        if mode not in ("classic", "learned"):
            raise ValueError("mode must be classic or learned")
        if mode == "learned" and (not isinstance(expected_model_id, str) or not expected_model_id):
            raise ValueError("learned mode requires an explicitly selected model ID")
        self.mode = mode
        self.expected_model_id = expected_model_id
        self.waypoint_index = 0
        self._last_now = None
        self._last_pose = None
        self._last_telemetry = None
        self._accepted_pose = None
        self._prev_rc = ZERO
        self._dwell = 0.0
        self._arrival_pose_time = None
        self._fault = None
        self._completed = False
        self._stale_since = None
        self._session_started_s = None
        self._braking = False

    def _zero(self, mode, reason, land=False, latch=False):
        self._prev_rc = ZERO
        self._dwell = 0.0
        self._arrival_pose_time = None
        self._braking = False
        if latch:
            self._fault = reason
        return Decision(ZERO, mode, reason, self.waypoint_index, land)

    def step(self, obs: Observation, now_s: float, candidate_action: ActionSample | None = None) -> Decision:
        c = self.cfg
        if self._fault:
            return self._zero("abort", self._fault, land=True)
        try:
            good_shape = len(obs.position_w) == len(obs.velocity_w) == 3
            scalars = (now_s, obs.pose_time_s, obs.telemetry_time_s, obs.yaw_rad,
                       obs.position_std_m, obs.velocity_std_mps, obs.battery_pct)
            valid = good_shape and finite((*scalars, *obs.position_w, *obs.velocity_w))
            valid = valid and all(type(v) is bool for v in (obs.link_ok, obs.airborne, obs.enabled))
            valid = valid and 0 <= obs.battery_pct <= 100 and obs.position_std_m >= 0 and obs.velocity_std_mps >= 0
            valid = valid and obs.frame == FRAME and type(obs.schema_version) is int and obs.schema_version == 1
        except (TypeError, AttributeError):
            valid = False
        if not valid:
            return self._zero("abort", "invalid_observation", land=True, latch=True)
        if obs.airborne:
            if self._session_started_s is None:
                self._session_started_s = now_s
            if now_s < self._session_started_s:
                return self._zero("abort", "control_clock_or_deadline", land=True, latch=True)
            if now_s - self._session_started_s >= c.max_session_s:
                return self._zero("abort", "session_timeout", land=True, latch=True)
        if not obs.enabled:
            self._last_now = None
            self._stale_since = None
            return self._zero("disabled", "operator_not_enabled")
        if not obs.airborne:
            self._last_now = None
            self._stale_since = None
            return self._zero("ground", "takeoff_owned_by_runner")
        dt = 0.05 if self._last_now is None else now_s - self._last_now
        if dt <= 0 or dt > c.max_control_gap_s:
            return self._zero("abort", "control_clock_or_deadline", land=True, latch=True)
        self._last_now = now_s
        if not obs.link_ok:
            return self._zero("abort", "link_lost", land=True, latch=True)
        if obs.battery_pct < c.min_battery_pct:
            return self._zero("abort", "low_battery", land=True, latch=True)
        pose_age, telemetry_age = now_s - obs.pose_time_s, now_s - obs.telemetry_time_s
        if pose_age < 0 or telemetry_age < 0:
            return self._zero("abort", "future_timestamp", land=True, latch=True)
        if ((self._last_pose is not None and obs.pose_time_s < self._last_pose)
                or (self._last_telemetry is not None and obs.telemetry_time_s < self._last_telemetry)):
            return self._zero("abort", "out_of_order_timestamp", land=True, latch=True)
        self._last_pose, self._last_telemetry = obs.pose_time_s, obs.telemetry_time_s
        if telemetry_age > c.max_telemetry_age_s or pose_age > c.land_after_pose_age_s:
            return self._zero("abort", "sensors_timed_out", land=True, latch=True)
        if pose_age > c.max_pose_age_s:
            if self._stale_since is None:
                self._stale_since = now_s
            if now_s - self._stale_since >= c.max_stale_hold_s:
                return self._zero("abort", "persistent_pose_stale", land=True, latch=True)
            return self._zero("hold", "pose_stale")
        self._stale_since = None
        if obs.position_std_m > c.max_position_std_m or obs.velocity_std_mps > c.max_velocity_std_mps:
            return self._zero("abort", "estimator_uncertain", land=True, latch=True)
        if not c.inside(obs.position_w):
            return self._zero("abort", "geofence", land=True, latch=True)
        if norm(obs.velocity_w) > c.max_measured_speed_mps:
            return self._zero("abort", "excess_speed", land=True, latch=True)
        if self._accepted_pose is not None:
            previous_time, previous_position, previous_std = self._accepted_pose
            elapsed = obs.pose_time_s - previous_time
            movement = norm(tuple(p - q for p, q in zip(obs.position_w, previous_position)))
            if elapsed == 0 and movement > 1e-6:
                return self._zero("abort", "pose_changed_without_new_frame", land=True, latch=True)
            allowance = c.max_measured_speed_mps * elapsed + 3 * (obs.position_std_m + previous_std)
            if elapsed > 0 and movement > allowance:
                return self._zero("abort", "implausible_pose_jump", land=True, latch=True)
        self._accepted_pose = (obs.pose_time_s, tuple(obs.position_w), obs.position_std_m)
        if self._completed:
            return self._zero("complete", "route_complete_timer_must_confirm", land=True)
        goal = c.waypoints_w[self.waypoint_index]
        if not c.segment_clear(obs.position_w, goal):
            return self._zero("abort", "path_near_pole", land=True, latch=True)
        err = tuple(g - p for g, p in zip(goal, obs.position_w))
        heading_target = 0.0
        if c.heading_targets_w is not None:
            landmark = c.heading_targets_w[self.waypoint_index]
            dx, dy = landmark[0] - obs.position_w[0], landmark[1] - obs.position_w[1]
            if math.hypot(dx, dy) < c.pole_clearance_m:
                return self._zero("abort", "heading_landmark_too_close", land=True, latch=True)
            heading_target = math.atan2(dy, dx)
        yaw_error = math.atan2(math.sin(heading_target - obs.yaw_rad),
                               math.cos(heading_target - obs.yaw_rad))
        aiming = abs(yaw_error) > c.max_yaw_error_rad
        braking = aiming and math.hypot(*obs.velocity_w[:2]) > c.turn_entry_speed_mps
        if (not aiming and norm(err) <= c.waypoint_tolerance_m
                and norm(obs.velocity_w) <= c.arrival_speed_mps):
            # Repeated video frames cannot prove a continuous dwell.
            if self._arrival_pose_time is not None and obs.pose_time_s > self._arrival_pose_time:
                self._dwell += min(obs.pose_time_s - self._arrival_pose_time, dt, 0.1)
            self._arrival_pose_time = obs.pose_time_s
            if self._dwell >= c.arrival_dwell_s:
                self._dwell = 0.0
                if self.waypoint_index == len(c.waypoints_w) - 1:
                    self._completed = True
                    return self._zero("complete", "route_complete_timer_must_confirm", land=True)
                self.waypoint_index += 1
                return self._zero("waypoint", "advance")
        else:
            self._dwell = 0.0
            self._arrival_pose_time = None
        co, si = math.cos(obs.yaw_rad), math.sin(obs.yaw_rad)
        if self.mode == "classic":
            desired_w = tuple(c.kp * e - c.kd * v for e, v in zip(err, obs.velocity_w))
            forward = co * desired_w[0] + si * desired_w[1]
            right = si * desired_w[0] - co * desired_w[1]
            yaw_clockwise = -c.yaw_kp * yaw_error
            raw = (100 * right / c.horizontal_mps_at_100,
                   100 * forward / c.horizontal_mps_at_100,
                   100 * desired_w[2] / c.vertical_mps_at_100,
                   100 * yaw_clockwise / c.yaw_rps_at_100)
        else:
            try:
                sample = candidate_action
                action = tuple(sample.values)
                if len(action) != 4 or not finite(action) or any(abs(v) > 1 for v in action):
                    raise ValueError("invalid normalized action")
                if (sample.frame != FRAME or tuple(sample.action_order) != ACTION_ORDER
                        or sample.model_id != self.expected_model_id
                        or type(sample.waypoint_index) is not int or sample.waypoint_index != self.waypoint_index
                        or not finite((sample.generated_at_s, sample.observation_pose_time_s))
                        or not 0 <= now_s - sample.generated_at_s <= c.max_action_age_s
                        or sample.observation_pose_time_s != obs.pose_time_s
                        or sample.generated_at_s < sample.observation_pose_time_s):
                    raise ValueError("action metadata mismatch or stale action")
                raw = tuple(100 * v for v in action)
            except Exception:
                return self._zero("abort", "candidate_failed", land=True, latch=True)
        # A heading correction must not assume zero RC stops existing momentum.
        # When the estimate is fresh, first oppose measured horizontal velocity;
        # wait to yaw until speed is low. This is not a calibrated stopping bound.
        if braking:
            if not self._braking:
                # Discard prior forward/sideways demand at brake entry. Slewing
                # from that old body-frame command after a yaw kick is unsafe.
                self._prev_rc = (0.0, 0.0, self._prev_rc[2], self._prev_rc[3])
            vx, vy = (-c.kd * value for value in obs.velocity_w[:2])
            raw = (100 * (si * vx - co * vy) / c.horizontal_mps_at_100,
                   100 * (co * vx + si * vy) / c.horizontal_mps_at_100,
                   raw[2], 0.0)
        # Scale the horizontal pair together. Independent clipping changes its
        # world direction when the body is yawed, creating sideways drift.
        horizontal = math.hypot(raw[0], raw[1])
        if horizontal > c.rc_limit:
            scale = c.rc_limit / horizontal
            raw = (raw[0] * scale, raw[1] * scale, raw[2], raw[3])
        limited = tuple(clamp(v, c.rc_limit) for v in raw)
        # Accumulate fractional slew so high-rate callers do not freeze at RC=0.
        delta = tuple(v - p for p, v in zip(self._prev_rc, limited))
        largest = max(abs(delta[0]), abs(delta[1]))
        fraction = min(1.0, c.rc_slew_per_s * dt / largest) if largest else 1.0
        # One fraction keeps the horizontal interpolation inside the vector cap.
        slewed = (self._prev_rc[0] + fraction * delta[0],
                  self._prev_rc[1] + fraction * delta[1],
                  self._prev_rc[2] + clamp(delta[2], c.rc_slew_per_s * dt),
                  self._prev_rc[3] + clamp(delta[3], c.rc_slew_per_s * dt))
        if aiming and not braking:
            slewed = (0.0, 0.0, slewed[2], slewed[3])
        self._prev_rc = slewed
        self._braking = braking
        return Decision(tuple(int(round(v)) for v in slewed), "brake" if braking else "aim" if aiming else "track",
                        "candidate" if self.mode == "learned" else "classic_pd", self.waypoint_index)
