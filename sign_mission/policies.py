"""Flight policies for the sign mission.

ApproachPolicy     SETTLE -> (SEARCH) -> ALIGN -> APPROACH -> FINAL -> STOP          (mode "done" at the stop)
OutAndBackPolicy   ... STOP -> TURN (180 deg) -> RETURN (same distance) -> STOP2     (mode "done" back at the start)

The runner does take-off before the policy and landing (LandingPolicy) after it.

How it stops 0.5 m in front of a 22.5 in sign
  * far: speed follows distance (slow near the sign) and a braking curve that allows for the command lag
  * the sign stops fitting in the camera image at about 0.75 m, so inside ``near_start_m`` (0.93 m) the
    last stretch is flown at creep speed on the distance estimate alone (range_est: near-field dead reckoning)
  * the distance flown is checked against what the camera saw: the true speed per stick unit is learned
    during the approach (``scale``), which also keeps the dead reckoning honest in the near field
Heading, height and the sign's vertical position are all closed loops; the sideways stick only trims the
last 3 m. Safety limits end the mission (mode "abort", the runner lands): session time, height, lost
telemetry, battery, range not closing, overshoot, sign never found.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .common import Action, RangeEstimate, RawObs
from .range_est import RangeCfg, SignRangeEstimator
from .spec import STOP_DISTANCE_M, CameraModel, SignSpec


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


@dataclass
class PolicyCfg:
    stop_m: float = STOP_DISTANCE_M        # camera -> sign plane at the stop
    stop_hard_m: float = 0.38              # never fly forward closer than this
    cruise_mps: float = 0.8                # far-field speed (also limited by rc_limit * mps_per_rc * scale)
    creep_mps: float = 0.18                # speed in the last stretch
    min_mps: float = 0.12                  # the Tello ignores tiny sticks: never ask for less while approaching
    probe_mps: float = 0.20                # until the speed per stick unit is measured
    slow_gain: float = 0.35                # speed cap = slow_gain * distance
    brake_mps2: float = 0.35               # braking curve
    latency_s: float = 0.5                 # brake as if the sign were this much flight time closer
    final_latency_s: float = 0.25
    mps_per_rc: float = 0.008              # assumed metres/second per stick unit (learned online as `scale`)
    rc_limit: int = 45                     # forward stick cap
    tau_s: float = 0.35                    # response lag assumed for the speed model
    # heading, sideways, height
    yaw_sign: float = 1.0                  # +1: telemetry yaw grows on a right (clockwise) turn
    k_yaw: float = 1.2
    max_yaw_rc: int = 35
    align_deg: float = 4.0
    align_hold_s: float = 0.4
    turn_first_deg: float = 20.0           # bearing larger than this: turn on the spot, do not fly
    k_lat: float = 25.0                    # sideways stick per metre the sign is off to one side
    ki_lat: float = 12.0                   # integral: absorbs a steady cross-wind
    max_lat_rc: int = 30
    lat_active_m: float = 20.0
    bearing_trim_deg: float = 8.0          # yaw follows the sign only beyond this; inside it the sideways stick does it
    heading_follow: float = 0.15           # fraction of the excess bearing taken into the heading reference per tick
    k_up: float = 80.0
    max_up_rc: int = 35
    height_min_m: float = 0.5
    height_max_m: float = 2.0
    height_follow_m: float = 0.6           # follow the sign's height at most this far from the height after take-off
    # search
    search_yaw_rc: int = 20
    search_max_deg: float = 380.0
    # stop
    stop_tol_m: float = 0.04
    settle_speed_mps: float = 0.10
    stop_hold_s: float = 1.0
    stop_min_s: float = 1.5
    k_damp: float = 1.0                    # fraction of a measured ground speed the sticks cancel while holding / returning
    k_pos: float = 0.35                    # ... plus this per second of the sideways drift accumulated (position hold)
    max_damp_rc: int = 30
    k_hold: float = 0.8                    # while stopped: forward speed (m/s) per metre of distance error (range hold)
    max_hold_rc: int = 25
    k_hold_i: float = 0.5                  # ... and per second of accumulated error (a steady push needs an integral)
    hold_band_m: float = 0.02
    stop_accept_m: float = 0.20            # a stop farther than stop_m + this is not accepted: approach again
    max_rearms: int = 3
    stop_max_s: float = 4.0                # stop waiting for a perfectly still drone after this (drift in wind never settles)
    # turn and return (OutAndBackPolicy)
    turn_deg: float = 180.0
    turn_yaw_rc: int = 35
    turn_min_yaw_rc: int = 14
    turn_tol_deg: float = 4.0
    return_cruise_mps: float = 0.6
    return_hold_s: float = 0.6
    # safety (any -> mode "abort")
    max_session_s: float = 150.0
    max_height_m: float = 2.4
    max_no_telemetry_s: float = 2.0
    min_battery_pct: float = 12.0
    not_closing_s: float = 5.0
    overshoot_frac: float = 0.35


class _Calib:
    """Learns true speed per stick unit relative to the assumed one, from the distance the camera sees closing."""

    def __init__(self) -> None:
        self.sum_dr = 0.0
        self.sum_dd = 0.0
        self.ref = None            # (t, range, unscaled_distance)
        self.scale = 1.0
        self.segments = 0

    def update(self, t: float, vision: bool, r: float | None, d_unscaled: float, window: float = 1.2) -> None:
        if not vision or r is None:
            self.ref = None
            return
        if self.ref is None:
            self.ref = (t, r, d_unscaled)
            return
        t0, r0, d0 = self.ref
        if t - t0 >= window:
            dd = d_unscaled - d0
            if dd > 0.2:
                self.sum_dr += r0 - r
                self.sum_dd += dd
                self.segments += 1
                self.scale = min(6.0, max(0.2, self.sum_dr / self.sum_dd)) if self.sum_dd > 0 else 1.0
            self.ref = (t, r, d_unscaled)

    @property
    def known(self) -> bool:
        return self.segments >= 1


class ApproachPolicy:
    name = "approach"
    title = "Fly to the sign, stop 0.5 m in front of it"

    def __init__(self, cfg: PolicyCfg | None = None, sign: SignSpec | None = None, cam: CameraModel | None = None,
                 range_cfg: RangeCfg | None = None):
        self.cfg = cfg or PolicyCfg()
        self.sign = sign or SignSpec()
        self.est_filter = SignRangeEstimator(self.sign, cam, range_cfg)
        self.est = RangeEstimate()
        self.state = "SETTLE"
        self.t0: float | None = None
        self.t_state = 0.0
        self.t_prev: float | None = None
        self.v_nominal = 0.0                # modelled forward speed assuming the nominal mps_per_rc, m/s
        self.cal = _Calib()
        self.d_unscaled = 0.0               # distance flown forward, nominal units
        self.h_ref: float | None = None
        self.h_target: float | None = None
        self.align_since: float | None = None
        self.r_start: float | None = None
        self.final_range: float | None = None
        self.final_std: float | None = None
        self.final_entry: tuple | None = None
        self.hold_heading: float | None = None
        self.stop_since: float | None = None
        self.rate_hist: deque = deque(maxlen=60)
        self.search_turned = 0.0
        self.search_prev: float | None = None
        self.no_height_since: float | None = None
        self.no_yaw_since: float | None = None
        self.abort_reason: str | None = None
        self.not_closing_since: float | None = None
        self.last_fwd_rc = 0
        self.v_meas: float | None = None
        self.hold_int = 0.0
        self.drift_int = 0.0
        self._dt = 0.05
        self.rearms = 0
        self.heading_ref: float | None = None
        self.lat_i = 0.0
        self.events: list[str] = []
        self.done = False

    # -- helpers -------------------------------------------------------------------------------
    @property
    def scale(self) -> float:
        return self.cal.scale

    def _set(self, state: str, t: float, note: str = "") -> None:
        self.events.append(f"{t - (self.t0 or t):5.1f}s {self.state} -> {state}" + (f" ({note})" if note else ""))
        self.state, self.t_state = state, t

    def _rc_for(self, v_mps: float) -> int:
        """Stick for a forward speed, with the learned speed-per-stick scale."""
        c = self.cfg
        rc = v_mps / (c.mps_per_rc * max(self.scale, 0.2))
        return int(round(clamp(rc, c.rc_limit)))

    def _model_speed(self, dt: float) -> None:
        a = 1.0 - math.exp(-dt / self.cfg.tau_s)
        self.v_nominal += (self.last_fwd_rc * self.cfg.mps_per_rc - self.v_nominal) * a
        self.d_unscaled += self.v_nominal * dt

    @property
    def v_est(self) -> float:
        """Estimated true forward speed, m/s: measured ground speed when the drone reports a trusted one (this also
        accounts for wind), otherwise the commanded speed through the lag model and the learned scale."""
        return self.v_meas if self.v_meas is not None else self.v_nominal * self.scale

    def _height_rc(self, raw: RawObs, est: RangeEstimate) -> int:
        c = self.cfg
        if raw.height_m is None:
            return 0
        if self.h_ref is None:
            self.h_ref = min(max(raw.height_m, c.height_min_m), c.height_max_m - 0.2)
            self.h_target = self.h_ref
        if est.tracking and est.source == "vision" and est.range_m and est.range_m > 1.0:
            want = raw.height_m + clamp(est.vertical_m, 0.5)
            want = min(max(want, self.h_ref - c.height_follow_m), self.h_ref + c.height_follow_m)
            want = min(max(want, c.height_min_m), c.height_max_m)
            self.h_target += 0.1 * (want - self.h_target)
        err = self.h_target - raw.height_m
        return 0 if abs(err) < 0.04 else int(round(clamp(c.k_up * err, c.max_up_rc)))

    def _heading_rc(self, raw: RawObs, heading: float) -> int:
        if raw.yaw_deg is None:
            return 0
        return int(round(clamp(self.cfg.k_yaw * wrap_deg(heading - raw.yaw_deg) * self.cfg.yaw_sign, self.cfg.max_yaw_rc)))

    def _damp(self, raw: RawObs, forward: bool = True) -> tuple[int, int]:
        """(right, forward) sticks that cancel the measured ground speed (wind, coasting). Zero if the drone does not
        report a trusted velocity. This is what the Tello's own hover does; here it also keeps the return on line."""
        c = self.cfg
        if raw.vel_right_mps is None or raw.vel_fwd_mps is None:
            return 0, 0
        unit = c.mps_per_rc * max(self.scale, 0.2)
        self.drift_int = clamp(self.drift_int + raw.vel_right_mps * self._dt, 2.0)
        r = int(round(clamp(-(c.k_damp * raw.vel_right_mps + c.k_pos * self.drift_int) / unit, c.max_damp_rc)))
        f = int(round(clamp(-c.k_damp * raw.vel_fwd_mps / unit, c.max_damp_rc))) if forward else 0
        return r, f

    def _bearing_rc(self, est: RangeEstimate) -> int:
        return int(round(clamp(self.cfg.k_yaw * est.bearing_deg, self.cfg.max_yaw_rc)))

    # -- safety --------------------------------------------------------------------------------
    def _safety(self, raw: RawObs, est: RangeEstimate) -> str | None:
        c, t = self.cfg, raw.t
        if t - self.t0 > c.max_session_s:
            return "session_timeout"
        if raw.height_m is None:
            self.no_height_since = self.no_height_since if self.no_height_since is not None else t
            if t - self.no_height_since > c.max_no_telemetry_s:
                return "no_height_telemetry"
        else:
            self.no_height_since = None
            if raw.height_m > c.max_height_m:
                return "too_high"
        if raw.yaw_deg is None:
            self.no_yaw_since = self.no_yaw_since if self.no_yaw_since is not None else t
            if t - self.no_yaw_since > c.max_no_telemetry_s:
                return "no_yaw_telemetry"
        else:
            self.no_yaw_since = None
        if raw.battery_pct is not None and raw.battery_pct < c.min_battery_pct:
            return "low_battery"
        if self.state == "APPROACH" and self.r_start is not None:
            if self.d_unscaled * self.scale > self.r_start * (1 + c.overshoot_frac) + 1.0:
                return "overshoot_sign_not_where_expected"
            moving = self.v_est > 0.25
            closing = est.source == "vision" and self.cal.known and self.cal.scale > 0.35
            if moving and not closing and est.source == "vision":
                self.not_closing_since = self.not_closing_since if self.not_closing_since is not None else t
                if t - self.not_closing_since > c.not_closing_s:
                    return "range_not_closing"
            else:
                self.not_closing_since = None
        return None

    # -- one tick ------------------------------------------------------------------------------
    def step(self, raw: RawObs, dt: float | None = None) -> Action:
        c = self.cfg
        t = raw.t
        if self.t0 is None:
            self.t0 = t
        dt = dt if dt is not None else (0.05 if self.t_prev is None else max(1e-3, t - self.t_prev))
        self.t_prev = t
        self._dt = dt
        self._model_speed(dt)
        self.v_meas = raw.vel_fwd_mps if (raw.vel_fwd_mps is not None and abs(raw.vel_fwd_mps) < 6.0) else None
        est = self.est = self.est_filter.step(t, raw.detections, closing_mps=self.v_est)
        self.cal.update(t, est.source == "vision" and est.fully_visible, est.range_m, self.d_unscaled)
        if est.source == "vision" and est.range_m is not None:
            self.rate_hist.append((t, est.range_m))
        if self.state != "ABORT":
            reason = self._safety(raw, est)
            if reason:
                self.abort_reason = reason
                self._set("ABORT", t, reason)
        if self.state == "ABORT":
            self.last_fwd_rc = 0
            return self._action((0, 0, 0, 0), "abort")
        rc = getattr(self, f"_s_{self.state.lower()}")(raw, est, t, dt)
        self.last_fwd_rc = int(rc[1])
        mode = "done" if self.done else "go"
        return self._action(rc, mode)

    def _action(self, rc, mode) -> Action:
        e = self.est
        info = {"state": self.state, "detected": e.detected, "tracking": e.tracking, "range_m": _r(e.range_m),
                "std_m": _r(e.std_m), "source": e.source, "bearing_deg": _r(e.bearing_deg, 1), "naive_m": _r(e.naive_m),
                "spread": _r(e.spread, 3), "near_field": e.near_field, "scale": round(self.scale, 3),
                "calibrated": self.cal.known, "v_est_mps": round(self.v_est, 2), "flown_m": round(self.d_unscaled * self.scale, 2),
                "r_start_m": _r(self.r_start), "final_range_m": _r(self.final_range), "abort_reason": self.abort_reason,
                "note": e.note}
        return Action(rc=tuple(int(v) for v in rc), mode=mode, state=self.state, info=info)

    # -- states --------------------------------------------------------------------------------
    def _s_settle(self, raw, est, t, dt):
        up = self._height_rc(raw, est)
        if t - self.t_state >= 1.2 and (raw.height_m is None or abs((self.h_target or 0) - raw.height_m) < 0.2):
            self._set("ALIGN" if est.tracking else "SEARCH", t)
            self.search_prev, self.search_turned = raw.yaw_deg, 0.0
        return (0, 0, up, 0)

    def _s_search(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        if est.tracking:
            self._set("ALIGN", t, "sign found")
            return (0, 0, up, 0)
        if raw.yaw_deg is not None and self.search_prev is not None:
            self.search_turned += abs(wrap_deg(raw.yaw_deg - self.search_prev))
        self.search_prev = raw.yaw_deg
        if self.search_turned > c.search_max_deg:
            self.abort_reason = "sign_not_found"
            self._set("ABORT", t, "full turn, no sign")
            return (0, 0, 0, 0)
        return (0, 0, up, int(c.search_yaw_rc * c.yaw_sign))

    def _s_align(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        if not est.tracking:
            if t - self.t_state > 1.0 and est.age_s > 1.0:
                self._set("SEARCH", t, "lost while aligning")
                self.search_prev, self.search_turned = raw.yaw_deg, 0.0
            return (0, 0, up, 0)
        yaw = self._bearing_rc(est)
        if abs(est.bearing_deg) < c.align_deg and est.source == "vision":
            self.align_since = self.align_since if self.align_since is not None else t
            if t - self.align_since >= c.align_hold_s and (est.std_m or 9) < 0.6:
                self.r_start = est.range_m
                self.heading_ref = raw.yaw_deg
                self.lat_i = 0.0
                self._set("APPROACH", t, f"range {est.range_m:.2f} m")
                self.align_since = None
        else:
            self.align_since = None
        return (0, 0, up, yaw)

    def _desired_speed(self, est: RangeEstimate, room_latency: float) -> float:
        c = self.cfg
        r = est.range_m
        room = max(0.0, r - c.stop_m - self.v_est * room_latency)
        v_brake = math.sqrt(2 * c.brake_mps2 * room)
        v_cap = max(c.creep_mps, c.slow_gain * r)
        v = min(c.cruise_mps, v_cap, v_brake)
        if not self.cal.known:
            v = min(v, c.probe_mps)
        if est.source != "vision" and not est.near_field:
            v = min(v, c.min_mps)
        if room > 0.03:
            v = max(v, min(c.min_mps, c.cruise_mps))
        return v

    def _s_approach(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        if not est.tracking:
            self._set("SEARCH", t, "sign lost")
            self.search_prev, self.search_turned = raw.yaw_deg, 0.0
            return (0, 0, up, 0)
        r = est.range_m
        yaw, right = self._lateral_control(raw, est, dt)
        if r <= c.stop_m + c.stop_tol_m:
            self._enter_stop(raw, est, t)
            return (0, 0, up, 0)
        if est.near_field or r <= self.est_filter.near_start_m:
            self.hold_heading = raw.yaw_deg
            self.final_entry = (t, r, self.d_unscaled)
            self._set("FINAL", t, f"range {r:.2f} m")
            return (self._lat_ff(), self._rc_for(c.creep_mps), up, 0)
        if abs(est.bearing_deg) > c.turn_first_deg:
            return (0, 0, up, self._bearing_rc(est))      # sign far off the nose: turn first
        v = self._desired_speed(est, c.latency_s)
        return (right, self._rc_for(v), up, yaw)

    def _lat_ff(self) -> int:
        c = self.cfg
        return int(round(clamp(c.ki_lat * self.lat_i, c.max_lat_rc)))

    def _lateral_control(self, raw: RawObs, est: RangeEstimate, dt: float) -> tuple[int, int]:
        """Hold the heading, keep the sign centred with the sideways stick (PI: a steady cross-wind is absorbed by the
        integral). The yaw reference follows the sign only when it is more than ``bearing_trim_deg`` off the nose."""
        c = self.cfg
        if raw.yaw_deg is None or self.heading_ref is None or est.source != "vision":
            return 0, self._lat_ff()
        excess = est.bearing_deg - clamp(est.bearing_deg, c.bearing_trim_deg)
        self.heading_ref = wrap_deg(self.heading_ref + c.heading_follow * excess * c.yaw_sign)
        yaw = self._heading_rc(raw, self.heading_ref)
        right = 0
        if est.range_m is not None and est.range_m < c.lat_active_m:
            self.lat_i = clamp(self.lat_i + est.lateral_m * dt, c.max_lat_rc / max(c.ki_lat, 1e-6))
            right = int(round(clamp(c.k_lat * est.lateral_m + c.ki_lat * self.lat_i, c.max_lat_rc)))
        return yaw, right

    def _s_final(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        r = est.range_m
        if not est.tracking or r is None:
            self._enter_stop(raw, est, t, note="estimate lost")
            return (0, 0, up, 0)
        remaining = r - c.stop_m
        if remaining <= c.stop_tol_m + self.v_est * c.final_latency_s or r <= c.stop_hard_m + 0.05:
            self._enter_stop(raw, est, t)
            return (0, 0, up, 0)
        t0, r0, d0 = self.final_entry
        if (self.d_unscaled - d0) * self.scale > (r0 - c.stop_m) * 1.25 + 0.1:          # flown more than it should
            self._enter_stop(raw, est, t, note="distance budget used")
            return (0, 0, up, 0)
        v = min(c.creep_mps, math.sqrt(2 * c.brake_mps2 * max(0.0, remaining - self.v_est * c.final_latency_s)))
        v = max(v, min(c.min_mps, c.creep_mps)) if remaining > 0.05 else 0.0
        yaw = self._heading_rc(raw, self.hold_heading) if self.hold_heading is not None else 0
        return (self._lat_ff(), self._rc_for(v), up, yaw)

    def _enter_stop(self, raw, est, t, note: str = "") -> None:
        self.stop_since = None
        self.hold_int = 0.0
        self._stop_t0 = t
        self._set("STOP", t, note or f"range {est.range_m:.2f} m" if est.range_m else note)

    def _settled(self, raw: RawObs, t: float | None = None) -> bool:
        """Forward motion has stopped; total telemetry speed counts only for a while (a steady drift never gets to zero)."""
        c = self.cfg
        spd = raw.speed_mps if raw.speed_mps is not None else self.v_est
        waited = t is not None and t - self._stop_t0 >= c.stop_max_s
        return self.v_est < 0.06 and (spd < c.settle_speed_mps or waited)

    def _rearm(self, raw, est, t) -> bool:
        """Blown or drifted away from the stop point: approach again (a few times at most)."""
        c = self.cfg
        if self.rearms >= c.max_rearms or not est.tracking or est.range_m is None or est.range_m <= c.stop_m + c.stop_accept_m:
            return False
        self.rearms += 1
        self.hold_heading = raw.yaw_deg
        if est.range_m <= self.est_filter.near_start_m:
            self.final_entry = (t, est.range_m, self.d_unscaled)
            self._set("FINAL", t, f"drifted to {est.range_m:.2f} m, approaching again")
        else:
            self._set("APPROACH", t, f"drifted to {est.range_m:.2f} m, approaching again")
        return True

    def _s_stop(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        yaw = self._heading_rc(raw, self.hold_heading) if self.hold_heading is not None else 0
        # one short reverse pulse bleeds off the last forward speed (the Tello's own braking is slow)
        dr, df = self._damp(raw)
        brake = df if raw.vel_fwd_mps is not None else (-10 if (t - self._stop_t0) < 0.4 and self.v_est > 0.08 else 0)
        if est.tracking and est.range_m is not None:
            err = est.range_m - c.stop_m                  # range hold: wind pushes the drone, the estimate says where it is
            unit = c.mps_per_rc * max(self.scale, 0.2)
            self.hold_int = clamp(self.hold_int + err * dt, 1.0)
            if abs(err) > c.hold_band_m:
                brake += int(round(clamp((c.k_hold * (err - math.copysign(c.hold_band_m, err)) + c.k_hold_i * self.hold_int) / unit,
                                         c.max_hold_rc)))
            if est.range_m < c.stop_hard_m + 0.02:
                brake = min(brake, -25)                   # pushed too close: back off
        if t - self._stop_t0 >= c.stop_min_s and self._settled(raw, t):
            if self._rearm(raw, est, t):
                return (0, 0, up, yaw)
            self.stop_since = self.stop_since if self.stop_since is not None else t
            if t - self.stop_since >= c.stop_hold_s:
                self.final_range, self.final_std = est.range_m, est.std_m
                self._finish_stop(raw, est, t)
        else:
            self.stop_since = None
        return (dr or self._lat_ff(), brake, up, yaw)

    def _finish_stop(self, raw, est, t) -> None:
        self.done = True
        self._set("DONE", t, f"stopped at {self.final_range:.2f} m" if self.final_range is not None else "")

    def _s_done(self, raw, est, t, dt):
        return (0, 0, self._height_rc(raw, est), 0)


class OutAndBackPolicy(ApproachPolicy):
    """Policy 2: fly to the sign, stop 0.5 m in front of it, turn 180 degrees, fly back the same distance."""

    name = "out_and_back"
    title = "Fly to the sign, turn 180°, fly back the same distance"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.out_distance_m: float | None = None
        self.out_distance_dr_m: float | None = None
        self.turn_target: float | None = None
        self.turn_prev: float | None = None
        self.turned_deg = 0.0
        self.turn_ok_since: float | None = None
        self.ret_dist = 0.0
        self.ret_hold = 0.0
        self.return_error_m: float | None = None

    def _finish_stop(self, raw, est, t) -> None:     # reached the sign: do not finish, turn around
        vision = None if (self.r_start is None or self.final_range is None) else max(0.0, self.r_start - self.final_range)
        dr = self.d_unscaled * self.scale
        self.out_distance_dr_m = dr
        self.out_distance_m = vision if vision is not None else dr
        if vision is not None and dr > 0.5 and abs(vision - dr) > 0.35 * max(vision, 0.5):
            self.events.append(f"{t - self.t0:5.1f}s WARNING vision distance {vision:.2f} m vs dead reckoning {dr:.2f} m")
        self.turn_target = None
        self.turned_deg = 0.0
        self.drift_int = 0.0
        self.turn_prev = raw.yaw_deg
        self._set("TURN", t, f"flown out {self.out_distance_m:.2f} m")

    def _s_turn(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        if raw.yaw_deg is None:
            return (0, 0, up, 0)
        if self.turn_prev is not None:
            self.turned_deg += wrap_deg(raw.yaw_deg - self.turn_prev) * c.yaw_sign
        self.turn_prev = raw.yaw_deg
        remaining = c.turn_deg - self.turned_deg
        if t - self.t_state > 25.0:
            self.abort_reason = "turn_failed"
            self._set("ABORT", t, "turn took too long")
            return (0, 0, 0, 0)
        if abs(remaining) < c.turn_tol_deg and abs(self.est_yaw_rate(raw, dt)) < 8.0:
            self.turn_ok_since = self.turn_ok_since if self.turn_ok_since is not None else t
            if t - self.turn_ok_since >= 0.5:
                self.hold_heading = raw.yaw_deg
                self.ret_dist = 0.0
                if self.out_distance_m is None or self.out_distance_m < 0.25:
                    self.done = True
                    self._set("DONE", t, "nothing to fly back")
                else:
                    self._set("RETURN", t, f"{self.out_distance_m:.2f} m")
                return (0, 0, up, 0)
        else:
            self.turn_ok_since = None
        mag = min(c.turn_yaw_rc, max(c.turn_min_yaw_rc, abs(remaining) * 0.6))
        dr, df = self._damp(raw)
        return (dr, df, up, int(math.copysign(mag, remaining) * c.yaw_sign))

    def est_yaw_rate(self, raw, dt) -> float:
        """Turn rate in deg/s over the last 0.6 s (the telemetry yaw is whole degrees: a single step is far too coarse)."""
        hist = getattr(self, "_yaw_hist", None)
        if hist is None:
            hist = self._yaw_hist = deque(maxlen=80)
        if raw.yaw_deg is None:
            return 0.0
        last = hist[-1][1] if hist else raw.yaw_deg
        hist.append((raw.t, last + wrap_deg(raw.yaw_deg - last)))        # unwrapped
        while hist and raw.t - hist[0][0] > 0.6:
            hist.popleft()
        if len(hist) < 4:
            return 99.0
        return (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], 1e-3)

    def _s_return(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        measured = raw.vel_fwd_mps is not None and abs(raw.vel_fwd_mps) < 5.0
        self.ret_dist += (raw.vel_fwd_mps if measured else self.v_nominal * self.scale) * dt
        dr, df = self._damp(raw, forward=False)
        remaining = self.out_distance_m - self.ret_dist
        if remaining <= c.stop_tol_m + self.v_est * c.final_latency_s:
            self._stop_t0 = t
            self.stop_since = None
            self._set("STOP2", t, f"flown {self.ret_dist:.2f} of {self.out_distance_m:.2f} m")
            return (0, 0, up, 0)
        if t - self.t_state > 6.0 + 3.0 * self.out_distance_m / max(c.min_mps, 0.05):
            self.abort_reason = "return_timeout"
            self._set("ABORT", t, "return took too long")
            return (0, 0, 0, 0)
        v = min(c.return_cruise_mps, max(c.creep_mps, c.slow_gain * remaining),
                math.sqrt(2 * c.brake_mps2 * max(0.0, remaining - self.v_est * c.latency_s)))
        v = max(v, min(c.min_mps, c.return_cruise_mps)) if remaining > 0.1 else v
        return (dr, self._rc_for(v), up, self._heading_rc(raw, self.hold_heading))

    def _s_stop2(self, raw, est, t, dt):
        c = self.cfg
        up = self._height_rc(raw, est)
        dr, df = self._damp(raw)
        brake = df if raw.vel_fwd_mps is not None else (-10 if (t - self._stop_t0) < 0.4 and self.v_est > 0.08 else 0)
        if t - self._stop_t0 >= c.stop_min_s and self._settled(raw, t):
            self.stop_since = self.stop_since if self.stop_since is not None else t
            if t - self.stop_since >= c.return_hold_s:
                self.return_error_m = self.out_distance_m - self.ret_dist
                self.done = True
                self._set("DONE", t, f"back at the start (flown {self.ret_dist:.2f} m)")
        else:
            self.stop_since = None
        return (dr, brake, up, self._heading_rc(raw, self.hold_heading))

    def _action(self, rc, mode) -> Action:
        a = super()._action(rc, mode)
        a.info.update(out_distance_m=_r(self.out_distance_m), out_distance_dr_m=_r(self.out_distance_dr_m),
                      returned_m=_r(self.ret_dist), turned_deg=round(self.turned_deg, 1))
        return a


def _r(v, n=2):
    return None if v is None else round(float(v), n)


POLICIES = {ApproachPolicy.name: ApproachPolicy, OutAndBackPolicy.name: OutAndBackPolicy}
