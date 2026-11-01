"""Long-course tag homing: start at A -> stop 5 m short of tag B -> turn -> stop 5 m short of tag A.

For tags far apart (73 m): no map needed. Each tick it gets the detections of the tags in
view (range + bearing straight from the existing tello_calibration TagDetector, i.e. the tag
pose in the camera) and two Tello telemetry values (height h, yaw). It returns rc sticks.

Legs (both identical):
  BLIND   no validated sighting of the target tag yet: hold the leg's heading (telemetry
          yaw) and fly straight at cruise speed. Tags are too small to see from far away.
  TRACK   target tag seen: yaw so the tag sits in the image centre, fly at cruise speed,
          brake so the drone stops ``stop_range_m`` (5 m) short of the tag.
  HOLD    at 5 m: hover until the range has been steady for ``hold_s``.
Between the legs:
  TURN    rotate (slower) by about 180 deg using telemetry yaw, then
  SEARCH  slow sweep around the expected direction until tag A shows up (or, if it is too
          far to see, carry on BLIND on the reverse heading).
Height is held the whole time from telemetry ``h``; the right/left stick is never used --
only the heading changes.

Validation loop (``RangeValidator``), on every sighting of the target tag:
  * only the expected tag id counts (other ids are logged as unexpected);
  * one-frame range jumps > ``max_jump_m`` are rejected as misdetections;
  * the range trend over the last ``window_s`` must be shrinking at least ``min_closing``
    of the speed we are flying; if not for ``not_closing_s``, slow down and re-centre, and
    after ``abort_s`` hover and ask to land;
  * dead-reckoning cross-check: distance flown this leg + current range should equal the
    first sighting's range + distance flown until then (a fake/misplaced tag breaks this).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


@dataclass
class HomingCfg:
    tag_a: int = 1
    tag_b: int = 2
    course_m: float = 73.0         # tag A to tag B; only a prior: replaced by a measured estimate
                                   # as soon as the first tag is seen
    adaptive_start: bool = True    # `policy start new` works with what is in front of it: the first
                                   # race tag it sees (1 or 2, in view or found by spinning) becomes
                                   # the first target, the other one the return target
    start_offset_m: float = 1.0    # how far in front of tag A the drone takes off
    blind_margin_m: float = 1.0    # stop flying blind this much before the 5 m line
    # no tag in view
    blind_mode: str = "auto"       # no tag detected -> blind mode switches on by itself:
                                   #   "auto": spin in place if the tag should be in detection range,
                                   #           otherwise fly the leg heading until it is, then spin
                                   #   "spin": always spin first; "fly": never spin
    spin_yaw_rc: int = 20          # slow turn so frames stay sharp enough to detect a far tag
    hop_m: float = 10.0            # a full circle found nothing: fly at least this far on the leg heading, spin again
    tag_size_m: float = 0.4877     # black square of the race tags. 2 ft sheet with the white border
                                   # -> ~488 mm square; 2 ft square itself -> 0.6096. MEASURE the print.
    detect_range_m: float | None = None  # how far the tag can be detected; None = 50 x tag_size_m
                                   # (TagDetector on rendered frames: 488 mm ~25 m, 610 mm ~35 m)
    align_deg: float = 20.0        # turn onto the heading before flying blind if further off than this
    # speed and stopping
    cruise_mps: float = 1.5        # same speed toward either tag
    search_speed_frac: float = 0.0 # forward speed while turning/searching, as a fraction of cruise
    stop_range_m: float = 5.0      # never closer than this to a tag (hard limit)
    stop_buffer_m: float = 0.4     # aim this much further out: braking lag carries it ~0.2 m on
    brake_mps2: float = 0.8        # braking profile into the 5 m stop
    stop_tol_m: float = 0.25
    hold_s: float = 1.0
    mps_per_rc: float = 0.008      # rc 100 ~ 0.8 m/s (tools/tello_policy.py estimate). MEASURE.
    # height
    height_m: float = 1.2          # telemetry h to hold all flight (put the tags at this height)
    k_up: float = 80.0
    max_up_rc: int = 40
    # heading
    k_yaw: float = 1.2             # rc per degree of bearing / heading error
    max_yaw_rc: int = 50
    turn_yaw_rc: int = 35          # slower yaw for the 180 turn
    search_yaw_rc: int = 20        # slow sweep while looking for the tag
    search_span_deg: float = 40.0  # sweep +-this around the expected heading
    yaw_sign: float = 1.0          # +1 if telemetry yaw grows on a right turn. VERIFY on the drone.
    # validation
    window_s: float = 1.5          # trend window close in; grows with range (far ranges are noisy)
    window_per_m: float = 0.08     # +0.08 s of window per metre of range (60 m -> 4.8 s)
    max_window_s: float = 6.0
    min_samples: int = 6
    min_closing: float = 0.35      # "closing" = range shrinks at >= 35% of our forward speed
    flat_mps: float = 0.05         # shrinking slower than this (or growing) = NOT closing
    not_closing_s: float = 2.0     # tracking but not closing this long -> flag, re-centre on the tag
    abort_s: float = 8.0           # tracking but not closing this long -> hover and request landing
    max_jump_m: float = 3.0        # one-frame range jump bigger than this = misdetection
    dr_tolerance: float = 0.3      # dead-reckoning mismatch allowed, as a fraction of the leg length
    lost_after_s: float = 1.0      # no sighting this long while tracking -> back to BLIND on last bearing
    # safety (any of these -> ABORT: sticks to zero, mode "land", reason in info["abort_reason"])
    max_session_s: float = 300.0   # whole mission time cap
    max_height_m: float = 2.5      # telemetry h above this -> abort
    max_no_height_s: float = 2.0   # no height telemetry this long -> abort
    max_no_yaw_s: float = 2.0      # no yaw telemetry this long -> abort (can't hold a heading or count a spin)
    speed_uncertainty: float = 0.3 # the real speed may be up to 30% above mps_per_rc: blind distance shrinks to match
    height_stuck_s: float = 4.0    # climbing hard this long without the height changing -> abort
    geofence_extra_m: float = 10.0 # dead-reckoned distance from takeoff > course + this -> abort
    keep_out_any_tag: bool = True  # never fly forward toward ANY detected tag closer than stop_range_m


@dataclass
class Sighting:
    tag_id: int
    range_m: float                 # distance camera -> tag centre (TagObservation.distance_m)
    bearing_deg: float             # + = tag right of image centre: atan2(tvec_x, tvec_z)
    tvec: tuple | None = None      # tag centre in the camera frame (x right, y down, z forward), m
    normal: tuple | None = None    # tag's printed-face direction in the camera frame (R[:, 2] of rvec)


@dataclass
class HomingObs:
    t: float
    sightings: list[Sighting] = field(default_factory=list)
    height_m: float | None = None  # telemetry h
    yaw_deg: float | None = None   # telemetry yaw


class RangeValidator:
    """Checks that the distance to the target tag really is shrinking."""

    def __init__(self, cfg: HomingCfg) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque()
        self.filtered = None
        self.rejected = 0
        self.not_closing_since = None
        self.first = None              # (range, flown) at the first accepted sighting
        self.status = "no_data"
        self.slope = None

    def add(self, t: float, r: float, flown_m: float) -> bool:
        """Feed one sighting. Returns False if it was rejected as a jump."""
        if self.filtered is not None and abs(r - self.filtered) > self.cfg.max_jump_m:
            self.rejected += 1
            if self.rejected < 5:      # 5 consistent "jumps" in a row = the range really moved
                return False
        self.rejected = 0
        self.hist.append((t, r))
        while self.hist and t - self.hist[0][0] > self.window(r):
            self.hist.popleft()
        recent = sorted(x[1] for x in list(self.hist)[-5:])
        self.filtered = recent[len(recent) // 2]                   # median of the last 5
        if self.first is None:
            self.first = (r, flown_m)
        return True

    def pause(self) -> None:
        """Stopped tracking (tag out of view): the not-closing clock only runs while tracking."""
        self.not_closing_since = None
        self.status = "blind"

    def window(self, r: float) -> float:
        c = self.cfg
        return min(c.max_window_s, max(c.window_s, c.window_per_m * r))

    def update(self, t: float, forward_mps: float, flown_m: float) -> str:
        c = self.cfg
        span = self.window(self.filtered or 0.0)
        if len(self.hist) >= c.min_samples and self.hist[-1][0] - self.hist[0][0] >= 0.5 * span:
            ts = [x[0] for x in self.hist]
            rs = [x[1] for x in self.hist]
            mt, mr = sum(ts) / len(ts), sum(rs) / len(rs)
            den = sum((a - mt) ** 2 for a in ts) or 1e-9
            self.slope = sum((a - mt) * (b - mr) for a, b in zip(ts, rs)) / den      # m/s, negative = closing
            if forward_mps < 0.15:                                                   # not trying to close
                self.status, self.not_closing_since = "holding", None
            elif self.slope <= -c.min_closing * forward_mps:
                self.status, self.not_closing_since = "closing", None
            elif self.slope <= -c.flat_mps:                   # closing, just slowly (headwind, gust)
                self.status, self.not_closing_since = "slow_closing", None
            else:                                             # flat or growing: something is wrong
                self.not_closing_since = self.not_closing_since or t
                waited = t - self.not_closing_since
                self.status = "abort" if waited >= c.abort_s else "not_closing" if waited >= c.not_closing_s else "closing?"
        # dead-reckoning cross-check
        if self.first is not None and self.filtered is not None:
            r0, f0 = self.first
            expected = r0 - (flown_m - f0)
            if abs(expected - self.filtered) > c.dr_tolerance * max(r0, 5.0):
                self.status = "dr_mismatch" if self.status in ("closing", "closing?", "no_data") else self.status
        return self.status


class HomingPolicy:
    def __init__(self, cfg: HomingCfg | None = None) -> None:
        self.cfg = cfg or HomingCfg()
        self.leg = "to_B"              # to_B -> turn -> to_A -> done
        self.state = "CLIMB"
        self.val = RangeValidator(self.cfg)
        self.leg_heading = None        # telemetry yaw to hold while BLIND
        self.last_seen_t = -math.inf
        self.last_bearing = 0.0
        self.flown_m = 0.0             # dead-reckoned distance this leg (from commanded speed)
        self.dr = [0.0, 0.0]           # dead-reckoned position from the takeoff spot (yaw + commanded speed)
        self.home = None               # where tag A is, in that same frame
        self.hold_since = None
        self.turn_from = None
        self.search_t0 = None
        self.unexpected: set[int] = set()
        self.out_tag, self.back_tag = self.cfg.tag_b, self.cfg.tag_a
        self.acquired = False          # first target chosen from what was actually seen
        self.course_est = None         # course length measured from the first sighting
        self.leg_len = self.cfg.course_m - self.cfg.start_offset_m
        self.search_reason = None
        self.hop_left = None           # blind distance left before the next spin (None = no limit)
        self.a_fixes: list = []
        self.t0 = None
        self.abort_reason = None
        self.no_height_since = None
        self.no_yaw_since = None
        self.climb_check = None        # (t, h) when a hard climb started
        self.last_up = 0
        self.keep_out = False        # (bearing_deg, dist_m) to tag A, measured from tag B's pose at the stop
        self.spin_prev = None
        self.spin_turned = 0.0
        self.spin_dir = 1
        self.spins = 0

    @property
    def target(self) -> int:
        return self.out_tag if self.leg == "to_B" else self.back_tag

    @property
    def course(self) -> float:
        """Measured course length once a tag has been seen, else the configured prior."""
        return self.course_est if self.course_est is not None else self.cfg.course_m

    @property
    def acquiring(self) -> bool:
        return self.cfg.adaptive_start and not self.acquired and self.leg == "to_B"

    def _up(self, obs: HomingObs) -> int:
        if obs.height_m is None:
            return 0
        return int(round(clamp(self.cfg.k_up * (self.cfg.height_m - obs.height_m), self.cfg.max_up_rc)))

    def _heading_rc(self, obs: HomingObs, heading: float) -> int:
        if obs.yaw_deg is None:
            return 0
        err = wrap_deg(heading - obs.yaw_deg) * self.cfg.yaw_sign
        return int(round(clamp(self.cfg.k_yaw * err, self.cfg.max_yaw_rc)))

    def _spin(self, obs: HomingObs) -> None:
        self.state, self.spin_prev, self.spin_turned = "SPIN", obs.yaw_deg, 0.0
        self.spin_dir = 1 if self.last_bearing >= 0 else -1
        self.spins += 1

    @property
    def detect_range(self) -> float:
        c = self.cfg
        return c.detect_range_m if c.detect_range_m is not None else 50.0 * c.tag_size_m

    def _blind(self, hop: bool) -> None:
        self.state = "BLIND"
        if hop and self.cfg.blind_mode != "fly":
            remaining = self.leg_len - self.flown_m        # dead-reckoned distance to the tag
            self.hop_left = max(self.cfg.hop_m, remaining - 0.8 * self.detect_range)
        else:
            self.hop_left = None

    def _no_tag(self, obs: HomingObs, lost: bool = False) -> None:
        """No tag detected: switch blind mode on. Spin where the tag could be seen from here,
        otherwise fly the leg heading until it should be in range (then spin)."""
        c = self.cfg
        if c.blind_mode == "fly":
            self._blind(hop=False)
            return
        remaining = self.leg_len - self.flown_m
        if c.blind_mode == "spin" or (not lost and remaining <= self.detect_range):
            self._spin(obs)
        else:
            self._blind(hop=True)

    def _start_leg(self, leg: str) -> None:
        self.leg, self.state = leg, "BLIND"
        self.val.reset()
        self.flown_m = 0.0
        self.hold_since = None
        self.leg_len = self.course - self.cfg.start_offset_m if leg == "to_B" else \
            getattr(self, "return_len", self.course - self.cfg.stop_range_m)

    def _safety(self, obs: HomingObs) -> str | None:
        """Hard limits, checked every tick before anything else."""
        c = self.cfg
        if obs.t - self.t0 > c.max_session_s:
            return "session_timeout"
        h = obs.height_m
        if h is None or not math.isfinite(h):
            self.no_height_since = self.no_height_since if self.no_height_since is not None else obs.t
            if obs.t - self.no_height_since > c.max_no_height_s:
                return "no_height_telemetry"
        else:
            self.no_height_since = None
            if h > c.max_height_m:
                return "too_high"
            if self.last_up > 15:                     # we asked for a real climb last tick
                if self.climb_check is None:
                    self.climb_check = (obs.t, h)
                elif obs.t - self.climb_check[0] > c.height_stuck_s:
                    if h - self.climb_check[1] < 0.1:
                        return "height_not_responding"
                    self.climb_check = (obs.t, h)
            else:
                self.climb_check = None
        if obs.yaw_deg is None or not math.isfinite(obs.yaw_deg):
            self.no_yaw_since = self.no_yaw_since if self.no_yaw_since is not None else obs.t
            if obs.t - self.no_yaw_since > c.max_no_yaw_s:
                return "no_yaw_telemetry"
        else:
            self.no_yaw_since = None
        # dead reckoning only matters while the target tag is not giving a measured range
        if self.state in ("BLIND", "SPIN", "SEARCH") and math.hypot(*self.dr) > self.course + c.geofence_extra_m:
            return "geofence"
        return None

    def step(self, obs: HomingObs, dt: float = 0.05):
        """Return ((right, forward, up, yaw), info)."""
        c = self.cfg
        if self.t0 is None:
            self.t0 = obs.t
        if self.state not in ("ABORT", "DONE"):
            reason = self._safety(obs)
            if reason:
                self.state, self.abort_reason = "ABORT", reason
        if self.state == "ABORT":
            self.last_up = 0
            return (0, 0, 0, 0), self._info("land", "abort", 0.0)
        # drop garbage detections (NaN, negative or zero range)
        sightings = [s for s in obs.sightings if math.isfinite(s.range_m) and s.range_m > 0
                     and math.isfinite(s.bearing_deg)]
        up = self._up(obs)
        if self.acquiring:                       # whichever race tag is in front of us -- but not one we
            # are already within the stop range of: that is the tag we took off next to
            tgt = [s for s in sightings if s.tag_id in (c.tag_a, c.tag_b)
                   and s.range_m > c.stop_range_m + c.stop_buffer_m]
        else:
            tgt = [s for s in sightings if s.tag_id == self.target]
        for s in sightings:
            if s.tag_id not in (c.tag_a, c.tag_b):
                self.unexpected.add(s.tag_id)
        seen = None
        if tgt:
            cand = min(tgt, key=lambda s: abs(s.bearing_deg))
            if self.acquiring and cand.tag_id != self.out_tag:    # it's facing the other tag: go there first
                self.out_tag, self.back_tag = cand.tag_id, self.out_tag
            if self.val.add(obs.t, cand.range_m, self.flown_m):
                seen = cand
                self.last_seen_t, self.last_bearing = obs.t, cand.bearing_deg
                if self.leg == "to_B" and self.course_est is None:
                    # measured course: what we flew blind + what we see + where we took off
                    self.course_est = cand.range_m + self.flown_m + c.start_offset_m
                    self.leg_len = self.course_est - c.start_offset_m
                self.acquired = True

        fwd_mps, yaw = 0.0, 0
        if self.state == "CLIMB":
            if obs.yaw_deg is not None and self.leg_heading is None:
                self.leg_heading = obs.yaw_deg          # assumes it starts at A, nose toward B
                th = math.radians(obs.yaw_deg * c.yaw_sign)
                self.home = (-c.start_offset_m * math.cos(th), -c.start_offset_m * math.sin(th))
            if obs.height_m is not None and abs(c.height_m - obs.height_m) < 0.15:
                if seen is not None:
                    self.state = "TRACK"
                elif self.acquiring:
                    self._spin(obs)                      # nothing in front: look around for a tag first
                else:
                    self._no_tag(obs)
        elif self.state in ("BLIND", "TRACK"):
            if seen is not None:
                self.state = "TRACK"
            elif self.state == "TRACK" and obs.t - self.last_seen_t > c.lost_after_s:
                self._no_tag(obs, lost=True)             # lost it: keep the heading we had, then spin
                self.val.pause()
                if obs.yaw_deg is not None:
                    self.leg_heading = wrap_deg(obs.yaw_deg + self.last_bearing * c.yaw_sign)
            if self.state == "TRACK":
                rng = self.val.filtered
                stop_at = c.stop_range_m + c.stop_buffer_m
                room = max(0.0, rng - stop_at)
                fwd_mps = min(c.cruise_mps, math.sqrt(2 * c.brake_mps2 * room))
                yaw = int(round(clamp(c.k_yaw * self.last_bearing, c.max_yaw_rc)))
                if seen is None:                         # between frames: keep turning gently
                    yaw = yaw // 2
                if rng <= stop_at + c.stop_tol_m:
                    self.state, fwd_mps = "HOLD", 0.0
            else:                                        # BLIND
                fwd_mps = c.cruise_mps
                yaw = self._heading_rc(obs, self.leg_heading) if self.leg_heading is not None else 0
                if self.leg_heading is not None and obs.yaw_deg is not None and \
                        abs(wrap_deg(self.leg_heading - obs.yaw_deg)) > c.align_deg:
                    fwd_mps = 0.0                        # face the leg heading before flying off
                if self.hop_left is not None:
                    self.hop_left -= fwd_mps * dt
                    if self.hop_left <= 0:
                        self._spin(obs)
                        fwd_mps, yaw = 0.0, 0
                blind_max = (self.leg_len - c.stop_range_m - c.blind_margin_m) / (1.0 + c.speed_uncertainty)
                if self.state == "BLIND" and self.flown_m >= blind_max:
                    # should be ~5 m from the tag by now and still can't see it: stop, look around
                    self.state, self.search_t0, self.search_reason = "SEARCH", obs.t, "limit"
                    fwd_mps, yaw = 0.0, 0
        elif self.state == "HOLD":
            if seen is not None and self.leg == "to_B" and seen.tvec is not None and seen.normal is not None:
                # tag B faces A, so A sits course_m straight out of tag B's face
                ax = seen.tvec[0] + self.course * seen.normal[0]
                az = seen.tvec[2] + self.course * seen.normal[2]
                self.a_fixes.append((math.degrees(math.atan2(ax, az)), math.hypot(ax, az), obs.yaw_deg))
            if seen is not None:
                yaw = int(round(clamp(c.k_yaw * self.last_bearing, c.max_yaw_rc)))
                err = self.val.filtered - (c.stop_range_m + c.stop_buffer_m)
                fwd_mps = clamp(0.5 * err, 0.3)          # small correction toward exactly 5 m
            self.hold_since = self.hold_since or obs.t
            if obs.t - self.hold_since >= c.hold_s:
                if self.leg == "to_B":
                    self.leg, self.state = "turn", "TURN"
                    self.turn_from = obs.yaw_deg
                    # heading back = straight at where A is, from the dead-reckoned track since takeoff
                    # (reversing the final approach to B is wrong if B was approached at an angle)
                    if len(self.a_fixes) >= 5 and obs.yaw_deg is not None:
                        # heading to A from tag B's pose (median over the hold: robust to pose flips)
                        heads = sorted(wrap_deg(y + b * c.yaw_sign) for b, _, y in self.a_fixes if y is not None)
                        h = heads[len(heads) // 2]
                        self.leg_heading = h
                        self.return_len = sorted(d for _, d, _ in self.a_fixes)[len(self.a_fixes) // 2]
                        self.return_from = "tag_B_pose"
                        if self.home is not None:          # re-anchor dead reckoning on the measured pose
                            th = math.radians(h * c.yaw_sign)
                            self.dr = [self.home[0] - self.return_len * math.cos(th),
                                       self.home[1] - self.return_len * math.sin(th)]
                    elif self.home is not None:
                        self.return_from = "dead_reckoning"
                        dx, dy = self.home[0] - self.dr[0], self.home[1] - self.dr[1]
                        self.leg_heading = wrap_deg(math.degrees(math.atan2(dy, dx)) * c.yaw_sign)
                        self.return_len = math.hypot(dx, dy)
                    elif obs.yaw_deg is not None:
                        self.leg_heading = wrap_deg(obs.yaw_deg + 180.0)
                else:
                    self.leg, self.state = "done", "DONE"
        elif self.state == "TURN":
            fwd_mps = c.search_speed_frac * c.cruise_mps
            err = wrap_deg(self.leg_heading - obs.yaw_deg) if obs.yaw_deg is not None else 0.0
            yaw = int(c.turn_yaw_rc if err * c.yaw_sign >= 0 else -c.turn_yaw_rc)
            if abs(err) < 10.0:
                self._start_leg("to_A")
                if seen is not None:
                    self.state = "TRACK"
                elif c.blind_mode == "fly":
                    self.state, self.search_t0, self.search_reason = "SEARCH", obs.t, "turn"
                else:
                    self._no_tag(obs)
        elif self.state == "SPIN":
            if seen is not None:
                self.state = "TRACK"
            else:                                        # hover in place and turn
                if obs.yaw_deg is not None and self.spin_prev is not None:
                    self.spin_turned += abs(wrap_deg(obs.yaw_deg - self.spin_prev))
                self.spin_prev = obs.yaw_deg
                yaw = self.spin_dir * c.spin_yaw_rc
                if self.spin_turned >= 360.0:            # full circle, nothing: hop forward and try again
                    self._blind(hop=True)
                    yaw = 0
        elif self.state == "SEARCH":
            if seen is not None:
                self.state = "TRACK"
            else:
                at_limit = self.search_reason == "limit"
                fwd_mps = 0.0 if at_limit else c.search_speed_frac * c.cruise_mps
                span = 90.0 if at_limit else c.search_span_deg  # at the limit: look wider
                # sweep +-span around the expected heading, slowly
                period = 4 * span / max(1e-6, c.search_yaw_rc * 0.6)
                phase = ((obs.t - self.search_t0) % period) / period
                offset = span * (1 - 4 * abs(phase - 0.5))
                yaw = self._heading_rc(obs, wrap_deg(self.leg_heading + offset))
                yaw = int(clamp(yaw, c.search_yaw_rc))
                if obs.t - self.search_t0 > period:
                    if at_limit:                         # a full wide sweep found nothing: give up safely
                        self.state, self.abort_reason = "ABORT", "tag_not_found"
                    else:                                # tag too far to see yet: fly blind on heading
                        self.state = "BLIND"
        elif self.state == "DONE":
            fwd_mps, yaw = 0.0, 0

        # keep-out: never move forward toward any tag that is already inside the 5 m limit
        self.keep_out = False
        if c.keep_out_any_tag and fwd_mps > 0:
            if any(s.range_m < c.stop_range_m and abs(s.bearing_deg) < 45.0 for s in sightings):
                fwd_mps, self.keep_out = 0.0, True

        # validation loop
        status = self.val.update(obs.t, fwd_mps, self.flown_m) if self.state in ("TRACK", "HOLD") else "blind"
        if self.state == "TRACK" and status in ("not_closing", "dr_mismatch"):
            # don't slow (in wind that lets the wind win); turn harder onto the tag instead
            yaw = int(round(clamp(2 * c.k_yaw * self.last_bearing, c.max_yaw_rc)))
        if self.state == "TRACK" and status == "abort":
            self.abort_reason = "range_not_shrinking"
            self.state = "ABORT"
        if self.state == "ABORT":
            self.abort_reason = self.abort_reason or "search_failed"
            self.last_up = 0
            return (0, 0, 0, 0), self._info("land", status, 0.0)

        self.flown_m += fwd_mps * dt
        if obs.yaw_deg is not None:
            th = math.radians(obs.yaw_deg * c.yaw_sign)
            self.dr[0] += fwd_mps * dt * math.cos(th)
            self.dr[1] += fwd_mps * dt * math.sin(th)
        fwd = int(round(clamp(fwd_mps / c.mps_per_rc, 100)))
        self.last_up = up
        return (0, fwd, up, int(yaw)), self._info("done" if self.state == "DONE" else "go", status, fwd_mps)

    def _info(self, mode: str, status: str, fwd_mps: float) -> dict:
        v = self.val
        return {"mode": mode, "leg": self.leg, "state": self.state, "target": self.target,
                "range_m": None if v.filtered is None else round(v.filtered, 2),
                "closing_mps": None if v.slope is None else round(-v.slope, 2),
                "validation": status, "fwd_cmd_mps": round(fwd_mps, 2), "flown_m": round(self.flown_m, 1),
                "unexpected_tags": sorted(self.unexpected), "spins": self.spins,
                "return_from": getattr(self, "return_from", None), "abort_reason": self.abort_reason,
                "first_target": self.out_tag, "course_est_m": None if self.course_est is None else round(self.course_est, 1),
                "keep_out": self.keep_out}
