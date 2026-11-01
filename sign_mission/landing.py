"""Landing policy: stop, settle, descend gently, then the SDK land command.

BRAKE    zero sticks, wait until the drone has stopped (telemetry speed) or the time is up
SETTLE   hold the height a moment so the Tello's own stabiliser has the drone steady
DESCEND  slow, straight down (the horizontal sticks stay at zero) to ``flare_m``; height telemetry is checked
         against the descent, and a descent that does not happen hands over to the land command at once
LAND     "land" is requested; the runner sends it (blocking SDK call, retried) and reports back

Never uses the motor-cut ("emergency") command: that is for a person, with a hand on the drone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import Action, RawObs


@dataclass
class LandingCfg:
    settle_speed_mps: float = 0.12
    brake_max_s: float = 3.0
    settle_s: float = 0.6
    descend_rc: int = 22               # down stick during the descent
    flare_m: float = 0.35              # hand over to the land command below this height
    descend_max_s: float = 14.0
    stuck_s: float = 3.0               # height not falling by 0.1 m in this time: land now
    k_up: float = 80.0
    k_damp_rc: float = 140.0           # stick per m/s of measured drift (when the drone reports a trusted velocity)
    max_damp_rc: int = 25


class LandingPolicy:
    name = "landing"

    def __init__(self, cfg: LandingCfg | None = None):
        self.cfg = cfg or LandingCfg()
        self.state = "BRAKE"
        self.t0: float | None = None
        self.t_state = 0.0
        self.h_hold: float | None = None
        self.h_ref: tuple[float, float] | None = None
        self.reason = ""
        self.command_sent = False

    def _set(self, state: str, t: float) -> None:
        self.state, self.t_state = state, t

    def step(self, raw: RawObs, dt: float = 0.05) -> Action:
        c, t = self.cfg, raw.t
        if self.t0 is None:
            self.t0 = self.t_state = t
        h = raw.height_m
        if self.h_hold is None and h is not None:
            self.h_hold = h
        hold_up = 0 if (h is None or self.h_hold is None or abs(self.h_hold - h) < 0.05) else \
            int(max(-25, min(25, c.k_up * (self.h_hold - h))))
        dr = df = 0
        if raw.vel_right_mps is not None and raw.vel_fwd_mps is not None:
            dr = int(max(-c.max_damp_rc, min(c.max_damp_rc, -c.k_damp_rc * raw.vel_right_mps)))
            df = int(max(-c.max_damp_rc, min(c.max_damp_rc, -c.k_damp_rc * raw.vel_fwd_mps)))
        if self.state == "BRAKE":
            spd = raw.speed_mps
            if (spd is not None and spd < c.settle_speed_mps and t - self.t_state > 0.4) or t - self.t_state > c.brake_max_s:
                self._set("SETTLE", t)
            return self._act((dr, df, hold_up, 0))
        if self.state == "SETTLE":
            if t - self.t_state >= c.settle_s:
                self._set("DESCEND", t)
                self.h_ref = (t, h if h is not None else 0.0)
            return self._act((dr, df, hold_up, 0))
        if self.state == "DESCEND":
            if h is None:
                self.reason = "no height telemetry"
                self._set("LAND", t)
            elif h <= c.flare_m:
                self.reason = f"flare height {h:.2f} m"
                self._set("LAND", t)
            elif t - self.t_state > c.descend_max_s:
                self.reason = "descent timeout"
                self._set("LAND", t)
            else:
                t_ref, h_ref = self.h_ref
                if t - t_ref >= c.stuck_s:
                    if h_ref - h < 0.1:
                        self.reason = "height not falling"
                        self._set("LAND", t)
                    self.h_ref = (t, h)
            if self.state == "DESCEND":
                return self._act((dr, df, -c.descend_rc, 0))
        if self.state == "LAND":
            if not self.command_sent:
                self.command_sent = True
                return Action(rc=(0, 0, 0, 0), mode="go", state="LAND", command="land", info={"reason": self.reason})
            return self._act((0, 0, 0, 0))
        return self._act((0, 0, 0, 0))

    def _act(self, rc) -> Action:
        return Action(rc=rc, mode="go", state=self.state, info={})
