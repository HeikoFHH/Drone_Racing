"""MissionRunner: take off, run a policy, stop and land. The only thing that writes rc sticks.

    runner = MissionRunner(drone)
    runner.start("approach")        # Start button: take-off, then the policy
    runner.stop()                   # Stop button: stop, then land
    runner.status()                 # JSON for the interface

Phases: IDLE -> PREFLIGHT -> TAKEOFF -> FLYING -> LANDING -> LANDED (or ABORTED / ERROR, then landed or not)
Stop works in every flying phase and always ends in the landing policy (stop, settle, descend, land).
Safety nets here, independent of the policies: global stick cap, pre-flight checks (battery, link, video, not
already airborne), watchdog on the state packets, maximum mission time, bounded landing retries.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from .common import Action, RangeEstimate, RawObs
from .drone import Drone, Telemetry
from .landing import LandingCfg, LandingPolicy
from .policies import POLICIES, ApproachPolicy, PolicyCfg
from .range_est import SignRangeEstimator
from .spec import SignSpec

TICK_S = 0.05


@dataclass
class RunnerCfg:
    rc_cap: int = 60                   # no stick ever exceeds this
    min_battery_pct: float = 20.0      # refuse to take off below this
    require_video: bool = True
    takeoff_timeout_s: float = 15.0
    max_mission_s: float = 240.0       # takeoff to landing
    link_warn_s: float = 1.5
    link_land_s: float = 6.0
    land_retries: int = 3
    landing_total_s: float = 60.0
    shadow: bool = False               # run the policy on live video and telemetry but never take off or send a stick (carry the drone by hand)
    shadow_max_s: float = 120.0


class MissionRunner:
    def __init__(self, drone: Drone, cfg: RunnerCfg | None = None, sign: SignSpec | None = None,
                 policy_cfg: PolicyCfg | None = None, landing_cfg: LandingCfg | None = None):
        self.drone = drone
        self.cfg = cfg or RunnerCfg()
        self.sign = sign or SignSpec()
        self.policy_cfg = policy_cfg or PolicyCfg()
        self.landing_cfg = landing_cfg or LandingCfg()
        self.lock = threading.RLock()
        self.stop_evt = threading.Event()
        self.quit_evt = threading.Event()
        self.thread: threading.Thread | None = None
        self.mon_thread: threading.Thread | None = None
        self.mon_pause = threading.Event()
        self.monitor_est = SignRangeEstimator(self.sign)
        self.phase = "IDLE"
        self.policy_name: str | None = None
        self.policy = None
        self.events: deque = deque(maxlen=80)
        self.est = RangeEstimate()
        self.dets: list = []
        self.info: dict = {}
        self.tel = Telemetry()
        self.result: dict = {}
        self.mission_t0: float | None = None
        self.last_rc = (0, 0, 0, 0)
        self.reason = ""
        self.link_warned = False

    # -- logging -------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        t = self.drone.clock.now()
        with self.lock:
            self.events.append({"t": None if self.mission_t0 is None else round(t - self.mission_t0, 1), "msg": msg})

    # -- monitor (idle: show detection and distance, before anything flies) ---------------------
    def start_monitor(self) -> None:
        if self.mon_thread and self.mon_thread.is_alive():
            return
        self.quit_evt.clear()
        self.mon_thread = threading.Thread(target=self._monitor_loop, daemon=True, name="monitor")
        self.mon_thread.start()

    def _monitor_loop(self) -> None:
        clock = self.drone.clock
        while not self.quit_evt.is_set():
            if self.mon_pause.is_set() or self.phase not in ("IDLE", "LANDED", "ABORTED", "ERROR"):
                clock.sleep(TICK_S)
                continue
            try:
                self.tel = self.drone.telemetry()
                fr = self.drone.poll_frame()
                dets = fr.detections if fr else []
                est = self.monitor_est.step(clock.now(), dets, 0.0)
                with self.lock:
                    if fr:
                        self.dets = dets
                    self.est = est
            except Exception as exc:                                  # noqa: BLE001 -- the monitor must not die
                self.log(f"monitor error: {exc}")
            clock.sleep(0.1)

    # -- control --------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, policy_name: str, confirmed: bool = False) -> tuple[bool, str]:
        with self.lock:
            if self.busy:
                return False, "a mission is already running"
            if policy_name not in POLICIES:
                return False, f"unknown policy {policy_name!r}"
            if self.drone.live and not confirmed:
                return False, "live drone: confirm that the area is clear and a catcher is ready"
            self.stop_evt.clear()
            self.mon_pause.set()
            self.policy_name = policy_name
            self.policy = POLICIES[policy_name](self.policy_cfg, self.sign)
            self.result, self.reason, self.link_warned = {}, "", False
            self.events.clear()
            self.phase = "PREFLIGHT"
            self.mission_t0 = self.drone.clock.now()
            self.thread = threading.Thread(target=self._run, daemon=True, name="mission")
            self.thread.start()
        return True, "started"

    def stop(self) -> str:
        """Stop button: stop moving and land. Safe to press at any time, any number of times."""
        if not self.busy:
            return "nothing is flying"
        self.stop_evt.set()
        self.log("STOP pressed: stopping and landing")
        return "stopping and landing"

    def close(self) -> None:
        self.stop_evt.set()
        self.quit_evt.set()
        if self.thread:
            self.thread.join(timeout=5)

    # -- helpers -------------------------------------------------------------------------------
    def _send(self, rc) -> None:
        cap = self.cfg.rc_cap
        r = tuple(max(-cap, min(cap, int(v))) for v in rc)
        self.last_rc = r
        if not self.cfg.shadow:
            self.drone.send_rc(*r)

    def _raw(self, t: float) -> RawObs:
        tel = self.tel = self.drone.telemetry()
        fr = self.drone.poll_frame()
        with self.lock:
            if fr:
                self.dets = fr.detections
        return RawObs(t=t, detections=fr.detections if fr else [], new_frame=fr is not None, height_m=tel.height_m,
                      yaw_deg=tel.yaw_deg, speed_mps=tel.speed_mps, battery_pct=tel.battery_pct,
                      vel_fwd_mps=None if tel.vel_body_mps is None else tel.vel_body_mps[0],
                      vel_right_mps=None if tel.vel_body_mps is None else tel.vel_body_mps[1])

    def _preflight(self) -> str | None:
        d, c = self.drone, self.cfg
        tel = d.telemetry()
        if not c.shadow and (tel.flying or (tel.height_m or 0) > 0.3):
            return "the drone is already in the air"
        if tel.link_age_s > 1.0:
            return "no fresh telemetry from the drone"
        if tel.battery_pct is not None and tel.battery_pct < c.min_battery_pct:
            return f"battery {tel.battery_pct:.0f}% is below {c.min_battery_pct:.0f}%"
        if c.require_video:
            t_end = d.clock.now() + 3.0
            while d.clock.now() < t_end:
                if d.poll_frame() is not None:
                    return None
                d.clock.sleep(0.05)
            return "no video frames"
        return None

    def _set_phase(self, p: str, why: str = "") -> None:
        with self.lock:
            self.phase = p
        self.log(f"{p}" + (f": {why}" if why else ""))

    # -- the mission ---------------------------------------------------------------------------
    def _run(self) -> None:
        d, clock = self.drone, self.drone.clock
        try:
            if hasattr(d, "prepare"):
                d.prepare()
            problem = self._preflight()
            if problem:
                self._set_phase("ERROR", problem)
                self.reason = problem
                return
            if self.cfg.shadow:
                self._set_phase("FLYING", "SHADOW MODE: policy runs, no take-off, no sticks are sent")
                self._fly()
                self._set_phase("LANDED", "shadow run finished (nothing was flown)")
                return
            self._set_phase("TAKEOFF", "sending takeoff")
            if self.stop_evt.is_set():
                self._set_phase("ABORTED", "stopped before take-off")
                return
            if not d.takeoff():
                self.reason = "take-off failed"
                self._set_phase("ERROR", self.reason)
                self._land()
                return
            self.log("airborne")
            action = self._fly()
            self._land(action)
        except Exception as exc:                                       # noqa: BLE001 -- always try to land
            self.reason = f"software error: {exc!r}"
            self.log(self.reason)
            try:
                self._send((0, 0, 0, 0))
                self._land()
            finally:
                self._set_phase("ERROR", self.reason)
        finally:
            self.mon_pause.clear()

    def _fly(self) -> Action | None:
        d, clock, c = self.drone, self.drone.clock, self.cfg
        self._set_phase("FLYING", f"policy: {self.policy.title}")
        next_t = clock.now()
        last_state = None
        action = None
        while True:
            now = clock.now()
            if self.stop_evt.is_set():
                self.reason = self.reason or "stopped by the operator"
                self._send((0, 0, 0, 0))
                return action
            if now - self.mission_t0 > (c.shadow_max_s if c.shadow else c.max_mission_s):
                self.reason = "maximum mission time"
                self.log(self.reason)
                self._send((0, 0, 0, 0))
                return action
            raw = self._raw(now)
            if self.tel.link_age_s > c.link_warn_s:
                if not self.link_warned:
                    self.log(f"WARNING: no telemetry for {self.tel.link_age_s:.1f} s")
                    self.link_warned = True
                if self.tel.link_age_s > c.link_land_s:
                    self.reason = "link lost"
                    self._send((0, 0, 0, 0))
                    return action
            else:
                self.link_warned = False
            action = self.policy.step(raw)
            self._send(action.rc)
            with self.lock:
                self.est, self.info = self.policy.est, dict(action.info, state=action.state)
            if action.state != last_state:
                last_state = action.state
                self.log(f"policy state {action.state}" + (f" ({action.info.get('abort_reason')})" if action.info.get("abort_reason") else ""))
            if action.mode in ("done", "abort", "land"):
                self.reason = action.info.get("abort_reason") or "mission complete"
                with self.lock:
                    self.result = {k: action.info.get(k) for k in ("final_range_m", "r_start_m", "out_distance_m", "returned_m",
                                                                  "scale", "flown_m", "abort_reason")}
                    if hasattr(d, "truth"):                 # simulation only: the true state at the moment the policy ended
                        self.result["truth_at_end"] = d.truth()
                return action
            next_t += TICK_S
            clock.sleep(max(0.0, next_t - clock.now()))
            if clock.now() - next_t > 0.5:
                next_t = clock.now()

    def _land(self, action: Action | None = None) -> None:
        d, clock, c = self.drone, self.drone.clock, self.cfg
        aborted = action is not None and action.mode in ("abort", "land")
        self._set_phase("LANDING", self.reason)
        lp = LandingPolicy(self.landing_cfg)
        t_end = clock.now() + c.landing_total_s
        next_t = clock.now()
        attempts = 0
        while clock.now() < t_end:
            raw = self._raw(clock.now())
            act = lp.step(raw)
            self._send(act.rc)
            with self.lock:
                self.info = dict(self.info, state=f"LAND:{lp.state}")
            if act.command == "land":
                self.log(f"landing command ({act.info.get('reason') or 'ok'})")
                ok = False
                while attempts < c.land_retries and not ok:
                    attempts += 1
                    self._send((0, 0, 0, 0))
                    ok = d.land()
                    if not ok:
                        self.log(f"land attempt {attempts} not confirmed")
                if ok:
                    self._set_phase("ABORTED" if aborted else "LANDED", self.reason)
                    return
                self.reason = "LANDING NOT CONFIRMED: catch the drone / use the remote"
                self._set_phase("ERROR", self.reason)
                return
            next_t += TICK_S
            clock.sleep(max(0.0, next_t - clock.now()))
        self.reason = "landing took too long: catch the drone"
        self._set_phase("ERROR", self.reason)
        d.land()

    # -- status for the interface ---------------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            est, info, tel = self.est, self.info, self.tel
            flying_phase = self.phase in ("TAKEOFF", "FLYING", "LANDING")
            return {
                "phase": self.phase, "policy": self.policy_name, "state": info.get("state", ""),
                "mode": "live" if self.drone.live else "simulation", "drone": self.drone.name,
                "detected": bool(est.detected), "tracking": bool(est.tracking), "source": est.source,
                "range_m": _r(est.range_m), "std_m": _r(est.std_m), "naive_m": _r(est.naive_m),
                "bearing_deg": _r(est.bearing_deg, 1), "lateral_m": _r(est.lateral_m), "near_field": est.near_field,
                "height_m": _r(tel.height_m), "battery_pct": _r(tel.battery_pct, 0), "speed_mps": _r(tel.speed_mps),
                "link_age_s": _r(tel.link_age_s, 1), "rc": list(self.last_rc),
                "mission_s": None if self.mission_t0 is None or not flying_phase and not self.result and self.phase == "IDLE"
                else round(self.drone.clock.now() - self.mission_t0, 1),
                "reason": self.reason, "result": dict(self.result), "stop_m": self.policy_cfg.stop_m,
                "scale": info.get("scale"), "events": list(self.events)[-14:], "can_start": not self.busy,
                "can_stop": self.busy and self.phase in ("PREFLIGHT", "TAKEOFF", "FLYING"),
            }


def _r(v, n=2):
    return None if v is None else round(float(v), n)
