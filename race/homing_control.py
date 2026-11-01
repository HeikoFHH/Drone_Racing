"""Start / stop switch around HomingPolicy, for an operator controller with buttons.

The controller's control loop calls ``step()`` every tick (20 Hz) and sends the rc it
returns. The buttons only flip a switch; they never touch the drone themselves:

    from race.homing_control import PolicySwitch, obs_from_detections

    switch = PolicySwitch()                    # HomingCfg() defaults; pass your own if needed

    # "Start policy" button  ->  switch.start()
    # "Stop policy"  button  ->  switch.stop()     # drone floats in place
    # "New mission"          ->  switch.start(new_mission=True)

    while flying:                              # ~20 Hz
        obs = obs_from_detections(t, tag_detector.detect(gray), height_m, yaw_deg)
        rc, info = switch.step(obs)
        send_rc(rc)                            # ALWAYS send, also while stopped
        if info["mode"] in ("land", "done"):
            land()                             # policy finished, or hit a safety limit

What the modes mean
  "float"  policy stopped (or never started): rc = (0, 0, 0, 0). The Tello holds its
           position on its own. Keep sending it: with no rc for ~15 s the Tello lands itself.
  "go"     policy is flying: send the rc as is.
  "done"   mission finished (5 m in front of tag A): rc is (0, 0, 0, 0); land or keep floating.
  "land"   a safety limit tripped: rc is (0, 0, 0, 0); land now. Reason in info["abort_reason"].

Stop takes effect on the very next tick. Start after a stop resumes the same mission where it
was paused; the paused time is hidden from the policy, so its timers (validation window,
300 s mission cap) do not count the pause. Thread-safe: buttons may be on another thread.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Iterable

from race.homing_policy import HomingCfg, HomingObs, HomingPolicy, Sighting

FLOAT = (0, 0, 0, 0)


def obs_from_detections(t: float, detections: Iterable, height_m: float | None,
                        yaw_deg: float | None) -> HomingObs:
    """Build the policy input from tello_calibration TagDetector observations.

    ``detections``: what ``TagDetector.detect(gray)`` returns (TagObservation). Only accepted
    detections with a size in tags.yaml are used. ``height_m``: Tello telemetry h in metres
    (state packet ``h`` / 100). ``yaw_deg``: Tello telemetry yaw in degrees.
    """
    import cv2
    import numpy as np

    sightings = []
    for o in detections:
        if not o.accepted or o.tvec is None:
            continue
        tvec = np.ravel(o.tvec).astype(float)
        R = cv2.Rodrigues(np.asarray(o.rvec, float))[0]
        sightings.append(Sighting(int(o.tag_id), float(np.linalg.norm(tvec)),
                                  math.degrees(math.atan2(tvec[0], tvec[2])),
                                  tuple(float(v) for v in tvec), tuple(float(v) for v in R[:, 2])))
    return HomingObs(t=t, sightings=sightings, height_m=height_m, yaw_deg=yaw_deg)


class PolicySwitch:
    def __init__(self, cfg: HomingCfg | None = None) -> None:
        self.cfg = cfg or HomingCfg()
        self._lock = threading.Lock()
        self._want_run = False
        self._new_mission = False
        self._policy: HomingPolicy | None = None
        self._paused_since: float | None = None
        self._paused_total = 0.0

    # --- buttons --------------------------------------------------------------------------
    def start(self, new_mission: bool = False) -> None:
        """Run the policy. Resumes a paused mission; ``new_mission=True`` starts over
        (the drone must then be back at A, nose toward B)."""
        with self._lock:
            self._want_run = True
            if new_mission or self._policy is None:
                self._new_mission = True

    def stop(self) -> None:
        """Stop the policy: from the next tick the drone floats (rc 0 0 0 0)."""
        with self._lock:
            self._want_run = False

    def reset(self) -> None:
        """Stop and forget the mission; the next start() begins a new one."""
        with self._lock:
            self._want_run = False
            self._policy = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._want_run

    # --- control loop ---------------------------------------------------------------------
    def step(self, obs: HomingObs, dt: float = 0.05):
        """Return ((right, forward, up, yaw), info). Call every tick, send the rc."""
        with self._lock:
            run, fresh = self._want_run, self._new_mission
            if run and fresh:
                self._policy = HomingPolicy(self.cfg)
                self._paused_since, self._paused_total, self._new_mission = None, 0.0, False
            policy = self._policy

        if not run or policy is None:
            if policy is not None and self._paused_since is None:
                self._paused_since = obs.t
            return FLOAT, {"mode": "float", "running": False,
                           "mission": "none" if policy is None else policy.state}

        if self._paused_since is not None:                 # resuming: hide the pause from the policy
            self._paused_total += obs.t - self._paused_since
            self._paused_since = None
        rc, info = policy.step(replace(obs, t=obs.t - self._paused_total), dt)
        info["running"] = True
        return rc, info
