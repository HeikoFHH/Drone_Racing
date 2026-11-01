"""The interface every drone backend implements (simulated and real), and the clocks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .common import SignObs


class Clock:
    def now(self) -> float:
        raise NotImplementedError

    def sleep(self, dt: float) -> None:
        raise NotImplementedError


class RealClock(Clock):
    """Wall time, optionally sped up (simulation only)."""

    def __init__(self, speed: float = 1.0):
        self.speed = speed
        self._t0 = time.monotonic()

    def now(self) -> float:
        return (time.monotonic() - self._t0) * self.speed

    def sleep(self, dt: float) -> None:
        time.sleep(max(0.0, dt) / self.speed)


class VirtualClock(Clock):
    """Time that only moves when someone sleeps: whole missions run in milliseconds (tests, demo recording)."""

    def __init__(self) -> None:
        self._t = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._t

    def sleep(self, dt: float) -> None:
        with self._lock:
            self._t += max(0.0, dt)


@dataclass
class Telemetry:
    t: float = 0.0
    height_m: float | None = None      # above the take-off spot
    yaw_deg: float | None = None       # + = clockwise
    speed_mps: float | None = None     # horizontal speed magnitude (units of the real drone's report are unverified)
    vel_body_mps: tuple[float, float] | None = None   # (forward, right) ground speed; None if not trusted/known
    battery_pct: float | None = None
    link_age_s: float = 0.0            # time since the last state packet
    flying: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class FrameResult:
    t: float
    frame: np.ndarray | None           # BGR image (None if the backend only produced detections)
    detections: list[SignObs]


class Drone:
    """One drone. The runner is the only caller of send_rc; takeoff/land are blocking SDK commands."""

    live = False
    name = "drone"
    clock: Clock

    def connect(self) -> bool:
        raise NotImplementedError

    def start_video(self) -> bool:
        raise NotImplementedError

    def telemetry(self) -> Telemetry:
        raise NotImplementedError

    def poll_frame(self) -> FrameResult | None:
        """The newest processed frame if it is newer than the last one returned, else None."""
        raise NotImplementedError

    def preview(self) -> np.ndarray | None:
        """Newest frame for display (may be slightly older than the detections)."""
        raise NotImplementedError

    def takeoff(self) -> bool:
        raise NotImplementedError

    def land(self) -> bool:
        raise NotImplementedError

    def send_rc(self, right: int, forward: int, up: int, yaw: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
