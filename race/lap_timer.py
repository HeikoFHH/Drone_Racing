"""One-lap timing for the two-pole AprilTag course.

Lap definition (authoritative — see docs/RACE_RULES.md):

  1. Start: drone position enters the Pole A gate (tag ``pole_a_tag``) while
     armed/racing, then leaves it. Timer starts on the leave (crossing out).
  2. Turn: drone must enter the Pole B gate (tag ``pole_b_tag``).
  3. Finish: after the turn, drone re-enters the Pole A gate. Timer stops.

Invalid if finish happens before turn, or if pose is lost for too long mid-lap
(caller decides abort). Pose is in the map frame from ``core/positional.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml


class LapPhase(Enum):
    IDLE = auto()
    AT_START = auto()
    OUTBOUND = auto()
    TURNED = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class CourseConfig:
    name: str
    pole_a_tag: int
    pole_b_tag: int
    gate_radius_m: float
    min_height_m: float
    max_height_m: float
    require_turn: bool = True


def load_course(path: Path | str | None = None) -> CourseConfig:
    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(path) if path else root / "config" / "race_course.yaml"
    data = yaml.safe_load(cfg_path.read_text()) or {}
    return CourseConfig(
        name=str(data.get("name", "two-pole-one-lap")),
        pole_a_tag=int(data["pole_a_tag"]),
        pole_b_tag=int(data["pole_b_tag"]),
        gate_radius_m=float(data.get("gate_radius_m", 0.8)),
        min_height_m=float(data.get("min_height_m", 0.4)),
        max_height_m=float(data.get("max_height_m", 2.5)),
        require_turn=bool(data.get("require_turn", True)),
    )


def _dist_xy(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return ((p[0] - q[0]) ** 2 + (p[2] - q[2]) ** 2) ** 0.5


class LapTimer:
    """Feed ``update(t_s, position_xyz, pole_poses)`` every pose tick."""

    def __init__(self, course: CourseConfig, pole_poses: dict[int, tuple[float, float, float]]):
        self.course = course
        if course.pole_a_tag not in pole_poses or course.pole_b_tag not in pole_poses:
            missing = {course.pole_a_tag, course.pole_b_tag} - set(pole_poses)
            raise KeyError(f"map missing pole tags: {sorted(missing)}")
        self.pole_a = pole_poses[course.pole_a_tag]
        self.pole_b = pole_poses[course.pole_b_tag]
        self.phase = LapPhase.IDLE
        self.t_start: float | None = None
        self.t_finish: float | None = None
        self._was_in_a = False
        self._was_in_b = False

    @property
    def lap_time_s(self) -> float | None:
        if self.t_start is None or self.t_finish is None:
            return None
        return self.t_finish - self.t_start

    def in_gate(self, pos: tuple[float, float, float], pole: tuple[float, float, float]) -> bool:
        return _dist_xy(pos, pole) <= self.course.gate_radius_m

    def update(self, t_s: float, pos: tuple[float, float, float]) -> dict[str, Any]:
        """Advance state machine. Returns a status dict for logging / UI."""
        in_a = self.in_gate(pos, self.pole_a)
        in_b = self.in_gate(pos, self.pole_b)
        event = None

        if self.phase == LapPhase.IDLE:
            if in_a:
                self.phase = LapPhase.AT_START
                event = "at_start"
        elif self.phase == LapPhase.AT_START:
            if self._was_in_a and not in_a:
                self.phase = LapPhase.OUTBOUND
                self.t_start = t_s
                self.t_finish = None
                event = "lap_start"
        elif self.phase == LapPhase.OUTBOUND:
            if in_b:
                self.phase = LapPhase.TURNED
                event = "turn"
            elif in_a and self.course.require_turn:
                # Re-entered start without turn — reset to at_start
                self.phase = LapPhase.AT_START
                self.t_start = None
                event = "false_start_reset"
        elif self.phase == LapPhase.TURNED:
            if in_a:
                self.phase = LapPhase.FINISHED
                self.t_finish = t_s
                event = "lap_finish"
        elif self.phase == LapPhase.FINISHED:
            pass

        self._was_in_a = in_a
        self._was_in_b = in_b
        return {
            "phase": self.phase.name,
            "event": event,
            "t_s": t_s,
            "in_a": in_a,
            "in_b": in_b,
            "lap_time_s": self.lap_time_s,
            "elapsed_s": None if self.t_start is None else (t_s - self.t_start),
        }

    def reset(self) -> None:
        self.phase = LapPhase.IDLE
        self.t_start = None
        self.t_finish = None
        self._was_in_a = False
        self._was_in_b = False


def pole_poses_from_map(map_path: Path | str | None = None) -> dict[int, tuple[float, float, float]]:
    """Load tag translation vectors from config/map.yaml as (x, y, z)."""
    root = Path(__file__).resolve().parent.parent
    path = Path(map_path) if map_path else root / "config" / "map.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    out: dict[int, tuple[float, float, float]] = {}
    for tid, entry in (data.get("tags") or {}).items():
        t = entry["t"]
        out[int(tid)] = (float(t[0]), float(t[1]), float(t[2]))
    return out
