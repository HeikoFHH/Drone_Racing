"""Course geometry in course_flu_v1, read from the repo's race configs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _repo  # noqa: F401
from . import frames
from race.lap_timer import load_course, pole_poses_from_map


@dataclass(frozen=True)
class CourseGeometry:
    pole_a_xy: tuple[float, float]
    pole_b_xy: tuple[float, float]
    gate_radius_m: float
    min_height_m: float
    max_height_m: float
    require_turn: bool
    pole_a_tag: int
    pole_b_tag: int

    @property
    def distance(self) -> float:
        return math.hypot(self.pole_b_xy[0] - self.pole_a_xy[0], self.pole_b_xy[1] - self.pole_a_xy[1])

    def with_b_distance(self, d: float) -> "CourseGeometry":
        return CourseGeometry(self.pole_a_xy, (self.pole_a_xy[0] + d, self.pole_a_xy[1]), self.gate_radius_m,
                              self.min_height_m, self.max_height_m, self.require_turn, self.pole_a_tag, self.pole_b_tag)


def load_course_geometry(course_path: Path | str | None = None, map_path: Path | str | None = None,
                         R=frames.DEFAULT_R_WORLD_MAP, t=frames.DEFAULT_T_WORLD_MAP) -> CourseGeometry:
    """Poles from config/map.yaml, converted to course_flu_v1 with A at the origin, B on +X."""
    cfg = load_course(course_path)
    poles = pole_poses_from_map(map_path)
    a = frames.map_to_course(poles[cfg.pole_a_tag], R, t)[:2]
    b = frames.map_to_course(poles[cfg.pole_b_tag], R, t)[:2]
    _, _, d = frames.align_to_pole_a(a, b)
    return CourseGeometry((0.0, 0.0), (d, 0.0), cfg.gate_radius_m, cfg.min_height_m, cfg.max_height_m,
                          cfg.require_turn, cfg.pole_a_tag, cfg.pole_b_tag)


def map_poles_for_lap_timer(geom: CourseGeometry, R=frames.DEFAULT_R_WORLD_MAP,
                            t=frames.DEFAULT_T_WORLD_MAP) -> dict[int, tuple[float, float, float]]:
    """Pole poses in the *map* frame, for feeding the repo's LapTimer from course positions."""
    out = {}
    for tag, xy in ((geom.pole_a_tag, geom.pole_a_xy), (geom.pole_b_tag, geom.pole_b_xy)):
        out[tag] = tuple(float(v) for v in frames.course_to_map([xy[0], xy[1], 0.0], R, t))
    return out
