"""Frame conversions. Sim frame is course_flu_v1 (policy_integration/CONTRACT.md).

course_flu_v1: X from Pole A toward Pole B, Y to the left, Z up, metres; yaw is
radians counterclockwise from +X. rc yaw is clockwise-positive, so psi_dot = -rc_yaw.
The repo map frame (config/map.yaml, core/positional.py) is x right, y up, z out of the
tag face; the lap timer measures horizontal distance on map (x, z).
"""

from __future__ import annotations

import math

import numpy as np

# Default fixture for the example map: x_c = x_m, y_c = -z_m, z_c = y_m.
DEFAULT_R_WORLD_MAP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
DEFAULT_T_WORLD_MAP = np.zeros(3)  # real value: (0, 0, origin-tag height), measured on site


def check_rotation(R, tol: float = 1e-6) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.allclose(R @ R.T, np.eye(3), atol=tol):
        raise ValueError("rotation must be orthonormal 3x3")
    if abs(np.linalg.det(R) - 1.0) > tol:
        raise ValueError("rotation must have determinant +1")
    return R


def map_to_course(p_map, R=DEFAULT_R_WORLD_MAP, t=DEFAULT_T_WORLD_MAP):
    """p_course = R @ p_map + t. Accepts (3,) or (N, 3)."""
    R = check_rotation(R)
    return np.asarray(p_map, dtype=float) @ R.T + np.asarray(t, dtype=float)


def course_to_map(p_course, R=DEFAULT_R_WORLD_MAP, t=DEFAULT_T_WORLD_MAP):
    R = check_rotation(R)
    return (np.asarray(p_course, dtype=float) - np.asarray(t, dtype=float)) @ R


def align_to_pole_a(a_xy, b_xy) -> tuple[np.ndarray, float, float]:
    """Return (origin_xy, rotation, distance) so that A -> (0, 0) and B -> (d, 0).

    ``rotation`` is the angle to subtract from vectors in the input frame.
    """
    a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
    d = b - a
    return a, math.atan2(d[1], d[0]), float(np.hypot(*d))


def wrap_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi
