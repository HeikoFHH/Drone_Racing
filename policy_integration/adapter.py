"""Explicit Positioner -> course-frame adapter; no sensor or socket ownership."""

import math
from .policy import Observation, finite


def matvec(r, v):
    return tuple(sum(r[i][j] * v[j] for j in range(3)) for i in range(3))


def transpose(r):
    return tuple(zip(*r))


def rotation(r):
    if len(r) != 3 or any(len(row) != 3 or not finite(row) for row in r):
        raise ValueError("rotation must be finite 3x3")
    for i in range(3):
        for j in range(3):
            dot = sum(r[k][i] * r[k][j] for k in range(3))
            if abs(dot - (1 if i == j else 0)) > 1e-5:
                raise ValueError("rotation is not orthonormal")
    a, b, c = r
    det = a[0] * (b[1]*c[2]-b[2]*c[1]) - a[1] * (b[0]*c[2]-b[2]*c[0]) + a[2] * (b[0]*c[1]-b[1]*c[0])
    if abs(det - 1) > 1e-5:
        raise ValueError("rotation must preserve handedness")
    return tuple(tuple(v for v in row) for row in r)


class PositionerAdapter:
    """Consume locate()'s xyz and R (camera->map), NOT legacy engine tag pose.

    R_world_map and t_world_map must be measured/verified by the sensing owner.
    World Z=0 is the ground. Camera forward +Z must align with drone forward.
    A camera/body extrinsic correction is required if this mounting assumption fails.
    Positioner returns a camera location, not a calibrated vehicle center of mass;
    this prototype assumes the offset is negligible relative to the agreed margin.
    """

    def __init__(self, R_world_map, t_world_map):
        self.R = rotation(R_world_map)
        if len(t_world_map) != 3 or not finite(t_world_map):
            raise ValueError("translation must be finite 3-vector")
        self.t = tuple(t_world_map)

    def point_to_world(self, xyz):
        if len(xyz) != 3 or not finite(xyz):
            raise ValueError("position must be finite 3-vector")
        return tuple(v + off for v, off in zip(matvec(self.R, xyz), self.t))

    def point_to_map(self, xyz):
        return matvec(transpose(self.R), tuple(v - off for v, off in zip(xyz, self.t)))

    def adapt(self, result, *, pose_time_s, telemetry_time_s, velocity_w,
              position_std_m, velocity_std_mps, battery_pct, link_ok, airborne, enabled=False):
        if result is None:
            raise ValueError("no pose: retain last observation with its ORIGINAL capture time")
        r_map_cam = rotation(result["R"])
        # The optical axis is camera +Z. Compute its heading in the canonical frame.
        forward = matvec(self.R, tuple(row[2] for row in r_map_cam))
        if math.hypot(forward[0], forward[1]) < 0.5:
            raise ValueError("camera forward axis is too vertical to determine heading")
        yaw = math.atan2(forward[1], forward[0])
        return Observation(pose_time_s, telemetry_time_s, self.point_to_world(result["xyz"]),
                           tuple(velocity_w), yaw, position_std_m, velocity_std_mps,
                           battery_pct, link_ok, airborne, enabled)
