"""Plain data types shared by the estimator, the policies, the runner and the drones."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SignObs:
    """One accepted detection of the sign in one frame (all geometry already derived)."""

    tag_id: int
    corners: np.ndarray            # (4, 2) raw image px, order TL, TR, BR, BL of the tag (IPPE order)
    margin: float                  # detector decision margin
    z_pnp: float                   # depth along the optical axis from the pose solution, m
    z_width: float                 # depth from the tag-horizontal edges (pinhole, yaw-corrected), m
    z_height: float                # depth from the tag-vertical edges (pinhole, pitch-corrected), m
    tan_x: float                   # tag centre direction: (u - cx) / fx   (+ = right of centre)
    tan_y: float                   # (v - cy) / fy   (+ = below centre)
    width_px: float
    height_px: float
    view_deg: float                # angle between the sign normal and the direction to the camera
    fully_visible: bool            # all four corners inside the image (with margin)
    normal_cam: tuple[float, float, float] | None = None   # sign normal in camera axes (pose solution)

    @property
    def side_px(self) -> float:
        return min(self.width_px, self.height_px)


@dataclass
class RangeEstimate:
    """What the drone believes about the sign right now."""

    t: float = 0.0
    detected: bool = False         # accepted detection in the latest frame
    tracking: bool = False         # estimate is usable (detected recently, or dead reckoning within its budget)
    source: str = "none"           # vision | dead_reckoning | near_field | none
    range_m: float | None = None   # filtered distance camera -> sign plane, along the viewing axis
    std_m: float | None = None
    raw_m: float | None = None     # fused single-frame value (before the filter)
    naive_m: float | None = None   # pose-solution distance alone, what a single-cue pipeline reports
    bearing_deg: float = 0.0       # + = sign right of centre
    lateral_m: float = 0.0         # + = sign to the right of the drone
    vertical_m: float = 0.0        # + = sign centre above the camera
    age_s: float = math.inf        # time since the last accepted detection
    spread: float = 0.0            # disagreement between the three distance cues (relative)
    near_field: bool = False       # sign too close to be seen completely: distance is dead-reckoned
    fully_visible: bool = False
    closing_mps: float = 0.0
    note: str = ""


@dataclass
class RawObs:
    """One control tick of input for a policy."""

    t: float
    detections: list[SignObs] = field(default_factory=list)   # empty when no frame arrived or no sign seen
    new_frame: bool = False                                    # a new video frame was processed this tick
    height_m: float | None = None                              # telemetry height above takeoff
    yaw_deg: float | None = None                               # telemetry yaw, + = clockwise
    speed_mps: float | None = None                             # |horizontal velocity| from telemetry (None: unknown)
    vel_fwd_mps: float | None = None                           # ground speed, body axes (None if not trusted)
    vel_right_mps: float | None = None
    battery_pct: float | None = None


@dataclass
class Action:
    rc: tuple[int, int, int, int] = (0, 0, 0, 0)   # right, forward, up, yaw (clockwise +), -100..100
    mode: str = "go"                               # go | done | land | abort
    state: str = ""
    command: str | None = None                     # "land" asks the runner to send the SDK land command
    info: dict = field(default_factory=dict)
