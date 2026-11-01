"""Sign and camera constants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

INCH = 0.0254
SIGN_WIDTH_IN = 22.5
SIGN_WIDTH_M = SIGN_WIDTH_IN * INCH          # 0.5715 m: edge of the black square of the printed tag
STOP_DISTANCE_M = 0.5                        # camera to sign plane, measured along the viewing direction


@dataclass(frozen=True)
class SignSpec:
    """What the sign is. ``width_m`` is the edge length of the tag's black square (AprilTag convention)."""

    width_m: float = SIGN_WIDTH_M
    tag_id: int | None = None                # None: any tag36h11 id; the largest one in view is the sign
    family: str = "tag36h11"
    height_m: float | None = None            # centre above the ground; None = unknown (the drone adapts to it)

    @classmethod
    def from_inches(cls, width_in: float = SIGN_WIDTH_IN, **kw) -> "SignSpec":
        return cls(width_m=width_in * INCH, **kw)


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / 2 / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / 2 / self.fy))

    def min_range_fully_visible(self, width_m: float, margin_px: float = 8.0) -> float:
        """Closest depth at which a centred sign still fits completely in the image (the detector needs all 4 corners)."""
        half = width_m / 2
        rx = self.fx * half / (min(self.cx, self.width - self.cx) - margin_px)
        ry = self.fy * half / (min(self.cy, self.height - self.cy) - margin_px)
        return max(rx, ry)


def load_camera(path: Path | str | None = None) -> CameraModel:
    p = Path(path) if path else REPO / "config" / "camera" / "default.yaml"
    c = yaml.safe_load(p.read_text())
    return CameraModel(K=np.array(c["camera_matrix"], float), dist=np.array(c["dist_coeffs"], float),
                       width=int(c["image_width"]), height=int(c["image_height"]))
