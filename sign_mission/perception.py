"""Frame -> sign detections, and the picture drawn for the operator."""

from __future__ import annotations

import cv2
import numpy as np
import yaml

from .common import RangeEstimate, SignObs
from .range_est import observation_from_corners
from .spec import REPO, CameraModel, SignSpec, load_camera

PUPIL_TO_IPPE = [1, 0, 3, 2]       # pupil_apriltags corner order -> TL, TR, BR, BL (as the repo's TagDetector)


class SignDetector:
    """AprilTag tag36h11 detector. Uses pupil_apriltags (same settings as the repo's TagDetector, config/tags.yaml);
    falls back to OpenCV's aruco module."""

    def __init__(self, sign: SignSpec | None = None, cam: CameraModel | None = None, backend: str = "auto"):
        self.sign = sign or SignSpec()
        self.cam = cam or load_camera()
        det = yaml.safe_load((REPO / "config" / "tags.yaml").read_text())["detector"]
        self.backend = None
        if backend in ("auto", "pupil"):
            try:
                import pupil_apriltags
                self._pupil = pupil_apriltags.Detector(
                    families="tag36h11", quad_decimate=float(det["quad_decimate"]),
                    refine_edges=bool(det["refine_edges"]), decode_sharpening=float(det["decode_sharpening"]))
                self.backend = "pupil"
            except ImportError:
                if backend == "pupil":
                    raise
        if self.backend is None:
            aruco = cv2.aruco
            d = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
            p = aruco.DetectorParameters()
            p.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
            self._aruco = aruco.ArucoDetector(d, p)
            self.backend = "aruco"

    def detect(self, frame_bgr: np.ndarray) -> list[SignObs]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        out = []
        if self.backend == "pupil":
            for d in self._pupil.detect(gray):
                corners = np.asarray(d.corners, float)[PUPIL_TO_IPPE]
                o = observation_from_corners(corners, d.decision_margin, d.tag_id, self.cam, self.sign.width_m)
                if o is not None:
                    out.append(o)
        else:
            corners, ids, _ = self._aruco.detectMarkers(gray)
            for c, i in zip(corners, [] if ids is None else ids.flatten()):
                o = observation_from_corners(c.reshape(4, 2), 50.0, int(i), self.cam, self.sign.width_m)
                if o is not None:
                    out.append(o)
        return out


def annotate(frame: np.ndarray, detections: list[SignObs], est: RangeEstimate | None, banner: str = "",
             stop_m: float = 0.5) -> np.ndarray:
    """Draw the detection and the distance on a copy of the frame (green = sign detected, amber = estimated only)."""
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
    for o in detections:
        cv2.polylines(img, [o.corners.astype(np.int32)], True, (60, 220, 60), 3)
    if est is not None and est.tracking and est.range_m is not None:
        col = (60, 220, 60) if est.detected else (0, 170, 255)
        txt = f"{est.range_m:.2f} m" + (f" +/-{est.std_m:.2f}" if est.std_m is not None else "")
        cv2.putText(img, txt, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5)
        cv2.putText(img, txt, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 2)
        cv2.putText(img, est.source.replace("_", " "), (14, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, est.source.replace("_", " "), (14, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1)
    if banner:
        cv2.putText(img, banner, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, banner, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return img
