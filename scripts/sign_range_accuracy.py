"""How accurate is the distance to the 22.5 in sign, and what does the filter buy?

Monte Carlo: sign at a known distance (random lateral offset, height and +-25 deg yaw), the four corners get Gaussian
pixel noise, then the estimator sees 150 frames. Reported per distance:
  pose      single-frame depth from the pose solution (what a one-cue pipeline reports)
  fused     the estimator's single-frame choice (pose, or the median of the three cues if the pose disagrees)
  filtered  the Kalman-filtered estimate (static drone)
The default corner noise (1.0 px) is a guess for Tello H.264 video; 0.4 px is the rendered-image detector noise.

    .venv-sim/bin/python scripts/sign_range_accuracy.py --out docs/figures/sign_range_accuracy.png
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from sign_mission.range_est import SignRangeEstimator, observation_from_corners
from sign_mission.sim import SignRenderer, sign_pose_in_camera
from sign_mission.spec import SIGN_WIDTH_IN, SignSpec, load_camera

DISTANCES = (0.8, 1.5, 2.5, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0)


def run(sigma: float, frames: int = 150, seed: int = 0) -> list[dict]:
    cam, sign = load_camera(), SignSpec()
    rend, rng = SignRenderer(cam, sign, 1), np.random.default_rng(seed)
    rows = []
    for d in DISTANCES:
        est = SignRangeEstimator(sign, cam)
        a, b, c = [], [], []
        for k in range(frames):
            yaw = math.radians(rng.uniform(-25, 25))
            R, t = sign_pose_in_camera(np.array([0.03, 0, 1.0]), 0.0, np.array([d, rng.uniform(-0.5, 0.5), 1.0 + rng.uniform(-0.3, 0.3)]))
            R = R @ np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
            dst = rend.project(R, t)
            if dst is None:
                continue
            o = observation_from_corners(dst + rng.normal(0, sigma, dst.shape), 60.0, 1, cam, sign.width_m)
            if o is None:
                continue
            truth = float(t[2])
            e = est.step(k * 0.05, [o], 0.0)
            a.append(o.z_pnp - truth)
            b.append(SignRangeEstimator.fuse(o)[0] - truth)
            if k > 15 and e.range_m:
                c.append(e.range_m - truth)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")
        mx = lambda v: float(np.max(np.abs(v))) if v else float("nan")
        rows.append({"distance_m": d, "pose_rmse_m": rms(a), "fused_rmse_m": rms(b), "filtered_rmse_m": rms(c),
                     "pose_max_m": mx(a), "fused_max_m": mx(b), "filtered_max_m": mx(c)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.4, 1.0, 2.0])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    allrows = {}
    for sg in a.noise:
        allrows[sg] = run(sg)
        print(f"corner noise {sg} px   (RMSE cm | worst cm)")
        for r in allrows[sg]:
            print(f"  {r['distance_m']:5.1f} m   pose {100*r['pose_rmse_m']:6.2f} | {100*r['pose_max_m']:6.1f}   "
                  f"fused {100*r['fused_rmse_m']:6.2f} | {100*r['fused_max_m']:6.1f}   filtered {100*r['filtered_rmse_m']:6.2f} | {100*r['filtered_max_m']:6.1f}")
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        sg = 1.0 if 1.0 in allrows else a.noise[0]
        rows = allrows[sg]
        d = [r["distance_m"] for r in rows]
        for key, lab, col in (("pose", "pose solution, one frame", "#c0392b"), ("filtered", "Kalman-filtered (this work)", "#2a9d4b")):
            axes[0].plot(d, [100 * r[f"{key}_rmse_m"] for r in rows], "o-", color=col, label=lab)
            axes[1].plot(d, [100 * r[f"{key}_max_m"] for r in rows], "o-", color=col, label=lab)
        for ax, t in zip(axes, ("RMSE (cm)", "worst single error (cm)")):
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("true distance to the sign (m)"); ax.set_ylabel(t); ax.grid(alpha=.3, which="both")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Distance error to a {SIGN_WIDTH_IN:g} in sign, corner noise {sg:g} px, 150 frames per distance (simulation)")
        fig.tight_layout(); a.out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(a.out, dpi=110)
        a.out.with_suffix(".json").write_text(json.dumps({str(k): v for k, v in allrows.items()}, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
