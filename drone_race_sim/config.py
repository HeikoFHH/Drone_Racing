"""Config dataclasses and parameter loading for the Mac race sim."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

PARAMS_DIR = Path(__file__).resolve().parent / "params"
DT_NOMINAL = 0.05  # 20 Hz control tick, same as policy_integration and stress_sim


@dataclass
class DynamicsParams:
    """Nominal Tello rc -> velocity model. See SIMULATION_PLAN.md section 6."""

    gain_xy: float = 0.8            # m/s at rc 100, forward and right
    gain_up: float = 0.6            # m/s at rc 100
    gain_yaw: float = math.pi / 3   # rad/s at rc 100
    tau_xy: float = 0.18            # s, first-order lag
    tau_up: float = 0.18
    tau_yaw: float = 0.15
    a_max: float = 3.0              # m/s^2, inf disables
    deadband: float = 8.0           # rc units, |rc| below this is zero
    action_delay_s: float = 0.10
    rc_loss_prob: float = 0.0
    battery_scale: float = 1.0
    lag_mode: str = "exact"         # "exact" or "euler" (stress_sim uses euler)

    @classmethod
    def stress_sim_preset(cls) -> "DynamicsParams":
        """The plant of policy_integration/stress_sim.py (no delay, deadband, yaw lag)."""
        return cls(tau_yaw=0.0, a_max=math.inf, deadband=0.0, action_delay_s=0.0, lag_mode="euler")


@dataclass
class EnvCfg:
    num_envs: int = 1024
    device: str = "cpu"
    dtype: str = "float32"          # "float64" only on cpu (parity tests)
    seed: int = 0
    dt: float = DT_NOMINAL
    episode_s: float = 45.0         # plan says 30 s; at rc cap 35 a lap needs ~25 s, see docs
    rc_cap: float = 35.0            # rc units for action = 1 (D5: 25-35 at stage 0)
    frame_stack: int = 4
    action_history: int = 3
    ref_height: float = 1.0
    dr_level: float = 0.0
    nominal_fraction: float = 0.2   # share of resets that stay nominal at any dr_level
    spawn_radius: float = 0.6       # inside the A gate (gate radius 0.8)
    spawn_yaw_range: float = 0.1    # rad, +/- around "facing B"
    spawn_height: tuple[float, float] = (0.7, 1.2)
    pole_radius: float = 0.25
    pole_height: float = 1.2
    ground_z: float = 0.2
    ceiling_z: float = 3.0
    max_xy_from_course: float = 8.0
    enable_terminations: bool = True
    terminate_on_finish: bool = True
    autoreset: bool = True


def _flatten(raw: dict) -> dict:
    return {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in raw.items()}


def load_dynamics(path: Path | str | None = None) -> tuple[DynamicsParams, str]:
    """Return (params, params_source). Prefers a calibrated yaml when it exists."""
    if path is None:
        calibrated = PARAMS_DIR / "tello_calibrated.yaml"
        path = calibrated if calibrated.exists() else PARAMS_DIR / "tello_placeholder.yaml"
    raw = yaml.safe_load(Path(path).read_text()) or {}
    source = str(raw.pop("params_source", "unknown"))
    flat = _flatten(raw)
    names = {f.name for f in fields(DynamicsParams)}
    return DynamicsParams(**{k: v for k, v in flat.items() if k in names}), source


def load_dr_ranges(path: Path | str | None = None) -> dict[str, tuple[float, float]]:
    raw = yaml.safe_load(Path(path or PARAMS_DIR / "dr_ranges.yaml").read_text())
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_reward_cfg(path: Path | str | None = None) -> dict[str, float]:
    raw = yaml.safe_load(Path(path or PARAMS_DIR / "reward.yaml").read_text())
    return {k: float(v) for k, v in raw.items()}


def params_hash(env_cfg: EnvCfg, dyn: DynamicsParams) -> str:
    """Short hash of everything that defines the simulated world; stored in checkpoints."""
    blob = {"env": asdict(env_cfg), "dyn": asdict(dyn)}
    for name in ("dr_ranges.yaml", "reward.yaml"):
        blob[name] = (PARAMS_DIR / name).read_text()
    text = json.dumps(blob, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]
