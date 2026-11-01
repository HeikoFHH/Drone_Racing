"""M0 gate: end-to-end env steps/s (SIMULATION_PLAN.md section 12; gate = 30k steps/s at 4096 envs)."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from drone_race_sim import EnvCfg, TelloRaceVecEnv


def bench(device: str, num_envs: int, steps: int, dr_level: float) -> float:
    env = TelloRaceVecEnv(EnvCfg(num_envs=num_envs, device=device, dr_level=dr_level))
    env.reset(seed=0)
    act = torch.rand(steps, num_envs, 4, device=env.device) * 2 - 1
    for k in range(10):  # warm-up
        env.step(act[k])
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for k in range(steps):
        env.step(act[k])
    if device == "mps":
        torch.mps.synchronize()
    return num_envs * steps / (time.perf_counter() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 4096, 8192])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--dr-level", type=float, default=1.0)
    a = ap.parse_args()
    for dev in a.devices:
        if dev == "mps" and not torch.backends.mps.is_available():
            print("mps unavailable"); continue
        for n in a.envs:
            print(f"{dev:4s} envs={n:5d} dr={a.dr_level}: {bench(dev, n, a.steps, a.dr_level):10.0f} env-steps/s", flush=True)
