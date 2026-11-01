"""Train the race policy (MVP: dynamics-side curriculum, perfect-state observations).

    .venv-sim/bin/python -m train.train --out outputs/sim/run1 --iterations 600

DR schedule: dr_level ramps linearly from 0 at ``--dr-start`` to ``--dr-max`` at ``--dr-end``
iterations (stage 0 = nominal, stage 1 = dynamics DR). Checkpoints carry the git SHA, the
simulator parameter hash and params_source (D8).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from drone_race_sim import EnvCfg, TelloRaceVecEnv
from train.ppo import PPO, PPOCfg


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return "unknown"


def dr_schedule(it, start, end, level_max):
    if it <= start:
        return 0.0
    return level_max * min(1.0, (it - start) / max(1, end - start))


def save(ppo: PPO, env: TelloRaceVecEnv, path: Path, extra: dict):
    torch.save({**ppo.state_dict(), "meta": {"git_sha": git_sha(), "params_hash": env.params_hash,
                                              "params_source": env.params_source, "rc_cap": env.cfg.rc_cap,
                                              "sensing": env.cfg.sensing, "obs_dim": env.obs_dim, "priv_dim": env.priv_dim, **extra}}, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/sim/run"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rc-cap", type=float, default=35.0)
    ap.add_argument("--dr-start", type=int, default=150)
    ap.add_argument("--dr-end", type=int, default=350)
    ap.add_argument("--dr-max", type=float, default=1.0)
    ap.add_argument("--sensing", choices=["perfect", "realistic"], default="perfect")
    ap.add_argument("--save-every", type=int, default=50)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    env = TelloRaceVecEnv(EnvCfg(num_envs=a.envs, device=a.device, seed=a.seed, rc_cap=a.rc_cap, sensing=a.sensing))
    ppo = PPO(env, PPOCfg(), device=a.device)
    if env.params_source == "placeholder":
        print("WARNING: training on placeholder dynamics (params_source=placeholder); not calibrated to a real Tello", flush=True)
    (a.out / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()},
                                                   "params_hash": env.params_hash, "git_sha": git_sha()}, indent=2))
    writer = None
    with open(a.out / "log.csv", "w", newline="") as f:
        for it in range(1, a.iterations + 1):
            level = dr_schedule(it, a.dr_start, a.dr_end, a.dr_max)
            env.set_dr_level(level)
            log = ppo.iterate()
            log["dr_level"] = level
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(log.keys()), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(log)
            f.flush()
            if it % 10 == 0 or it == 1:
                print(f"it {it:4d} dr={level:.2f} succ={log['success']:.2f} lap={log['lap_time_s']:.1f}s "
                      f"ret={log['ep_return']:.1f} ep={log['ep_len_s']:.1f}s kl={log['kl']:.4f} "
                      f"{log['steps_per_s'] / 1e3:.0f}k st/s", flush=True)
            if it % a.save_every == 0:
                save(ppo, env, a.out / "last.pt", {"dr_level": level})
        save(ppo, env, a.out / "last.pt", {"dr_level": level})
    print("saved", a.out / "last.pt")


if __name__ == "__main__":
    main()
