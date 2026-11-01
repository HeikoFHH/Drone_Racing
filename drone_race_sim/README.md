# drone_race_sim: Mac simulator for the Tello two-pole race

Implements milestones **M0-M2 (the MVP)** of [`docs/SIMULATION_PLAN.md`](../docs/SIMULATION_PLAN.md):
a vectorized torch simulator in the `course_flu_v1` frame, the repo's lap rules, rc dynamics with
lag/delay/deadband/wind, parity with `policy_integration/stress_sim.py`, and a custom PPO.
Sensing (camera/tags/estimator), calibration, ONNX export and deployment are M3+ and not built yet.

```bash
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python -r requirements-sim.txt
.venv-sim/bin/python -m pytest tests                     # 28 tests, incl. exact stress_sim parity
.venv-sim/bin/python scripts/bench_throughput.py         # M0 throughput gate
.venv-sim/bin/python -m train.train --out outputs/sim/run1 --device mps   # ~16 min on an M-series Mac
.venv-sim/bin/python -m train.eval --ckpt outputs/sim/run1/last.pt        # PPO vs BaselinePolicy table
```

| Module | Role |
|---|---|
| `frames.py`, `course.py` | map <-> `course_flu_v1`, poles from `config/map.yaml` / `race_course.yaml` |
| `lap.py` | `VecLapTimer`, torch port of `race/lap_timer.py` (parity-tested) |
| `dynamics.py` | rc -> delay -> deadband -> gain -> lag -> integrate; wind; segment collision test |
| `randomization.py`, `params/*.yaml` | domain randomization (`dr_level` 0..1), placeholder dynamics, reward weights |
| `env.py` | `TelloRaceVecEnv` (gymnasium `VectorEnv`, same-step autoreset, `info["final_obs"]`) |
| `rewards.py` | reward components, each logged |
| `baselines/baseline_policy_adapter.py` | runs Maude's `BaselinePolicy` closed loop; reproduces `stress_sim` exactly |
| `../train/ppo.py`, `train.py`, `eval.py` | asymmetric-critic PPO, training CLI, evaluation table |

**Placeholder parameters.** Dynamics come from `params/tello_placeholder.yaml` (repo-derived or guessed,
each annotated). Checkpoints are stamped `params_source: placeholder` until real flights are calibrated.
