# Simulator guide (`drone_race_sim`)

A fast simulator of the two-pole Tello race that runs on a Mac without a GPU, speaks the Gymnasium API, and lets you
test the race policies of the whole team side by side. Everything you need to run it is in this file; the design
reasoning is in [SIMULATION_PLAN.md](SIMULATION_PLAN.md) and the test results in [SIM_TEST_REPORT.md](SIM_TEST_REPORT.md).

**Important:** all dynamics and sensor numbers are placeholders (taken from the repo where possible, otherwise guessed).
There is no real flight data yet, so the results show how the policies compare in the simulator, not how the real drone
will fly.

## What it simulates

```
 policy output (4 sticks)  ->  Tello dynamics  ->  true drone state
                                                         |
              camera + tag detections + state packet     v
   policy input  <-  Kalman filter + dead reckoning  <-  sensor model        (realistic mode)
```

| Part | How it is modelled | Grounded in |
|---|---|---|
| Sticks | right, forward, up, yaw (clockwise), integers -100..100, 20 Hz | `core/tello_dual_video.py` `send_rc` |
| Dynamics | stick delay, deadband (default 8), speed gain (0.8 m/s forward, 0.6 up, 60 deg/s yaw at stick 100), first-order lag (0.18 s), acceleration limit, wind, battery scale | `tools/tello_policy.py` motion model, `policy_integration/stress_sim.py` |
| Course | two poles 4 m apart, gate radius 0.8 m, lap rules A -> B -> A | `config/race_course.yaml`, `config/map.yaml`, `race/lap_timer.py` (the sim's timer is tested against it) |
| Camera | 960x720, 55 deg x 43 deg field of view, tag visibility by angle, distance (apparent size) and viewing angle; tags are one-sided and face each other | `config/camera/default.yaml`, `config/tags.yaml` |
| Tag measurements | detection probability, noise growing with distance, 0.15 s video delay, range scale error | guesses (flagged in `params/sensors_placeholder.yaml`) |
| State estimate | the repo's own Kalman filter (`KalmanCV`), lost after 0.4 s, then dead reckoning on the Tello's velocity report | `core/positional_kalman_filter.py` |
| Lap time | two timers: the true one (for evaluation) and one driven by the estimate (what flight would report) | plan decision D12 |

**Two sensing modes.** `perfect` hands the policy the true state (fast, good for first training). `realistic` hands it
what the real stack would have: a filtered estimate that goes blind when no tag is in view.

**Not modelled yet:** rendered camera images and blur, wrong-pose outliers, burst dropouts, landing, rotor physics,
collisions beyond "hit a pole". Details in [CONTEXT/KNOWN_ISSUES.md](../CONTEXT/KNOWN_ISSUES.md).

## Run it

```bash
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python -r requirements-sim.txt stable-baselines3 opencv-python pupil-apriltags
uv pip install --python .venv-sim/bin/python -e cal_and_map

.venv-sim/bin/python -m pytest -q                                   # about 80 tests, 15 s
.venv-sim/bin/python scripts/bench_throughput.py                    # about 1.8M env-steps/s on CPU
.venv-sim/bin/python -m train.train --out outputs/sim/run --device mps --sensing realistic   # about 20 min
.venv-sim/bin/python scripts/make_report.py --docs-dir docs/figures # all figures + interactive replay
open docs/figures/replay.html                                       # interactive viewer
```

## Use it for reinforcement learning

It follows the Gymnasium API, single and vectorized.

```python
import gymnasium as gym, drone_race_sim

env = gym.make("DroneRace-v0")                                   # one env, numpy, gym.Env
venv = gym.make_vec("DroneRace-v0", num_envs=4096, vectorization_mode="vector_entry_point",
                    sensing="realistic", device="mps")           # torch tensors on the device
obs, info = venv.reset(seed=0)
obs, reward, terminated, truncated, info = venv.step(actions)    # actions in [-1, 1], shape (E, 4)
```

- Observation: 112 numbers (4 stacked frames of 25 + last 3 actions). Action: 4 numbers in [-1, 1], scaled to sticks by `rc_cap` (35).
- Autoreset on the same step; the last observation of a finished episode is in `info["final_obs"]`.
- Extra critic-only inputs (truth) are in `info["priv"]` for asymmetric actor-critic.
- Stable-Baselines3: `SB3VecEnv.create(TelloRaceVecEnv(EnvCfg(num_envs=16)))` (see `tests/test_sim_env.py`).
- The repo's own trainer: `train/ppo.py` (PPO with an asymmetric critic).

## The policies and the safety net

| Policy | Author | Idea |
|---|---|---|
| `maude_baseline` | Maude | slow, careful PD controller along a fixed route with a full set of safety checks |
| `race_policy` | Tvisha | gate racer: keeps the nose on Pole B, steers along the A-B line, searches when no tag is visible |
| `ppo_perfect`, `ppo_realistic` | this repo | reinforcement-learning policies trained in the simulator |

**Maude's policy is the backup for every policy.** `GuardedController(any_policy)` lets the policy fly normally. It takes over,
using Maude's limits, when something is clearly wrong:

- the position estimate leaves the safe box, or the speed is too high -> land;
- no tag fix for too long -> hover, then land;
- the policy outputs invalid numbers, makes no lap progress for 20 s, or gets within Maude's 0.3 m clearance of a pole ->
  Maude's PD route takes over (it lands if its own checks fail).

While the primary policy is healthy the guard does not change its commands (checked by a test).

## Figures

`scripts/make_report.py` writes, under `docs/figures/`:

| File | Shows |
|---|---|
| `replay.html` | interactive replay: top-down and side view, camera cone, estimate vs truth, stick gauges, guard demos |
| `outcomes_perfect.png`, `outcomes_realistic.png` | share of laps / safe landings / crashes per scenario and policy |
| `paths_perfect.png`, `paths_realistic.png` | 10 flown paths per policy (phase colours, backup in orange) |
| `run_<policy>_<sensing>.png` | one run in detail: path, estimate, height, sticks, tag detections |
| `guard_<fault>.png` | the guard reacting to an injected fault |
| `training_*.png` | learning curves |
