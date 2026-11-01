# Sim MVP results (M0-M2)

Checkpoint `outputs/sim/run1/last.pt` (not committed), git SHA `ac084e6`, params `9ecd516d9d54`, **params_source: placeholder** (uncalibrated placeholder dynamics), rc cap 35.
Training: 800 iterations x 4096 envs (about 105M env steps, about 16 min on an M-series Mac, learner on MPS), dynamics DR ramped 0 to 1 over iterations 200-450.

Evaluation: 100 seeds per cell, 60 s horizon, same start states for all policies (stress_sim's seeded initial conditions). Success = truth lap timer finished A to B to A with no crash; lap time is the median of successful laps.

## stress_sim plant (no delay, deadband 0, no yaw lag)

| Scenario | PPO | Baseline (rc cap 20) | Baseline (rc cap 35) |
|---|---|---|---|
| baseline | 100/100, 12.6 s | 100/100, 41.7 s | 100/100, 25.2 s |
| mild-wind | 100/100, 13.2 s | 100/100, 43.1 s | 100/100, 26.1 s |
| strong-wind | 100/100, 14.2 s | 0/100 | 0/100 |
| deadband | 100/100, 12.7 s | 26/100, 42.5 s | 26/100, 26.1 s |

## placeholder plant (delay 0.1 s, deadband 8, lag)

| Scenario | PPO | Baseline (rc cap 20) | Baseline (rc cap 35) |
|---|---|---|---|
| baseline | 100/100, 12.9 s | 100/100, 41.9 s | 100/100, 25.4 s |
| mild-wind | 100/100, 13.3 s | 100/100, 43.4 s | 100/100, 26.4 s |
| strong-wind | 100/100, 14.5 s | 0/100 | 0/100 |
| deadband | 100/100, 12.9 s | 26/100, 42.5 s | 26/100, 25.9 s |

## Reading the numbers

- The MVP criterion of `docs/SIMULATION_PLAN.md` section 15 is met: PPO beats `BaselinePolicy` on nominal lap time, strong wind and deadband 12.
- Lap time is not a like-for-like comparison: the baseline dwells at each waypoint (0.3 s) and is capped at rc 20 by `PolicyConfig`; the rc 35 column matches the PPO cap. Even then the PPO is about 2x faster.
- Under the full random mix used in training (deadband up to 14, delay, gain/lag changes, wind up to 0.25 m/s, wider spawn) the training success rate over the last 50 iterations is about 77%, so the 100/100 above reflects the benign, narrow start states of the stress scenarios, not general robustness.
- Not covered yet (M3+): camera/tag visibility, estimator, latency spikes, sensor faults, link loss. These results say nothing about the real drone: the dynamics are placeholders and no real flight data exists.
