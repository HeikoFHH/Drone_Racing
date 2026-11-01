# Simulate the policy before flying

All work stays on `maude/policy-integration`. The simulation calls the actual
`BaselinePolicy.step()` from this branch, not a separate JavaScript imitation.
No flight commands are sent. No paid cloud machine is provisioned.

## 1. CPU simulation: available now

From the repository root, with Python 3.10+:

```bash
python3 -m unittest policy_integration.test_policy policy_integration.test_stress_sim -v
python3 -m policy_integration.stress_sim --seeds 20 --output policy_integration/reports/stress_sim.json
```

The scenario runner and policy use the standard library. The older independent
lap-timer integration test additionally uses upstream `pyyaml`. Eight scenarios
vary initial conditions and measurement noise across 20 seeds: baseline, two
lateral drifts, two camera delays, camera loss, link loss and an actuator deadband.

The vehicle model is a simple first-order velocity response. Its nominal RC scale
comes from the upstream approximate mapping, not a measured Tello. Lateral
"wind" is an artificial world-velocity disturbance; deadband 12 and the chosen
delays are synthetic stress parameters. These are NOT measured Tello limits.
There are no rotor/body physics, image-rendered AprilTags, real Wi-Fi, or
simulated landing dynamics. Each abort trace ends at the landing request.

Truth position is used only for the plant and independent race checks. The policy
receives noisy, delayed observations. A route-complete decision is reported
separately from a ground-truth lap. This keeps stale/incorrect sensing from being
mistaken for genuine task success.

## Results from this run

| Scenario | Completed | Aborted | Timed out at 90 s |
|---|---:|---:|---:|
| Baseline | 20/20 | 0 | 0 |
| Synthetic lateral drift 0.06 m/s | 20/20 | 0 | 0 |
| Synthetic lateral drift 0.12 m/s | 0/20 | 20 | 0 |
| Camera delay 150 ms | 20/20 | 0 | 0 |
| Camera delay 300 ms | 0/20 | 20 | 0 |
| Camera lost at 5 s | 0/20 | 20 | 0 |
| Link lost at 5 s | 0/20 | 20 | 0 |
| Synthetic RC deadband 12 | 6/20 | 14 | 0 |

The 0.12 m/s drift stalls this PD controller outside its arrival tolerance.
The deadband can remove the small commands needed to finish. These are current
controller weaknesses, not simulation passes. Next experiments should compare
bounded disturbance compensation/integral control and calibrated deadband
compensation. Do not tune to an unmeasured synthetic constant and call the
controller hardware-ready. The new 90-second policy session deadline now bounds
both stalled cases and requests landing; the runner must honor that request.

Continuous 300 ms camera delay exposed an unlimited-hold bug in the previous
policy: the data could stay over the freshness limit without ever exceeding the
absolute pose timeout. The policy now aborts after 0.5 seconds of persistent
staleness; a regression test covers it. This new safeguard is why the delayed
scenario aborts rather than hanging.

## Visual playback

Generate data and a compact playback fragment:

```bash
python3 -m policy_integration.build_view --report policy_integration/reports/stress_sim.json --output /workspace/drone-policy-simulator.html
```

The visual shows recorded Python results. Scenario selection and timeline
scrubbing do not rerun training. To change the controller or plant, modify Python,
rerun the scenarios and rebuild the view. The inline fragment uses the host's
theme/utility styles; it is not a standalone deployed website. The conversation
playback is saved outside git; the reusable template and builder are in this branch.

## 2. CPU rigid-body physics: optional next tier

`gym-pybullet-drones` provides drone physics/RL examples and its upstream README
reports testing on Ubuntu and Apple Silicon macOS:
https://github.com/learnsyslab/gym-pybullet-drones

It is not a ready-made Tello replacement. Its vehicle model and action interface
must be adapted to the four RC-style velocity commands, and its parameters must
be checked against measured Tello responses. Prefer it when the team needs local
rigid-body debugging and has time to build that adapter. It has not been installed
or run here. Avoid changing the RL partner's simulator without agreeing on the
shared contract and test course.

## 3. Existing Isaac Lab environment: team/cloud NVIDIA machine

The team's repository already has `AS-Tello-Race-v0`. Reuse it on a compatible
GPU workstation with the matching Isaac Sim 5.x/Isaac Lab/Action_Space versions
documented in `simulation/README.md`. Do not install an arbitrary latest release
over that older stack. Newer Isaac Lab versions change dependencies and APIs.

For that repository's Isaac Sim 5.1 path, NVIDIA documents Windows/Linux and
supported RTX hardware, not a native macOS workstation:
https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html

Use the team's existing compatible machine first. If using an internet-accessible
GPU host, check RTX support and VRAM before renting: the 5.1 documentation says
GPUs without RT cores, including A100/H100, are unsupported for this Isaac Sim
workflow. No access, cost or availability for a particular provider is assumed.

Before any large training run, the RL owner should register the task IN the
training process, launch ONE environment, step fixed actions, verify action axes,
and check a canonical A/B/A trace against the lap rules. Then add sensor delay,
noise, dropouts and real-state estimation before comparing trained policies.

## 4. Fit the simulator to your real drone

Grounded/shadow logs establish tag scale, camera capture age, coordinate signs
and measurement noise. Controlled physical baseline logs can then estimate RC
deadband, response lag and command-to-speed mapping. Substitute measured ranges
into the scenario suite and rerun identical seeded evaluations for classic and
RL policies. Physical flight and model fitting are still pending.
