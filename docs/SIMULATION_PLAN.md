# Mac simulation plan for the Tello drone race (`drone_race_sim`)

**Status: final plan, v1 · 2026-11-01** · Repo: `HeikoFHH/Drone_Racing` (master at `ac084e6`, includes Maude's `policy_integration/`)

This is the single source of truth for building the Mac-native simulator, the RL training setup, calibration, and the sim-to-real path. It supersedes the earlier drafts of the Mac-sim plan and absorbs the reconciliation with `policy_integration/`. Fault injection and robustness work lives in [`ROBUSTNESS_PLAN.md`](ROBUSTNESS_PLAN.md) and slots into the milestones in section 15.

Conventions: `file:line` points at the repo as of `ac084e6`. **(unverified)** marks numbers or behaviours that no file in the repo confirms (external knowledge, estimates, proposals). All `drone_race_sim/*` modules are proposed; nothing in this plan is implemented yet. "Maude stages" are called **MS0-MS6** (from `policy_integration/EXECUTION_PLAN.md:33-41`) to avoid clashing with curriculum stages and milestones M0-M6.

---

## 1. Goals, scope, non-goals

**Goal.** A vectorized (thousands of parallel environments), Gymnasium-compatible simulator that runs on an Apple Silicon Mac (no NVIDIA GPU) and is as close as possible to the real two-pole Tello race, so that a PPO policy trained in it can be deployed to a real DJI Tello behind the existing safety layers.

**In scope**
- Kinematic Tello dynamics (rc sticks to velocity) with lag, delay, deadband, wind, battery.
- Faithful sensing: AprilTag visibility from the real camera model, noise, dropout, latency, state packet, the repo's Kalman filter, estimator-driven lap timer.
- Course, lap rules and rewards identical to `race/lap_timer.py`.
- Maude's `Observation` / `ActionSample` contract, guards and 8 stress scenarios as the first regression suite.
- Calibration pipeline from real flight logs, domain randomization, curriculum, custom PPO, ONNX export, deployment wrapper and staged real tests.
- Fault-injection layer and robustness benchmark (`ROBUSTNESS_PLAN.md`).

**Non-goals**
- Rigid-body or rotor physics (the Tello closes its own attitude and velocity loops; rc is effectively a velocity setpoint, cf. `tools/tello_policy.py:182-198`). `gym-pybullet-drones` is dropped.
- Rendered camera images or end-to-end vision policies (perception stays the AprilTag detector).
- Replacing the Isaac Lab scaffold (`simulation/`); it is left untouched and gets a one-line pointer to this plan.
- Flying the real drone from the Mac: the engine is Linux-only (finding F2).
- Claiming flight readiness: this plan produces sim evidence and a staged path to it.

---

## 2. Verified findings and constraints

| ID | Finding | Evidence |
|---|---|---|
| F1 | The real engine cannot see race tags 1 and 2: it only knows tags 5, 6, 7, 10 (origin 10) and a legacy `TAG_ID = 14`; it also does not use `race/lap_timer.py`, `positional.py` or the Kalman filter. Deployment needs a real integration step, not a drop-in. | `core/tello_tags.py:18-25`, `core/tello_dual_video.py:34,69`, `config/race_course.yaml:3-4`; confirmed by `python -m policy_integration.audit` |
| F2 | The engine is Linux-only (`SO_BINDTODEVICE`, ioctl `0x8915`, `QT_QPA_PLATFORM=xcb`). The Mac runs sim, training, calibration; real flights need the Linux laptop. | `core/tello_dual_video.py:18,92,105` |
| F3 | Frame conventions conflict: map/positional use x right, y up, z out of the tag face; `tello_tags.py` says y down; the lap timer measures horizontal distance on indices 0 and 2 (x,z); Isaac uses z-up XY; Maude uses `course_flu_v1`. | `config/map.yaml:2-3`, `core/tello_tags.py:74`, `race/lap_timer.py:57-59`, `policy_integration/CONTRACT.md` |
| F4 | The example map is geometrically awkward: both tags have R = I (normals along map +z) and B sits 4 m along +x, so flying A to B sees tag B at about 90 degrees (undetectable). Real tag facing/height/spacing decide whether the task is visible at all. | `config/map.yaml:6-16` |
| F5 | The camera sees a narrow forward cone: 960x720, fx 914.5, fy 914.4, cx 470.6, cy 361.0, about 55 deg x 43 deg FOV. Tag A is invisible when flying A to B. | `config/camera/default.yaml` |
| F6 | Tag range: apparent side is about 120.7 / range px (side 132.08 mm), so about 30 px at 4 m and 15 px at 8 m; detection gate `min_side_px 12`, `max_view_angle_deg 90`. | `config/tags.yaml:6,17,20` |
| F7 | The engine hard-codes focal 871 while the calibrated value is 914.5, a systematic range/pose scale error of about 5%. | `core/tello_dual_video.py:71` vs `config/camera/default.yaml` |
| F8 | The control loop is slower and jittery: `stop.wait(0.05)` after the tick work gives about 13-20 Hz; Kalman self-test uses dt 0.074 s. | `core/tello_dual_video.py:629`, `tools/rc_rate_bench.py:1-21`, `core/positional_kalman_filter.py:182` |
| F9 | "fresh" tag data can be 0.4 s old, and the engine's search override fires at `SEARCH_AFTER_S = 0.4`. It would hijack a racing policy whenever the tag is legitimately out of view; the shim needs a racing threshold of 2-3 s. | `core/tello_dual_video.py:65,421-427,507-516`, `KalmanCfg.lost_after` `core/positional_kalman_filter.py:54` |
| F10 | The real training log is thin and partly misleading: height is overwritten when flying (`:440-442`), no rotation/pixel/tag id/packet age/pitch/roll/ToF, `vel.yaw` is an absolute heading, rc capped at 25-35 in existing flights, no ground truth. | `core/tello_dual_video.py:440-442,614-627`, `tools/tello_policy.py:19-22` |
| F11 | Upstream pieces are missing from this repo: `scripts/fit_tello_motion.py`, `action_space_kit/tags/kalman.py`, `cfg.tag_sensor.video_delay_steps`. | `tools/rc_rate_bench.py:21`, `core/positional_kalman_filter.py:17`, `core/landing_gate.py:27` |
| F12 | The Isaac scaffold calls `_update_lap()` inside the observation step, so phase bonuses and FINISHED lag one step and can mix with resets. It also runs at 60 Hz with privileged observations. Do not replicate. | `simulation/.../tello_race_env.py:155`, `tello_race_env_cfg.py:26,35` |
| F13 | Known hardware risk: on this non-EDU Tello the control link drops within about 2 s of the motors spinning (EMI). Land-on-loss is the fail-safe; a lap on dead reckoning is not allowed. | `FLYING.md:123-131`, `core/tello_dual_video.py:57,373,383-385`, `docs/RACE_RULES.md:31` |
| F14 | The Isaac PPO observation (21 features incl. body angular velocity and gravity) cannot be produced from real sensing; the policy must be retrained on a sensor-realistic interface. | `policy_integration/CONTRACT.md` (RL handoff), `audit.py` |
| F15 | Maude's branch is offline software only: 37 unit tests pass, 160 stress runs reproduce the handoff; it is not wired into the engine and uses 3 mm pose noise and a fixed 20 Hz step. | `policy_integration/HANDOFF.md`, `stress_sim.py:74,118` |
| F16 | The example route of Maude's baseline flies the return leg backward at fixed heading; tag coverage on that leg is unproven. | `policy_integration/README.md`, `HANDOFF.md` |
| F17 | `recordings/` and `outputs/` are empty: there is no real flight data yet, so all dynamics/sensor numbers start as placeholders. | repo tree |
| F18 | The repo is a git repository (`HeikoFHH/Drone_Racing`), so a git SHA is available for model manifests. | `git log` |

Not verified from the code (all treated as placeholders until calibrated): real speed at rc 100 (the repo's 0.8 m/s forward may be low), packet units and quantization (the engine assumes cm/s and cm, `core/tello_dual_video.py:126-165`), packet rate, video latency, camera pitch, Tello behaviour during rc loss, `pupil-apriltags` arm64 wheel availability, MPS throughput.

---

## 3. Decisions

All defaults are proposals; "confirm" means a person must agree before the dependent work is locked.

| # | Decision | Default | Status |
|---|---|---|---|
| D1 | Sim frame | **Adopt `course_flu_v1`** (X from A to B, Y left, Z up, yaw CCW from +X; SI units). Keep only a `map <-> course_flu_v1` conversion (section 5). Drop the earlier `(x_m, -z_m, y_m)` sim frame as a *sim* frame; it survives only as the default test fixture of the conversion. | Decision (default, confirm) |
| D2 | Contract | `Observation`, `ActionSample`, model manifest from `policy_integration/` are the interface of env, estimator, ONNX export and deployment. | Decision (default, confirm with Maude) |
| D3 | Guards | `safety.py` imports thresholds from `PolicyConfig` (`policy_integration/policy.py:78`) and re-implements the logic vectorized, with a parity test against `BaselinePolicy.step()`. Two profiles: `classic` (= `PolicyConfig` defaults) and `rl` (D4). | Decision (default, confirm) |
| D4 | RL guard profile | **Relaxed pose-age with odometry bridging**: the shared estimator bridges tag gaps; the profile raises hold/land pose-age limits (proposal: hold 2.5 s, land 4.0 s) and scales the uncertainty abort (proposal: position std 0.5 m, velocity std 0.5 m/s) so the abort stays hard but is reachable only after real drift. Pose timestamps still keep the original fix time (the contract forbids refreshing a missing fix), so this needs a **CONTRACT v1.1 amendment** from Maude: a stale pose is acceptable when the estimator reports bounded std. The `classic` profile is unchanged. | Decision (default, confirm) |
| D5 | RL rc cap | Own cap schedule: starts at 25-35 (the repo's existing flight caps, `tools/tello_policy.py:19-22`), rises per curriculum stage and **never above the highest rc actually calibrated in real flights**. Maude's `rc_limit = 20` (`policy.py:104`) stays the cap of the `classic` baseline. | Decision (default, confirm) |
| D6 | Learner | Custom compact PPO (about 300 lines, `train/ppo.py`), asymmetric actor-critic. SB3 is only an optional cross-check adapter. | Decision (default, confirm) |
| D7 | Device | Env and learner both on MPS (unified memory), CPU fallback, float32 only, `PYTORCH_ENABLE_MPS_FALLBACK=1`. | Decision (default) |
| D8 | Manifest | Training git SHA = repo SHA (`git rev-parse HEAD` of `HeikoFHH/Drone_Racing`) plus checkpoint hash and `params_source` (placeholder/calibrated). | Decision (default) |
| D9 | Calibration tooling | `scripts/fit_tello_motion.py` is not in this repo: write `calibration/fit_dynamics.py` (and `fit_sensors.py`) here. Action_Space values are reused only if the repo is shared. | Decision (default) |
| D10 | Real flights | Real flights and real fault trials need the Linux laptop with the USB WiFi dongles (engine is Linux-only). The Mac does sim, training, calibration, replay of logs. | Decision (default, confirm laptop availability) |
| D11 | Baseline | `BaselinePolicy` (classic PD waypoint) is baseline #1 in every comparison. The planned `race_pid` is dropped. The ROBUSTNESS_PLAN's references to `race_pid` mean `BaselinePolicy`. | Decision (default) |
| D12 | Official lap time | The repo timer `race/lap_timer.py` fed by the estimate is authoritative for a scored lap (as `CONTRACT.md` requires); the truth-driven timer is used for reward and evaluation only. | Decision (default, confirm) |
| D13 | Heading | Heading is a policy choice, not fixed: the RL policy may yaw to face the next gate; the fixed-heading backward return of Maude's example route stays a baseline-only behaviour until tag coverage is shown. | Decision (default, confirm) |
| D14 | Dynamics placeholder | First-order lag tau 0.18 s (matches `stress_sim.py:118`); gains 0.8 / 0.6 m/s and pi/3 rad/s at rc 100 (`tools/tello_policy.py:192-197`, `policy.py:111-113`); DR widens around them. | Decision (default) |
| D15 | Noise model | Keep the range-dependent pose noise model (section 7); `stress_sim`'s 3 mm noise is used only in the parity cross-check. | Decision (default) |
| D16 | Environment | Separate `.venv-sim` via `uv venv --python 3.11`; the root `requirements.txt` (djitellopy, av) stays untouched. | Decision (default) |

---

## 4. Architecture, layout, dependencies

**Engine:** pure PyTorch, all state in tensors `(E, ...)`, `device` = `mps` or `cpu`. **Policy-side code** (`policy_integration/`) is pure Python and imported unchanged.

**Dependencies** (`requirements-sim.txt`, `pyproject.toml` extras): `torch>=2.4`, `gymnasium>=1.1`, `numpy`, `pyyaml`, `tensorboard`, `matplotlib`, `pytest`. `[calib]`: `scipy`, `opencv-python`, `pupil-apriltags` (arm64 wheel unverified). `[deploy]`: `onnx`, `onnxruntime`. SB3 optional.

**Reuse (no duplication)**

| Reuse | How |
|---|---|
| `race/lap_timer.py` (`load_course`, `pole_poses_from_map`, `update` at `:87-130`) | Single source of gate radius, heights, poles. `VecLapTimer` is a torch port with a parity test. Two timers per env: truth (reward, termination, eval) and estimate-driven (phase observation, official lap time). |
| `core/positional_kalman_filter.py` (`KalmanCV`, `KalmanCfg`: accel 1.5, meas 0.15, chi2 16.27, max_rejects 3, lost_after 0.4, `:44-54`) | Used as is with `xp=torch`; dt passed as an `(E,1)` tensor for loop jitter. |
| `policy_integration/policy.py` (`Observation :34`, `ActionSample :66`, `PolicyConfig :78`, `BaselinePolicy :151`) | Contract, guard thresholds, baseline #1. |
| `policy_integration/stress_sim.py` (`SCENARIOS :31`, `run :43`) | First regression suite; pure-Python plant is the parity reference for the torch dynamics. |
| `policy_integration/adapter.py` | Source of the measured `R_world_map`, `t_world_map` conversion. |
| `tools/tello_policy.py` (`Pid`, `track_rc`, `predict_tag`) | Placeholder motion numbers; `engine_replica` regression of the engine's approach logic. |
| `config/*.yaml`, `config/camera/default.yaml` | Read by `course.py`, `camera.py`. |
| `core/build_flight_viz.py`, `race/time_lap.py` | Sim exports `*.live.jsonl` so existing viz/timer run on sim rollouts. |

**Layout (create)**

```
requirements-sim.txt, pyproject.toml          (extras: calib, deploy)
drone_race_sim/
  __init__.py       register "DroneRace-v0" (verify gymnasium 1.1 vector_entry_point)
  _repo.py          sys.path shim: repo root, core/, tools/ (same trick as tello_dual_video.py:29-31)
  config.py         dataclasses EnvCfg, DynamicsParams, SensorCfg, SafetyCfg, RewardCfg, DRCfg, CurriculumCfg; params hash
  frames.py         map<->course_flu_v1 conversion (R_world_map, t_world_map); yaw sign
  course.py         CourseTensors from race_course.yaml/map.yaml (+ geometry DR hooks)
  lap.py            VecLapTimer (parity with race.lap_timer)
  contract.py       ObsBatch (tensor mirror of Observation), ActionSample builder, manifest writer
  dynamics.py       TelloDynamics.step(): rc pipeline + kinematics + wind + pole collision
  camera.py         projection, FOV/size/view-angle visibility, detection-probability LUT
  sensors.py        StatePacketSensor, TagSensor, true-state ring buffers
  estimator.py      RaceStateEstimator (KalmanCV + heading anchor + fix_age/lost/std); xp-generic, shared sim/deploy
  safety.py         vectorized guards from PolicyConfig; profiles classic / rl; search override; link-loss handling
  obs.py            ObsBuilder: actor frame + history; critic privileged
  rewards.py        reward components dict
  randomization.py  DR sampler by dr_level
  curriculum.py     stage manager (stages 0-4, R1-R4)
  faults.py, scenarios.py, adr.py, failure_buffer.py, indicators.py, overrides.py   (ROBUSTNESS_PLAN 6.1)
  env.py            DroneRaceCore + TelloRaceVecEnv (SAME_STEP autoreset, info["final_obs_*"])
  wrappers.py       NumpyVecWrapper, SB3VecAdapter, SingleEnv (num_envs=1)
  baselines/baseline_policy_adapter.py   runs policy_integration.BaselinePolicy per env (scalar loop for eval)
  baselines/engine_replica.py            replica of the engine's approach/land logic (regression only)
  params/tello_placeholder.yaml, dr_ranges.yaml, reward.yaml, safety.yaml, faults.yaml (+ generated tello_calibrated.yaml)
train/ppo.py, train.py, eval.py, export_onnx.py, configs/ppo_race.yaml
calibration/logs.py, fit_dynamics.py, fit_sensors.py, validate.py, synth_detect_curve.py, PROTOCOL.md
tools/tello_sysid_policy.py                    (flight script for the engine)
eval/robustness_suite.py, robustness_report.py; benchmarks/robustness.yaml
core/race_policy_runtime.py (deploy), core/fault_injection.py (engine shim, flag-gated)
viz/topdown.py, viz/to_live_jsonl.py
scripts/bench_throughput.py, scripts/run_robustness_ci.sh
tests/ ...
```

`simulation/` is untouched apart from a pointer line in `simulation/README.md`.

**Throughput reasoning (estimate, measured at M0).** About 300-400 small tensor ops per step on 4096-length vectors, roughly 5-15 ms per env step; the PPO update dominates (batch about 131k, 5 epochs: about 1 TFLOP per iteration; about 5-10 s on CPU, about 1 s on MPS) *(unverified)*. Expected 50-150k end-to-end steps/s, so 100M steps in about 15-35 minutes. MPS caveats: no float64, avoid `linalg.inv` (write 3x3 maths explicitly), per-device `torch.Generator`.

---

## 5. Frames and contract

### 5.1 `course_flu_v1`

World: X from Pole A toward Pole B, Y to the left of that direction, Z up from the ground, metres, m/s. Yaw is radians CCW from +X. Pole A at the origin, Pole B at `(d, 0)`; `d` from `config/map.yaml` (4 m in the example) and geometry DR. rc order is `(right, forward, up, yaw_clockwise)` (`policy.py` `ACTION_ORDER`). Body axes: forward `f = (cos psi, sin psi)`, left `l = (-sin psi, cos psi)`, right `= -l = (sin psi, -cos psi)`.

### 5.2 Map conversion

`p_course = R_world_map @ p_map + t_world_map` (rigid, orthonormal, det +1; `t` includes the origin tag's height above the floor). Default test fixture for the current `map.yaml` (x right, y up, z out of tag face): rows `x_c = x_m`, `y_c = -z_m`, `z_c = y_m`:

`R_world_map = [[1,0,0],[0,0,-1],[0,1,0]]` (det +1), `t = (0, 0, h_tagA)`.

The real values come from a hand-carry measurement (forward, left, up, yaw-left sign checks and tape-measured scale) owned by the sensing owner (open question Q5). The Kalman filter is per-axis independent, so the axis permutation does not affect it. The repo lap timer takes map X/Z: pass map coordinates through the inverse transform (`PositionerAdapter.point_to_map`).

### 5.3 Observation (estimator output, per policy tick)

Matches `Observation` (`policy.py:34`): `pose_time_s` (effective capture time of the last accepted fix, never refreshed), `telemetry_time_s`, `position_w`, `velocity_w`, `yaw_rad`, `position_std_m`, `velocity_std_mps`, `battery_pct`, `link_ok`, `airborne`, `enabled`, `frame = "course_flu_v1"`, `schema_version = 1`. In the sim these are tensors (`ObsBatch`, same field names, `to_observation(i)` for parity tests). One monotonic clock for all timestamps.

### 5.4 Action and manifest

Policy output `Box(-1, 1, (4,))` = `(right, forward, up, yaw_cw)`, mapped to integer rc by the cap (`rc = round(clip(a * rc_cap, -100, 100))`). Deployed as `ActionSample` (`policy.py:66`): values, `generated_at_s`, `observation_pose_time_s`, `model_id`, `waypoint_index`, action order and frame. Constraints from the contract: at most 0.1 s old, not future-dated, same pose time and waypoint index as the current observation, pinned model ID.

Manifest written by `export_onnx.py` next to the model: git SHA (D8), checkpoint hash, ordered observation features with units/frames/normalization/clip rules, action semantics and caps, simulator parameter hash and `params_source`, evaluation seeds and results, deterministic-inference settings.

**Contract amendments proposed (need Maude's agreement):** v1.1 allows a stale pose under the `rl` guard profile when std is bounded (D4); adds optional fault-indicator fields (section 7.3) to the manifest feature list (not to `Observation`).

---

## 6. Dynamics model (`dynamics.py`)

The Tello's flight controller closes attitude and velocity loops internally, so rc is a velocity setpoint with a short settling time. State per env: position `p` (3), heading `psi` (CCW), body velocity `(right, forward, up)`, yaw rate; all in `course_flu_v1`.

Per control step, with per-env `dt` (0.05-0.08 s; the policy observes its own dt, F8):

1. **Action to rc:** `rc = round(clip(a * rc_cap, -100, 100))`.
2. **Packet loss:** with probability `p_loss` the rc is not delivered and the previous applied rc persists.
3. **Action delay:** ring buffer of the last 5 applied rc vectors, linearly interpolated at a continuous `action_delay_s` (placeholder 0.10 s). *This is new relative to `stress_sim`, which only delays the camera.*
4. **Deadband:** `|rc| < db -> 0` (placeholder db = 8, evidence: the PID creeps at +/-8, `tools/tello_policy.py:150-154`; `stress_sim` stresses db = 12).
5. **Gain map:** `v_target = G_axis * sign(u) * |u/100|^p` (p = 1 by default), `battery_scale` in [0.85, 1.0].
6. **First-order lag (exact for any dt):** `v += (1 - exp(-dt/tau)) * (v_target - v)`, plus acceleration clamp `|dv| <= a_max * dt`.
7. **Wind and drift:** `v_world = Rz(psi) * v_body_xy + k_wind * wind(t) + drift_bias(t)`; wind = per-episode mean plus Ornstein-Uhlenbeck gust.
8. **Integrate:** `p += v_world * dt`, `psi_dot = -G_yaw * u_yaw/100` (rc yaw positive is clockwise), height clamped to `[0, ceiling]` with a ground-contact flag.
9. **Collisions:** analytic distance from each pole to the segment `p_prev -> p_new`; pole radius about 0.25 m (pole half-width plus drone half-width). A hit sets the crash flag.

| Parameter | Placeholder (source) | Calibrated by | DR range |
|---|---|---|---|
| G forward / right | 0.8 m/s per 100 rc (repo; **real value may be higher, unverified**) | E1 step responses | x [0.8, 1.2], +/-5% left/right asymmetry |
| G up | 0.6 m/s | E1 | x [0.8, 1.2] |
| G yaw | pi/3 rad/s (60 deg/s) | E1 | x [0.8, 1.2] |
| tau xy / up / yaw | 0.18 s (stress_sim), 0.18, 0.15 (guess) | E1 | [0.12, 0.45] / [0.12, 0.40] / [0.10, 0.35] |
| a_max | 3 m/s^2 (guess) | E1 | [2, 5] |
| deadband | 8 (repo-derived) | E2 | [4, 12] |
| action delay | 0.10 s (guess) | E1 + log timing | [0, 0.15] |
| rc packet loss | 1% (guess) | rc_rate_bench / link logs | [0, 3%] |
| loop period | 0.05 s nominal | E6 | [0.05, 0.08] |
| wind mean / gust / k_wind | 0.5 / 0.3 / 0.2 (guess) | outdoor logs | [0, 1.5] / [0, 0.5] / [0.05, 0.4] |
| battery gain scale | 1 | per-run battery log | [0.85, 1.0] |

Not captured, with approximation: propwash and ground effect (gain/tau DR plus small altitude-dependent up-gain noise), camera pitch lean under acceleration (phase-2 term, needs packet pitch), real collision response (termination), Tello onboard behaviours (15 s auto-land, thermal/battery throttling; battery appears only as a gain scale). Takeoff and landing are blocking SDK calls outside the policy (`core/tello_dual_video.py:568-587`); episodes start in post-takeoff hover (height N(0.9, 0.1) m, v = 0).

**Parity cross-check.** `stress_sim.run()` (`stress_sim.py:43-129`) implements the same plant in pure Python (dt 0.05, tau 0.18, rc mapping). A test drives both with identical rc sequences and asserts trajectories match within 1e-4 m when delay, deadband and noise are off (semigroup and batch-independence tests are listed in section 14).

---

## 7. Sensors, estimator, observation vectors

Principle (`core/positional_kalman_filter.py:17-20`): train on exactly the estimate the policy gets in flight.

### 7.1 TagSensor (`sensors.py`, `camera.py`)

For each tag (A = 1, B = 2, `config/tags.yaml`), pose and normal come from `map.yaml` converted to `course_flu_v1`; tag height from DR (placeholder 1.2 m, `tello_race_env_cfg` pole_height).

- **Latency:** the fix is generated from the true pose at `t - video_delay` (placeholder 0.15 s, DR 0.08-0.30 s) and consumed as if current (as the real code does). Name mirrors `video_delay_steps` (`core/landing_gate.py:27`).
- **Detector rate:** placeholder 12 Hz, DR 8-20 Hz; a detection opportunity occurs with probability `min(1, f_det * dt)`.
- **Visibility gate** (OpenCV camera frame): `z_cam > 0.2 m`; tag square projected inside 960x720 with margin; projected side `fx * size / z * cos(view)` at least 12 px; view angle at most `max_detect_view_deg` (placeholder 70, DR 60-80; `tags.yaml` allows 90); camera pitch placeholder -10 deg, DR +/-5 *(calibration item)*.
- **Detection probability** `p(side_px, view_angle, blur)` LUT built offline by `calibration/synth_detect_curve.py` (render tag with `warpPerspective`, run the real `TagDetector` with the repo's K and gate); blur proportional to speed and yaw rate; plus floor dropout 0-15% and Markov burst dropouts (mean 3 frames).
- **Pose noise:** depth sigma `a_d * z^2` with `a_d` about 0.0058 per m (about 9 cm at 4 m); lateral sigma `a_l * z`, `a_l` about 0.015 rad (about 6 cm at 4 m); times a DR factor [0.7, 1.6]. Ambiguity flips with probability [0, 5%] (rising with range and near-frontal view) displace the fix 0.3-1.0 m, exercising the KF chi-square gate. Per-episode systematic errors: range scale N(0, 3%) (covers 871 vs 914.5, F7) and pole map error N(0, 0.05-0.2 m).
- **Fusion:** both tags accepted in one frame are averaged (`core/positional.py:70`); heading fix from `R` (`core/positional.py:79`) with sigma 2-5 deg growing with range.

### 7.2 StatePacketSensor and estimator

- **Packet:** h, vgx/vgy/vgz, yaw (`core/tello_dual_video.py:126-165`). Rate 10 Hz sample-and-hold with jitter *(unverified)*; latency placeholder 0.07 s (DR 0.03-0.15); velocity noise 0.01-0.05 m/s plus bias plus scale +/-15%; height noise 0.01-0.03 m with quantization 0.01 or 0.1 m; yaw 1 deg quantization plus gyro drift 0-0.5 deg/s. **Calibration step 1 must confirm units** (cm/s vs dm/s).
- **`RaceStateEstimator`** wraps `KalmanCV`: `predict(dt)` every tick, `update(fix)` on fix, `lost` after `lost_after`, outputs position, velocity, `fix_age`, `lost`, position std. New: heading estimate `yaw_est = yaw_packet + offset`, offset re-anchored at each tag fix by a complementary filter (alpha about 0.3). Same class runs on numpy (deploy) and torch (sim); parity test against `FilteredPositioner`.

### 7.3 Observation vectors

**Actor, per frame (25 dims, all realizable)**, in `course_flu_v1` body axes (forward, **left**):

| Dims | Content |
|---|---|
| 2 | rel A in estimated body frame (forward, left), /4 m |
| 2 | rel B in estimated body frame |
| 1 | height estimate minus reference height |
| 3 | measured body velocity (packet) |
| 1 | measured yaw rate (packet yaw differences) |
| 4 | tag A: seen_now, u, v (-1..1), log range (zeros if unseen) |
| 4 | tag B: same |
| 1 | `fix_age` / 2 s, clipped |
| 1 | `lost` flag |
| 1 | KF position std |
| 4 | lap phase one-hot from the estimate-driven timer (IDLE, AT_START, OUTBOUND, TURNED) |
| 1 | normalized dt of the previous loop |

Fault-awareness indicators from `ROBUSTNESS_PLAN.md` section 3.5 (about +10 dims: per-tag fix age, packet age, KF reject count/NIS, velocity-vs-tag residual, loop dt, height residual, frozen frame, `rc_cap`, `shim_active`) are appended at milestone RB2; they only enter if computed by the shared estimator code path. Actor input = 4 stacked frames plus last 3 actions: **112 dims at M2/M3, about 155 dims with indicators** (exact dims frozen in `obs.py`).

**Critic (asymmetric), about 160 dims at M3, about 200 with fault flags:** the actor observation plus true position relative to poles, true body velocity, yaw (sin/cos) and rate, true tag visibility, estimator error (position/heading), true phase and lap time, wind, normalized dynamics and sensor parameters, pole geometry; plus per-family active-fault one-hots, severities and time-to-fault-end (`ROBUSTNESS_PLAN.md` section 3.4).

---

## 8. Course, lap logic, rewards, terminations, guards

**Course.** Geometry from `load_course()` / `pole_poses_from_map()` (`race/lap_timer.py`), converted to `course_flu_v1` (gate radius 0.8 m, min/max height 0.4/2.5 soft, `require_turn: true`, `config/race_course.yaml:7-14`). The horizontal gate distance is Euclidean in sim x,y. Spawn: inside the A gate (radius up to 0.6 m), height 0.7-1.2 m, yaw curriculum-dependent.

**Lap rules** (`VecLapTimer` must reproduce `race/lap_timer.py:87-130` exactly): IDLE to AT_START when in A; AT_START to OUTBOUND needs "was in A last tick and now not"; in OUTBOUND, `in_b` takes precedence, re-entering A without B resets to AT_START when `require_turn`; TURNED to FINISHED on `in_a`. The `_update_lap` ordering bug of the scaffold (F12) is avoided: lap state updates before rewards and dones.

**Geometry DR** (the map is "remap on site"): B distance 3-7 m, B bearing +/-45 deg, tag normals +/-45 deg around "facing the other pole", tag heights 0.9-1.5 m.

**Terminations**

| Event | Handling |
|---|---|
| Truth lap FINISHED | success; optional `finish_brake_steps = 20` (rc 0 then land) with pole hit / out-of-bounds penalized |
| Pole collision | terminate, -10 |
| Ground contact (z < 0.2) | terminate, -10 (engine treats < `MIN_HEIGHT_M` 0.25 as grounded) |
| Ceiling (z > 3.0) | terminate (soft limit in yaml 2.5) |
| Out of bounds (> 8 m from the course midpoint) | terminate |
| All tags lost > 4 s (stage 3+) | terminate with penalty (the real engine would be sweeping; attempt dead) |
| Timeout 30 s | truncate (scaffold used 45 s; Maude's baseline needs about 50 s, so baseline comparisons use a 60 s horizon, stated in every report) |
| Link outage | Poisson hazard (DR 0-0.2/s, duration 0.5-3 s): a **truncation, not a failure**: `LAND_ON_LOSS` lands on reconnect and the attempt is void (`docs/RACE_RULES.md:31`); logged as a separate "uninterrupted lap" metric. Hold last rc about 0.3 s then hover *(unverified)* |

**Reward (per control step, not multiplied by dt)**: potential progress `w_p * (d_prev - d_curr) / (v_ref * dt)` using true state; events leave A +1, hit B +5, finish +10 + `k * (T_ref - T_lap)`; time -0.02 per step while the lap is active; false start -1; smoothness -0.01*|a_t - a_(t-1)|^2 and -0.005*|jerk|^2; height penalty outside [0.5, 2.2] m; visibility -0.05 per step when the needed tag has been unseen > 0.5 s (escalating after 2 s) and suppressed during `unavoidable` faults (`ROBUSTNESS_PLAN.md` 3.3); FOV-edge penalty `relu(|u| - 0.8)^2`; pole proximity shaping below 0.4 m; crash -10. Weights in `params/reward.yaml`; every component logged separately.

**Safety shim (`safety.py`)** runs inside the training loop at deployment thresholds so the policy trains against what it will experience (masked policy-gradient on shim-controlled steps, `ROBUSTNESS_PLAN.md` 3.8).

| Guard | `classic` profile (= `PolicyConfig`) | `rl` profile (D4, proposals) |
|---|---|---|
| pose age: hold / land | 0.25 s / 0.75 s (`policy.py:86-89`) | 2.5 s / 4.0 s, estimator bridges by odometry |
| persistent staleness abort | 0.5 s | folded into the pose-age limits |
| control gap abort | 0.2 s | 0.2 s |
| telemetry age | 0.25 s | 0.25 s |
| session deadline | 90 s from first airborne observation | 90 s |
| position / velocity std abort | 0.15 m / 0.25 m/s | 0.5 m / 0.5 m/s |
| speed abort | 1.5 m/s | per calibrated max speed |
| geofence, pole clearance 0.3 m, battery 25%, implausible pose jump | as `PolicyConfig` | as `PolicyConfig` |
| aim-before-translate yaw guard | |yaw error| > 0.35 rad zeroes right/forward | disabled for RL (heading is a policy choice, D13); replaced by pole-clearance guard |
| rc cap / slew | 20 / 60 rc/s | stage schedule from 25-35 (D5) / slew tuned at calibration |
| search override | n/a | after 2.5 s without any tag, engine-style yaw sweep (yaw 25, 2 s legs, `tello_dual_video.py:65-67,507-516`) instead of the engine's 0.4 s |
| abort | latched; `request_land` advisory | latched; handover to hover-then-land, not to the PID baseline |

`params/safety.yaml` mirrors the engine constants; an AST test parses `core/tello_dual_video.py` (it cannot be imported on a Mac) and `policy_integration/policy.py` and asserts the values still match.

---

## 9. Scenario and regression suite

**Suite 1 (the 8 ported `stress_sim` scenarios, 20 seeds each, `stress_sim.py:31-40`)**, run through `BaselinePolicy` closed loop in both the pure-Python reference and the torch sim, and later through the trained policy:

| Scenario | Parameter | Baseline result (verified, 160 runs) | Role for RL |
|---|---|---|---|
| baseline | none | 20/20 | nominal lap-time reference |
| mild-wind | lateral drift 0.06 m/s | 20/20 | must stay >= baseline |
| strong-wind | lateral drift 0.12 m/s | 0/20 (session aborts) | **RL target**: PD stalls outside tolerance |
| delay-150 | camera delay 150 ms | 20/20 | must stay >= baseline |
| delay-300 | camera delay 300 ms | 0/20, expected abort | must end safely |
| pose-loss | camera lost at 5 s | 0/20, expected abort | `classic` aborts; `rl` profile must bridge then land safely |
| link-loss | link lost at 5 s | 0/20, expected abort | void attempt; safe landing request |
| deadband | rc deadband 12 | 6/20 | **RL target**: needs deadband compensation |

Additions to suite 1 in the torch sim: **action delay** (absent in `stress_sim`), loop-period jitter, and realistic noise (D15). Dynamics-side scenarios (baseline, drift, deadband) are available from M1; sensor-side ones (camera delay, pose loss, link loss) from M3.

**Suite 2 (robustness benchmark):** about 45 scenarios from the catalogue in `ROBUSTNESS_PLAN.md` section 2 (families P, T, S, C, A, E, SW, X) x severities S0-S3 x 64 seeds, with thresholds in its section 4.4. Mapping of suite 1 onto the catalogue: strong-wind/mild-wind = A-4/A-5, deadband = A-8 (deadband shift), delay-150/300 = T/S latency spikes (S-11, C-5), pose-loss = T-1/T-3, link-loss = C-7/C-8 *(IDs follow the robustness catalogue; confirm when `params/faults.yaml` is frozen)*.

Both suites report lap success (truth and estimate-driven timers), lap time, crash causes, and the `BaselinePolicy` column; regression gates as in `ROBUSTNESS_PLAN.md` 4.4/4.5.

---

## 10. Calibration and validation

Until real data exists, the sim runs on `params/tello_placeholder.yaml`: every parameter annotated `source: repo | guess | doc` with a confidence; the env prints a warning and stamps `params_source: placeholder` into checkpoint metadata.

**Logging patch first (engine, around `core/tello_dual_video.py:614-627`)**: add raw height (not the 0.5 fudge at `:440-442`), packet age and the full raw packet (pitch, roll, ToF, baro, accelerations, battery), tag id, pixel center/size, `R`, decision margin, tag age, measured dt, `t_send`, wifi state, rc as sent. A sysid flight mode (`TELLO_POLICY=sysid`, `tools/tello_sysid_policy.py`) replaces `track_rc` (`:495`) while keeping the safety overrides.

**Flights to record (priority order; Linux laptop, D10)**

| ID | Experiment | Gives |
|---|---|---|
| E0 | Hover 60 s at 3 heights | drift, packet noise, h/vg quantization, vg unit check |
| E1 | Step responses per channel: rc in {15, 30, 50, 75, 100 as safe} for 1.5 s then 0 for 2.5 s, 3 repeats | gain, tau, delay, saturation |
| E2 | Deadband staircase 0 to 30 in steps of 1, 1.5 s each | deadband per channel |
| E3 | Random multi-channel holds (0.3-1.0 s, 60-90 s) | held-out validation set (never fit) |
| E4 | Tag sensor tests with motors off (carried/tripod): marked distances 1-6 m, angles 0/30/60 deg, 100+ frames each; a few with motors on | pose noise vs range, detection vs range/angle/blur, 871-vs-914 scale error, tag-vs-packet latency |
| E5 | PID approach runs with existing logic | `engine_replica` regression set |
| E6 | Loop timing (`tools/rc_rate_bench.py`) on the race laptop | dt distribution |

Record battery per run. Ground truth is weak (the Tello's velocity is optical-flow based): use tag-derived camera pose, a tape grid or phone video; tag-derived velocity is the cross-check.

**Fitting.** `calibration/fit_dynamics.py` fits the same torch dynamics code (float64, CPU) in 3-4 s windows initialized from the measured state (multiple shooting) with `scipy.optimize.least_squares`; delay by grid search (0-300 ms in 25 ms steps); noise statistics from E0. `fit_sensors.py` fits pose-noise parameters, detection LUT correction, latency and frame rate. Output: `params/tello_calibrated.yaml`.

**Validation thresholds (proposed)**

| Level | Metric | Threshold |
|---|---|---|
| Open loop (E3, 2 s windows re-initialized) | normalized velocity RMSE (RMSE / p95 abs v) per channel | <= 0.15 |
| | R-squared | >= 0.8 |
| | steady-state gain error across amplitudes | <= 10% (else enable the power law or a lookup) |
| | yaw-rate RMSE | <= 10 deg/s |
| | 3 s integrated displacement median error vs tag truth | <= 0.20 m |
| Sensors | detection probability per range/angle bin (>= 30 frames) | within 0.10 |
| | fix noise sigma ratio sim/real | [0.67, 1.5] |
| | flip rate | within 2x |
| | latency / fix rate | within 60 ms / within 20% |
| Closed loop (E5 replayed through `engine_replica`, n >= 5) | time-to-land | within 25% |
| | range-vs-time RMSE | <= 0.3 m |
| | mode sequence (climb, aim, lane, track, land) | identical |
| Coverage | every fitted parameter +/-2 sigma inside the DR range | `validate.py --check-dr-covers` |

---

## 11. Domain randomization and curriculum

DR ranges scale with `dr_level` in [0, 1] and are sampled per episode at reset (dynamics table in section 6; sensors in section 7; wind, link-loss hazard 0-0.2/s, battery 0.85-1.0, geometry in section 8). Additional sensor DR: vg noise/bias/scale [0.01, 0.05] / [0, 0.08] / +/-15%, h noise [0.01, 0.03] m, yaw drift [0, 0.5] deg/s, pose noise factor [0.7, 1.6], flip probability [0, 5%], dropout floor [0, 15%], focal/range scale +/-3%, pole map error [0.05, 0.2] m.

**Curriculum** (advance when lap success >= 80% over the last 2000 episodes; 20% of envs always sample an earlier level; at least 20-25% purely nominal episodes in every stage):

| Stage | Adds | rc cap (D5) |
|---|---|---|
| 0 | perfect state, nominal dynamics, spawn facing B (learns the lap) | 25-35 |
| 1 | dynamics DR, action delay, loop jitter | 25-35 |
| 2 | sensor and estimator path (FOV, dropout, noise, latency), roughly pole-facing spawn | 35 |
| 3 | full DR, wind, random spawn yaw up to 180 deg, geometry DR, link loss | 35-50 |
| 4 | raise the cap toward the highest calibrated rc; larger lap-time weight | up to calibrated max |
| R1-R4 | robustness stages (`ROBUSTNESS_PLAN.md` 3.1): off-path/actuation/connection faults (R1), tag/sensor/software faults (R2), compound + environment (R3), ADR + failure-seed replay (R4) | randomized (A-9) |

---

## 12. Training setup

**Policy:** actor MLP [256, 256, 128] ELU, tanh-bounded action mean, state-independent log-std init -0.7; input = 4 stacked frames + last 3 actions; asymmetric critic [256, 256, 128]; running mean/std normalization clipped at 10; return-based reward scaling. GRU deferred (decision gate at R3, `ROBUSTNESS_PLAN.md` 3.6).

| PPO setting | Value |
|---|---|
| Envs | 4096 (try 8192 if memory allows) |
| Horizon / batch | 32 steps / about 131k |
| Epochs / minibatches | 5 / 8 |
| gamma / lambda | 0.99 / 0.95 (the scaffold's 0.995 was tuned for 60 Hz, `sb3_ppo_cfg.yaml`) |
| LR | 3e-4, adaptive KL schedule (target 0.01) |
| Clip / entropy | 0.2 / 0.003 (floor 0.003 in R-stages) |
| Value coef / grad norm | 1.0 with value clipping / 1.0 |
| Budget | about 100-300M env steps across stages |

**Throughput gate (M0):** `scripts/bench_throughput.py` must show at least **30k end-to-end env-steps/s at 4096 envs** (expected 50-150k, unverified). If missed: CPU env with MPS learner, or nets shrink to [128, 128].

**Logging (TensorBoard):** per-component reward, DR level and stage, lap success (estimate and truth timers), median/p90 lap time, crash rate by cause, void-outage rate, tag-visible fraction, mean |delta action| and saturation fraction; evaluation every 50 iterations on a fixed 1024-seed set at DR level 1 and nominal, with `BaselinePolicy` on the same seeds. Checkpoints carry the param-yaml hash and `params_source`.

---

## 13. Evaluation, baselines, sim-to-real, staged real tests

**Baselines.** #1 `BaselinePolicy` (classic PD, `policy_integration`), run in the same sim through an adapter (scalar per-env loop for evaluation; not part of the training hot path). #2 `engine_replica`: a pure-function replica of the engine's approach/land logic (`track_rc` plus search override), used only for the calibration regression (it is an approach-and-land policy, not a racer). Hover-only control for the robustness benchmark. Comparisons use the same seeds, horizon, and budget. The deterministic fallback in flight is hover-then-land, never the PID.

**Visualization.** Top-down matplotlib (true path, estimated path, gates, FOV cone); export to `*.live.jsonl` so `core/build_flight_viz.py` and `race/time_lap.py` run on sim rollouts; Maude's `build_view.py` / `trace_view.html` for stress-sim traces.

**Export and deployment.** `export_onnx.py` folds the normalizer into the graph and exports the actor only (onnxruntime vs torch parity <= 1e-5); history stacking lives in the numpy `ObsBuilder` shared with the sim through `estimator.py`. A new `core/race_policy_runtime.py` (numpy + onnxruntime) emits `ActionSample` packets and is selected by an engine flag (e.g. `--policy onnx:<path>`), replacing the `track_rc` call at `core/tello_dual_video.py:495`. Required engine work (F1): use `Positioner`/`FilteredPositioner` with tags 1 and 2 and `config/map.yaml`, run the shared `RaceStateEstimator`, keep every engine safety layer (cap/slew, search with the racing threshold, `LAND_ON_LOSS`, height guards, takeoff/land), and exactly one RC writer per drone (CONTRACT). Recommended engine fix: deadline-paced loop instead of `stop.wait(0.05)` after the work (`tools/rc_rate_bench.py --paced`).

**Staged real-world protocol** (Maude's MS2-MS4 and RL comparison map onto S0-S3; fault trials per `ROBUSTNESS_PLAN.md` 5.4):

| Stage | Setup | Exit criteria |
|---|---|---|
| S0 shadow (`--no_fly`, carried by hand) | policy runs, outputs only logged | estimator fix rate and lost events sane; `time_lap.py` fires lap events on the carried path; loop timing; command directions correct per phase (MS3) |
| S1 tethered/held hover | outputs heavily scaled or hover only | no oscillation |
| S2 segments | A to B only at cap 25, then 40, then 60; catcher, battery >= 50%, wind < 3 m/s | no pole contact, safe landing on every abort |
| S3 full lap | calibrated cap | lap success and time compared with the matching sim cells (the sim-to-real gap metric); classic baseline demo gate of MS4 is 3 consecutive controlled laps |

Safety rules for all stages: catcher ready (`docs/RACE_RULES.md`, `README.md:91`), emergency stop as in `FLYING.md:133-139`, stop a stage if the same fault fails twice, software-side fault injection first (`core/fault_injection.py`), RL only after the classic baseline passes MS4 and the policy passes at least 18/20 held-out simulated successes with zero guard-bypass events (MS5 gate).

---

## 14. Testing strategy (`tests/`)

- **Frames:** map <-> `course_flu_v1` round trip; default fixture det = +1; yaw sign convention (rc yaw positive reduces `psi`).
- **Lap parity:** `VecLapTimer` vs `race.lap_timer.LapTimer` on thousands of random and scripted trajectories (false start, `require_turn`, radius edge cases).
- **Dynamics:** first-order step response reaches 63% at tau; deadband threshold; delay shift exact; saturation; semigroup (one dt step equals two half steps); batch independence; seed determinism; collision segment test; **parity with `stress_sim.run()` plant**.
- **Camera/visibility:** projection vs `cv2.projectPoints` with `default.yaml`; FOV edge; view-angle cutoff.
- **Sensors:** noise sigma vs range; latency exactly k steps; dropout statistics.
- **Estimator:** torch vs numpy `KalmanCV` vs scalar reference; the `_selftest` scenario (`core/positional_kalman_filter.py:178-198`) reproduced; heading anchor; parity with `FilteredPositioner`.
- **Guards:** vectorized shim vs `BaselinePolicy.step()` guard decisions on random observations (every abort reason); AST sync of engine and policy constants.
- **Contract:** `ObsBatch.to_observation()` accepted by `policy.py` validation; `ActionSample` metadata rules; ONNX parity.
- **Env:** gymnasium `env_checker` on `SingleEnv`; vector space shapes; autoreset and `final_obs`; no NaN under 1e6 random-action steps; per-component reward sums; seed determinism.
- **Regression:** suite 1 (section 9); `engine_replica` lands within `LAND_TOL_M` with the expected mode sequence on placeholders, later against real E5 logs; fault tests per `ROBUSTNESS_PLAN.md` 3.9.
- **Performance:** optional nightly steps/s check against the M0 gate.

---

## 15. Milestones

Effort: one engineer, rough. **MVP = M0 + M1 + M2** (about 1 week): a trainable vectorized sim in `course_flu_v1` with the repo's lap rules, dynamics with lag/delay/deadband/wind, perfect-state observations (stage 0), custom PPO, the dynamics-side scenarios of suite 1 (baseline, mild/strong drift, deadband) running `BaselinePolicy` and the PPO policy, the `stress_sim` plant parity test and lap-timer parity test, and a result table in which the policy beats `BaselinePolicy` on nominal lap time and on strong-wind and deadband. Not in the MVP: camera/tag sensing, estimator, DR beyond dynamics, calibration, ONNX, deployment.

| Milestone | Deliverables | Effort | Depends on | Exit criteria |
|---|---|---|---|---|
| **M0** Setup and decisions | `.venv-sim`; `scripts/bench_throughput.py`; adopt `course_flu_v1` (D1); `stress_sim` cross-check harness; logging-patch spec; sysid flight script spec; confirm D1-D5, D13 | 1 day | none | throughput gate met (>= 30k steps/s at 4096 envs) or fallback chosen; decisions confirmed |
| **M1** Core sim | `frames`, `course`, `lap`, `contract`, `dynamics` (with action delay), `env` (perfect-state obs), placeholder params, tests, `BaselinePolicy` adapter, suite 1 dynamics-side | 2-3 days | M0 | tests pass; parity with `stress_sim` and `race.lap_timer`; suite-1 baseline results reproduce (20/20, 20/20, 0/20, 6/20) within tolerance |
| **M2** PPO and MVP eval | `train/ppo.py`, stage-0 training, eval harness, matplotlib viz | 2 days | M1 | MVP criterion above |
| **M3** Sensing and robustness base | `camera`, `sensors`, `estimator`, `safety` (profiles, D4), privileged critic, DR, curriculum stages 1-4, suite 1 sensor-side scenarios; guard-profile decision implemented; RB0/RB1 (fault skeleton, P/A/C-1..3/SW injectors, stage R1, benchmark v0) | 3-4 days (+ RB0-RB1: about 5-6 days) | M2 | actor sees only realizable inputs; shim in the loop; lap success >= 80% at stage 3 |
| **M4** Calibration (parallel, needs real flights) | logging patch, E0-E6 flights, `fit_dynamics`/`fit_sensors`, `validate`, `tello_calibrated.yaml` | 2 days (flights excluded) | Linux laptop, MS2 prerequisites | validation thresholds met (section 10); DR covers fitted values |
| **M5** Deployment | ONNX + manifest, `core/race_policy_runtime.py`, engine integration (tags 1/2, `Positioner`, `RaceStateEstimator`), shadow mode (MS3), contract v1.1 | 2-3 days | M3, M4 | S0 shadow exit criteria met |
| **M6** Real flights | S1-S3, MS4/MS5 gates, sim-to-real gap log | ongoing | M5 | sim vs real within tolerance on the matching cells |
| RB2-RB5 | tag/sensor injectors + indicators + stage R2; compound/ADR/failure buffer/R3-R4 and full benchmark; engine fault shim + real fault trials R-S0..R-S3; severity recalibration | about 3 weeks after the MVP (P0 scenarios about 10 working days) | M3 (RB2-RB3), M5/M4 (RB4-RB5) | thresholds in `ROBUSTNESS_PLAN.md` 4.4 |

Order change from the first draft: the contract, frame sign check and tag 1/2 unification (F1, MS2) come **before** RL training of a deployable policy; sim training itself starts at M2 without waiting for them. Dropped: `race_pid`, the axis-swap `frames.py`, `gym-pybullet-drones`. Total: about 2.5-3.5 weeks to a deploy-ready policy plus the robustness layer.

**What to cut for the MVP:** GRU, wind/pitch-lean extras beyond the drift scenarios, detection LUT (use FOV + range gate + Bernoulli first), automatic curriculum (manual stages), ONNX, SB3 cross-check, rich visualization.

---

## 16. Risks

| Risk | Mitigation |
|---|---|
| Pose-age guard vs visibility: the `classic` guard lands after 0.75 s without a fresh pose; a forward-facing 55 deg camera loses tags for seconds on any racing route | `rl` guard profile with estimator bridging (D4); needs the contract v1.1 agreement; otherwise a route that always keeps a tag in view |
| RL action cap: `rc_limit = 20` is about 0.16 m/s, far below racing speed | separate RL cap schedule (D5); cap limited to the highest calibrated rc |
| Std limits (0.15 m / 0.25 m/s) are exceeded by a bridged estimate | profile-scaled limits (D4); check against the noise model at M3 |
| Noise scale: `stress_sim`'s 3 mm noise is unrealistic | range-dependent model (D15); real E4 data to confirm |
| Fixed-heading route vs a yaw-facing policy; unproven backward return leg (F16) | heading is a policy choice (D13); demonstrate tag coverage or redesign with sensing before using the example route |
| Example map makes tag B invisible (F4) | geometry DR including tag normals; re-map on site |
| Extrapolating dynamics above the calibrated rc | cap = highest calibrated rc |
| Link drops from motor EMI make real flights unusable (F13) | measure outage rate early with `tools/tello_hover_test.py`; hardware fixes (USB extension, external antenna, EDU) before racing |
| Tello optical-flow velocity outdoors worse than modeled; no ground truth | tag-derived cross-check; tape grid/phone video |
| Video latency and detector throughput on the race laptop | measured at E4/E6; DR ranges cover them |
| Engine integration larger than it looks (F1, F8) | MS2 items scheduled before deployment; M5 effort includes it |
| Over-conservative policy / fault space too large | nominal share >= 20-25% per stage; P0 first (`ROBUSTNESS_PLAN.md` 3.7, 6.5) |
| MPS missing ops or numerical differences | CPU fallback; float32 only; `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Contract drift between Maude's code and the sim | manifest + parity tests; contract changes reviewed before merge |
| `pupil-apriltags` arm64 wheel | only needed for `[calib]`; fall back to the repo's detector in `cal_and_map/` *(unverified)* |

---

## 17. Open questions for the user

1. **Real course:** pole spacing, tag height, tag facing direction, pole thickness, field boundaries, typical wind? (Decides whether the task is visible, F4.)
2. **Hardware:** standard or EDU Tello (EDU accepts the video-shrink commands, `FLYING.md:123-131`)? Is the Linux laptop with the dongles available for real flights and for E0-E6?
3. **Ground truth:** motion capture, or only tape grid / phone video?
4. **Speed limits:** maximum rc / speed acceptable for real flights?
5. **Transform ownership:** who measures `R_world_map` / `t_world_map`, and by when (blocks MS2, M5)?
6. **Route:** is the fixed-heading, backward-return example route the real race route?
7. **Deadline** of the hack (decides the MVP cut)?
8. **Action_Space repo access:** can we read `action_space_kit` (tag sensor, Kalman, `fit_tello_motion.py`, `TelloVelocityController`) to reuse identified parameters?
9. From `ROBUSTNESS_PLAN.md` 6.6 (real-world dependent): Tello behaviour when rc stops mid-motion; packet units and rate; which faults can be reproduced safely at the venue; whether a void lap is acceptable on race day.

---

## 18. Prioritized next steps (first week)

1. [ ] Confirm decisions D1-D5, D10, D12, D13 (section 3); agree the contract v1.1 amendment (D4) with Maude.
2. [ ] M0: create `.venv-sim`, run `scripts/bench_throughput.py` on the M4 (steps/s at 4096 envs on MPS and CPU).
3. [ ] Write `frames.py` with the default fixture and sign tests; adopt `course_flu_v1` in `course.py`.
4. [ ] Port the lap timer (`lap.py`) with the parity test against `race/lap_timer.py`.
5. [ ] Implement `dynamics.py` (lag, delay, deadband, wind) with the parity test against `stress_sim.run()`.
6. [ ] Build `env.py` (perfect-state obs, SAME_STEP autoreset) and the `BaselinePolicy` adapter; reproduce suite-1 baseline results.
7. [ ] Write `train/ppo.py` and train stage 0; compare with `BaselinePolicy` (MVP table).
8. [ ] In parallel (needs the Linux laptop): apply the engine logging patch and record E0/E1 and a short hover link test (`tools/tello_hover_test.py`) to get the real outage rate and first calibration data.
9. [ ] Start the P0 fault catalogue freeze (`params/faults.yaml`, `ROBUSTNESS_PLAN.md` section 7) once M1 is green.
10. [ ] Answer open questions 1, 2, 5 and 6 (they gate sensing realism, the real-flight plan, and the deployment path).
