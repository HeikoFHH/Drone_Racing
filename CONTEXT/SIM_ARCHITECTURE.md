# Simulator architecture (`drone_race_sim/`)

Vectorized torch simulator of the two-pole Tello race. All state tensors have leading dim E (envs). Runs on cpu or mps.
Default dtype float32; float64 only on cpu (parity tests).

## Data flow per control tick (`TelloRaceVecEnv._step`)
```
action in [-1,1] --(x rc_cap, round)--> rc ints
rc --TelloDynamics.step--> pos, vel(world), yaw        # packet loss -> delay -> deadband -> gain -> lag -> integrate
truth VecLapTimer.update(pos)                          # reward/termination/eval only
if sensing == "realistic": SensorModel.step(...)       # tag fixes -> KalmanCV -> bridged estimate; packet vel/yaw/height
                           lap_est.update(bridge_pos)  # estimate-driven timer (what flight would report; plan D12)
rewards (truth) -> terminations -> obs (estimate or truth) -> same-step autoreset (info["final_obs"])
```

## Modules
| Module | Purpose | Key API |
|---|---|---|
| `config.py` | `EnvCfg`, `DynamicsParams` (+ `stress_sim_preset()`), loaders, `params_hash` | `load_dynamics()` |
| `frames.py`, `course.py` | map<->course conversion; poles from `config/map.yaml` | `load_course_geometry()`, `map_poles_for_lap_timer()` |
| `lap.py` | `VecLapTimer` (parity with `race.lap_timer.LapTimer`) | `update(t, xy, mask)` -> event masks |
| `dynamics.py` | rc -> velocity model, collision geometry | `TelloDynamics.step(rc, dt, gen)` |
| `randomization.py` | per-episode DR of dynamics (`dr_level` in [0,1]) | `DynamicsSampler.sample` |
| `sensors.py` | FOV/range/view-angle visibility, noise vs range, latency, KalmanCV, state packet, bridged estimate | `SensorModel`, `load_sensor_cfg()` |
| `env.py` | gymnasium `VectorEnv`, 112-dim actor obs, 16-dim critic extras, `env.sensed` | `reset(seed, options)`, `step(a)`, `step_rc(rc)` |
| `wrappers.py` | `SingleEnv` (gym.Env, `gym.make("DroneRace-v0")`), `NumpyVecWrapper`, `SB3VecEnv.create(env)` | |
| `controllers.py` | PPO / Maude / Race controllers, `GuardedController`, `FaultyController` | `act(env, obs, now) -> (rc, infos)` |
| `rollout.py` | closed-loop runner with recording; outcomes `lap` (truth timer), `lap_est_only`, `landed`, `crash:<cause>`, `timeout` | `rollout(controller, scenario, seeds, sensing=, record=)` |
| `viz/plots.py`, `viz/replay_html.py` | matplotlib figures, self-contained interactive HTML replay | |
| `baselines/*_adapter.py` | older per-policy loops (kept; tests rely on `baseline_policy_adapter`) | |
| `../train/` | custom PPO (`ppo.py`), CLI (`train.py`), evaluation (`eval.py`) | |

## Observation layout (actor frame = 25 dims, stacked x4 + last 3 actions = 112)
rel Pole A (fwd,left)/4 [2], rel Pole B [2], height-1.0 [1], body velocity (fwd,left,up) [3], yaw rate [1],
tag A (seen,u,v,log range) [4], tag B [4], fix_age/2 [1], lost [1], pos_std [1], lap phase one-hot (IDLE,AT_START,OUTBOUND,TURNED) [4], dt/0.05 [1].
Perfect mode: truth, tag/estimator slots are zero. Realistic mode: all from the estimate/packet/detections.
Critic extras (16): truth position, yaw sin/cos, world velocity, wind, lap elapsed, normalized dynamics params. Always truth.

## `env.sensed` (what controllers may use; never read `env.dyn` truth in a controller)
keys: `pos` (bridged estimate, KF while tracked else dead-reckoned), `kf_pos` (what FilteredPositioner reports),
`vel_w`, `yaw`, `lost`, `fix_age`, `fix_now`, `seen` (E,2), `tag_img`, `pos_std`, `fix_pos` (last raw fix),
`est_at_fix` + `fix_time` (held filtered pose with ORIGINAL capture time, for Maude's contract).
In perfect mode the same keys return truth.

## Sensor model facts (placeholders in `params/sensors_placeholder.yaml`)
HFOV 55.4 / VFOV 43.0 deg from `config/camera/default.yaml`; tag side 0.13208 m; min apparent side 12 px (range <= ~10 m at
frontal view); max tag viewing angle 70 deg; tag A faces B and tag B faces A (tags are one-sided); tag centre height 1.0 m;
video delay 0.15 s (3 ticks, exact); detector 12 Hz; fix noise sigma_depth = 0.0058 r^2, sigma_lateral = 0.015 r;
per-episode range scale error N(0, 3%); KalmanCV from `core/positional_kalman_filter.py` (torch backend) with
`lost_after` 0.4 s. At reset the filter is seeded with a noisy fix (drone armed inside the A gate).

## How to add things
- **Controller**: subclass `Controller`, implement `act`; read only `env.sensed`; return rc tensor + per-env info dicts with
  `mode` and `land`. Register in `scripts/make_report.py::controllers`.
- **Scenario**: stress scenarios come from `policy_integration.stress_sim.SCENARIOS` (id -> wind_y, deadband, delay, pose/link
  loss). Dynamics-side fields (wind, deadband) are applied via `reset(options=...)`; sensor-side faults (delay/pose-loss/
  link-loss) are only implemented in `baselines/baseline_policy_adapter.py`, not yet in `SensorModel` (see KNOWN_ISSUES).
- **Sensor effect**: add to `SensorModel.step`, expose through `env.sensed`, add a unit test in `tests/test_sim_sensors.py`.
- **Reward term**: `rewards.py` + weight in `params/reward.yaml`; every term is logged by `PPO.iterate`.
