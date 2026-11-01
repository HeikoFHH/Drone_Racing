# Known issues and findings (verified in this repo; update when fixed)

1. **Maude's example route cannot be flown on camera sensing alone.** With realistic sensing her fixed-heading route (waypoints 0.5 m
   off the A-B line) loses tag B about 1 m before the B waypoint (tag leaves the 55 deg FOV at ~28 deg) and the policy lands
   (`sensors_timed_out`); with relaxed pose ages and bridging it lands on `estimator_uncertain` or can hit the pole by drift.
   Perfect sensing: completes laps (~42 s median at rc 20). Needs a route/heading redesign with sensing, or bridging + contract v1.1.
2. **Maude's `max_position_std_m = 0.15` is stricter than `KalmanCV` can report** (meas_sigma 0.15 gives std >= ~0.12 and grows).
   `classic` guards abort immediately with `estimator_uncertain` under realistic sensing.
3. **`RacePolicy` passes 0.32-0.39 m from a pole centre at the start** (pole crash radius 0.25 m, Maude clearance 0.3 m).
   Over 100 seeds the closest approach is 0.320 m: no crash, small margin. Guard `pole_takeover_m` therefore equals 0.30, not more.
4. **Sensor-side stress scenarios (camera delay, pose loss, link loss) are not in `SensorModel`.** They exist only inside
   `baselines/baseline_policy_adapter.py` (stress_sim port). Realistic sensing has its own fixed latency (0.15 s).
5. **Not modelled**: rendered images/blur, pose-flip outliers, dropout bursts, pole map error, packet rate/quantization, camera
   pitch (0 deg), wind on the sensor path, collisions beyond a hit flag, landing dynamics. Dynamics lag is applied in the world
   frame (matches `stress_sim`; physically it is body frame).
6. **Everything is uncalibrated**: `recordings/` is empty; real rc-100 speed may exceed 0.8 m/s; packet units unverified.
7. **Episode length** is 45 s in training (plan said 30 s; at rc 35 a lap needs ~25 s).
8. **Tvisha's `race/sim_race.py`** (renders real tag images, runs the real detector) uses its own kinematics (0.35 s lag) and a
   different course map (B at z=+spacing, tag 2 flipped). Use it for perception checks; it is not the training sim.
   Observed: 3 m/s wind + 1 m/s gust blows the drone ~42 m off (policy never recovers); an 8 m course at 1.0 m/s cruise does not
   finish within 40 s.
9. `race/fly_race.py` (real-flight runner) takes off by itself and bypasses Maude's guards and one-RC-writer rule; it is merged but
   was not run and must not be used without a catcher and the staged protocol in `docs/SIMULATION_PLAN.md` section 13.
10. `env.sensed` in perfect mode returns truth; `info["priv"]` critic features are truth in both modes (no estimator error feature yet).

## Added after the full test matrix (docs/SIM_TEST_REPORT.md)
11. **`ppo_realistic` flies blind on purpose.** It yaws about 65 deg away from the tags and dead-reckons for up to 6.3 s (max `fix_age`
    6.3 s). Dead reckoning is cheap in the sim (packet velocity bias 0.03 m/s, std growth 0.05 m/s). The rl guard profile lands every
    such run (`pose_land_s` 4 s). Fix before trusting the policy: reward for tag visibility (plan section 8: -0.05 per step unseen > 0.5 s),
    wider odometry drift DR, then retrain. Result tables for it are therefore an upper bound.
12. **Estimate timer vs truth timer.** With realistic sensing `lap_est` (estimate-driven, plan D12 "official") and `lap` (truth) disagree: mean 0.1 s,
    max 1.2 s lap time, and the estimate can finish up to ~0.2 m early. `rollout` reports outcome `lap_est_only` when only the estimate timer
    finished within a 3 s grace; `lap` means the true timer finished. Do not call `lap_est_only` a clean lap.
13. **`ppo_perfect` is not a deployable policy.** Under realistic sensing it almost never laps (obs slots it never saw are now filled).
14. **Guard false positives**: `GuardedController` (rl profile) landed 2-3 of 30 healthy `race_policy` runs (no fix > 4 s). Tunable via `GuardCfg`.
15. `env.lap_est` is the truth position in perfect mode (kept in sync so reports work in both modes).
16. **Tag-blackout baseline (outputs/sim/blackout_baseline.json, realistic sensing, 30 seeds, blackout at t=6 s)**: `race_policy` recovers from every length tried
    (0.5-8 s); `ppo_realistic` hits poles in 12/30 runs at 4 s; `GuardedController` (rl profile) lands every run with a blackout >= 2 s, cancelling the racer's own recovery.
    See docs/RECOVERY_PROPOSAL.md. Injection hook: `SensorModel.inject_blackout(duration_s)`, `SensorFaultController(inner, duration_s, t_start)`.
17. **Maude's branch has moved** (turn controller, landmark heading, turn_sim, dashboard, offline CI workflow `.github/workflows/policy-offline.yml`); not merged. Her `policy.py`
    diff is backward compatible (new fields default to the old behaviour). `connect-trial` branch (tello_basic.py, real-link SDK tool) not merged.
