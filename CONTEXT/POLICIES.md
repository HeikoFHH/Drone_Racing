# Policies and the guard

| Controller | Source | Frame / inputs | Cap | Notes |
|---|---|---|---|---|
| `maude_baseline` | `policy_integration/policy.py::BaselinePolicy` (Maude) | course_flu_v1 `Observation` (CONTRACT.md) | +/-20 rc, slew 60 rc/s | PD route A-offset -> B-offset -> A-offset at fixed heading +X; full guard envelope; aborts latch |
| `race_policy` | `race/race_policy.py::RacePolicy` (Tvisha) | map frame `Obs` as `FilteredPositioner` gives it (pos None while lost, raw fix with camera->map R) | stick cap `PolicyCfg.max_rc`, cruise 0.8 m/s | gate racer: nose on B, pure-pursuit along A-B; searches (yaw sweep) when lost; flies `look_at="B"` so the return leg is backward |
| `ppo_perfect` | `train/` (this repo) | 112-dim obs from truth | rc_cap 35 | trained without sensing; only meaningful as a control upper bound / sim-to-real gap probe |
| `ppo_realistic` | `train/ --sensing realistic` | 112-dim obs from the estimator | rc_cap 35 | trained on camera/tag/estimator observations |

## Maude's policy as the backup for every policy (`GuardedController`)
Owner requirement: Maude's policy survives the merge and is the backup for all policies. Implemented as a supervisor, in this
order per env and tick (`drone_race_sim/controllers.py`):
1. `LANDING` is latched (Maude: aborts latch): zero sticks, `land` requested.
2. Guard limits from Maude's `PolicyConfig`: estimated position outside the geofence (x -1..5, y -1..1.5, z 0.4..2.0) or measured
   speed > 1.5 m/s -> LANDING. No tag fix for `pose_land_s` -> LANDING; for `pose_hold_s` -> hover (zero sticks).
   Profiles: `classic` = Maude's 0.25 s / 0.75 s; `rl` = 2.5 s / 4.0 s (plan D4; needs contract v1.1 from Maude).
3. Primary output non-finite or beyond +/-100, no lap-phase progress for `stall_s` (20 s), or estimated distance to a pole centre
   below Maude's `pole_clearance_m` (0.3 m) -> BACKUP: a fresh `BaselinePolicy` flies Maude's PD route starting at the waypoint
   that matches the lap phase (A=0, B=1, return=2), with all of its own guards; it lands when it aborts or completes.
The guard never edits the primary's output while healthy (tested: identical lap times). It consumes only `env.sensed`.
Backup under `rl` profile uses `maude_observation(bridged=True)` (estimator dead reckoning stamped now, growing std) and
`PolicyConfig` relaxed to position std 0.5 m, velocity std 0.5 m/s, pose ages 2.5 / 4.0 s.

## Observation hand-off to Maude (CONTRACT.md rules the sim enforces)
Realistic mode default: held filtered pose `est_at_fix` with its ORIGINAL capture time `fix_time = t_fix - video_delay`; never
refreshed between fixes (else her `pose_changed_without_new_frame` abort fires). Bridged mode deliberately violates this (D4).
pose/position std from the KF (`pos_std`), velocity std 0.1.
