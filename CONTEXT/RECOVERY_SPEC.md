# Recovery layer spec (for implementation; human version: docs/RECOVERY_PROPOSAL.md)

Status: specification of a proposal; nothing here is implemented except `SensorModel.inject_blackout` and
`SensorFaultController` (tag blackout injection used for the baseline measurements).

## Placement
`RecoverySupervisor(primary: Controller, guard_cfg)` wraps a controller like `GuardedController`. Authority per tick:
guard limits (geofence, speed, link, battery -> LANDING) > recovery states > primary. `GuardedController` stays the outer wrapper;
its pose-age limits (`pose_hold_s`, `pose_land_s`) must be suspended while `recovery.active` and bounded by `recovery.budget_s` (20 s).
Reads only `env.sensed`. Returns the same `(rc, infos)`; infos carry `mode` in {nominal, coast, brake, reacquire, relocalize, resume,
retrace, hold_land} and `reason`.

## States and transitions (defaults; all in `RecoveryCfg`, placeholders until calibrated)
| From | To | Condition |
|---|---|---|
| NOMINAL | COAST | `fix_age > 0.4` or health == DEGRADED |
| COAST | NOMINAL | fix returns and 3 consistent fixes within 0.5 s |
| COAST | BRAKE | `fix_age > 1.0` or health == BAD |
| BRAKE | REACQUIRE | measured speed < 0.05 m/s, or 1.5 s elapsed |
| REACQUIRE | RELOCALIZE | a tag detected |
| RELOCALIZE | RESUME | 3 fixes within 0.3 m of each other, `pos_std < 0.25`; if mismatch with dead reckoning > 0.5 m -> log jump, reset primary history |
| RELOCALIZE/REACQUIRE | RETRACE | REACQUIRE budget 8 s exhausted |
| RETRACE | REACQUIRE | arrived at last good pose (breadcrumb: last trusted pose, `fix_age < 0.2`) |
| any recovery state | HOLD_LAND | total budget 20 s or retrace budget 6 s exhausted |
| RESUME | NOMINAL | speed ramp 0.2 -> 1.0 over 2 s done; waypoint chosen from lap phase |
Hysteresis: minimum dwell 0.5 s per state; never re-enter COAST within 1 s of RESUME without a new fault.

## Actions
- BRAKE: rc from `-kd * v_measured` in body axes (as `policy_integration/policy.py` brake logic), up held, yaw 0.
- REACQUIRE: yaw toward predicted tag bearing from `est pos`, `est yaw` and `SensorModel.tag_pos`; sweep +/-60 deg, 25 rc, 0.5 s dwell.
- Speed governor (applies in COAST/RESUME/NOMINAL for any primary): `v_max = sqrt(2*a_brake*max(0, d_clear - 0.1 - 2*pos_std))`
  x `clamp(1 - fix_age/3, 0.2, 1)`; `a_brake` 0.5 m/s^2 placeholder; clip the primary's horizontal rc to `v_max / gain_xy * 100`.
- Health monitor `HealthMonitor.update(env.sensed, dt) -> (OK|DEGRADED|BAD, reasons)`; inputs listed in docs/RECOVERY_PROPOSAL.md 3.4.

## Files to create
`drone_race_sim/recovery.py` (supervisor, `RecoveryCfg`, `HealthMonitor`, breadcrumbs), `drone_race_sim/faults.py` (kicks, frozen frame,
latency spike, outliers; reuse `inject_blackout`), `tests/test_sim_recovery.py`, report cells in `scripts/make_report.py`.

## Tests to write first (map to ROBUSTNESS_PLAN IDs)
T-1 blackout 0.5/1/2/4/8 s for race_policy, ppo_realistic, guarded variants: 0 crashes, >= 90 % laps up to 4 s (baseline in
`outputs/sim/blackout_baseline.json`: ppo_realistic 12/30 pole crashes at 4 s; guard lands everything >= 2 s). T-3 never visible -> land <= 20 s.
P-2/P-3 kicks. T-12 outliers. Transition unit tests with scripted `env.sensed` sequences (no sim needed). Guard coordination test:
recovery active -> no landing before budget; budget exceeded -> landing.

## Do not
- Do not lengthen blind-flight timeouts to hide failures (Maude TURN_RESULTS.md).
- Do not refresh a missing fix's timestamp when talking to Maude's policy (CONTRACT.md); use bridged mode only under contract v1.1.
- Do not fault-inject an airborne real drone from software before shadow mode passes.
