# sign_mission: fly to ONE sign, stop 0.5 m in front, land (and a turn-and-return variant)

Human guide: `docs/SIGN_MISSION.md`. Do not confuse with `race/homing_policy.py` (Tvisha, 73 m, two tags, 2-3 m stop): different code, different course.

## Rules
- **One sign only.** Never add waiting for, searching for, or navigating to a second tag. After the 180 deg turn the return leg is dead reckoning (measured speed, else learned stick scale).
- Sign width is 22.5 in = 0.5715 m = edge of the BLACK square (`spec.SIGN_WIDTH_M`). Stop distance 0.5 m is camera -> sign plane along the viewing axis (`spec.STOP_DISTANCE_M`).
- Only `MissionRunner` writes rc sticks (one writer). Policies are pure `step(RawObs) -> Action`; they own their `SignRangeEstimator`.
- Never use the Tello `emergency` command from code. Stop = stop + land policy.
- Telemetry conventions: yaw + = clockwise (same as rc yaw); height m above take-off; `vel_body_mps` (forward, right) is `None` unless trusted (`TelloDrone(trust_velocity=True)`); policies fall back to the commanded-speed model + learned `scale`.

## Map
| File | Role |
|---|---|
| `spec.py` | sign + stop constants, `CameraModel` (config/camera/default.yaml), `min_range_fully_visible` (0.744 m for 22.5 in) |
| `common.py` | `SignObs`, `RangeEstimate`, `RawObs`, `Action` |
| `range_est.py` | `observation_from_corners`, `SignRangeEstimator` (Kalman on depth, closing-speed input, near-field dead reckoning) |
| `policies.py` | `ApproachPolicy`, `OutAndBackPolicy`, `PolicyCfg`, `POLICIES` |
| `landing.py` | `LandingPolicy` (BRAKE, SETTLE, DESCEND, LAND command) |
| `runner.py` | `MissionRunner` (PREFLIGHT, TAKEOFF, FLYING, LANDING; start/stop/status; shadow mode; stick cap) |
| `drone.py` | `Drone` interface, `RealClock`, `VirtualClock`, `Telemetry`, `FrameResult` |
| `sim.py` | `SimDrone`, `Scenario`, `SignRenderer` (analytic or rendered+real detector perception) |
| `tello.py` | `TelloDrone` (SDK UDP, state, H.264 video, detection); tested only against a fake UDP drone |
| `perception.py` | `SignDetector` (pupil_apriltags like the repo's TagDetector, aruco fallback), `annotate` |
| `server.py`, `web/index.html` | control interface (MJPEG video, /api/status|start|stop|sim) |

## Policy states
approach: SETTLE, SEARCH, ALIGN, APPROACH, FINAL, STOP, DONE (ABORT). out_and_back adds TURN, RETURN, STOP2. Mode `done` -> runner lands; `abort` -> runner lands (reason in `info["abort_reason"]`).
Control details: heading held on a reference that follows the sign only beyond 8 deg bearing; sideways stick = PI on the sign's lateral offset (absorbs a steady cross-wind); stop = forward braking + range hold (P+I on `est.range_m - stop_m`) + re-approach if drifted > 0.2 m; speed scale learned in `_Calib`.

## Testing
`.venv-sim/bin/python -m pytest tests/test_sign_mission.py` (35 tests, ~10 s): estimator, 14 approach scenarios, policy 2, stop-button, landing, safety nets, shadow mode, TelloDrone vs fake UDP drone, server. Scope is normal conditions and light wind only (owner decision); do not add extreme-condition tests.
Sim missions run in virtual time: `clk = VirtualClock(); MissionRunner(SimDrone(Scenario(...), clock=clk)).start("approach")`; the result carries `result["truth_at_end"]` (sim truth when the policy ended).

## Known limits (see docs/SIGN_MISSION.md)
Not flown. 0.5 m/s wind toward the sign still reaches the sign during landing in the sim; policy 2 return drifts in cross-wind (no reference behind the drone); the sim drone has no position hold (pessimistic). Estimator: ~1 % common pose bias at 1 px corner noise; fusing the three cues does not beat the pose solution per frame, only the Kalman filter helps.
Static demo website (Start/Stop replay of recorded sims) was deliberately NOT built yet (owner: hardware control first).
