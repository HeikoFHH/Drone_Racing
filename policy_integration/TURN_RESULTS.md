# Turn and sensing review: results for teammate review

Recommendation: continue simulation and grounded/shadow integration. Physical
flight readiness is unproven. A turn controller alone does not solve localization.

## Changes implemented

- Preserve fixed +X heading/backward return as the default. Add explicit per-leg
  landmark heading targets as an opt-in review candidate, not an agreed deployment
  maneuver or replacement simulator.
- When fresh heading error is large and horizontal speed exceeds 0.05 m/s,
  oppose measured velocity before allowing yaw. Discard old horizontal body-frame
  demand at brake entry. This threshold is synthetic, not a measured stop bound.
- Preserve horizontal command direction by scaling the right/forward pair
  together to RC magnitude 20. Interpolate both channels with the same slew
  fraction. Integer rounding can increase magnitude by at most sqrt(0.5).
- Require heading alignment as well as position, speed and dwell for waypoint
  advancement. Existing freshness, model identity and latched abort guards remain.

## Reproduced comparison

960 attempts: 16 scenarios × 20 fixed seeds × three controllers (previous reviewed
policy at `d53e2d460ad130ed66c756e9f0ab37f26143e1f8`, current fixed heading, and
current landmark candidate). Raw per-attempt JSONL and policy hashes accompany
`reports/turn_sim_summary.json`. These seeds are development seeds, not a held-out
validation set. No claim of statistically demonstrated improvement is made.

| Scenario family | Each previous/current fixed controller | Landmark candidate |
|---|---|---|
| 12 ideal-pose scenarios: nominal, signed 30/60/90°, 180°, momentum, pose/command delay, slower response | 20/20 complete per scenario | 20/20 complete per scenario |
| Two-tag camera, no injected fault | 0/20 complete; B gate reached 0/20 | 0/20 complete; B gate reached 20/20 |
| Two-tag camera with yaw, with yaw + 150 ms pose delay | 0/20 complete in each | 0/20 complete in each |
| Ideal pose with turn + one-second blackout | 0/20 complete | 0/20 complete |

All failed cells above end in controller aborts. No true geofence/clearance
violations occurred in these runs. An abort ends the trace at an advisory landing
request: it does not prove subsequent stopping, landing or touchdown. The simple
truth gate check tracks ordered A exit, B entry and A return; it is not the complete
production race timer. This comparison does not establish lap-time equivalence.

The camera uses the repository example calibration, not a measured camera on the
actual drone. A pinhole point-tag calculation puts B tag center outside the image
at X≈3.066 m and gate entry at X≈3.376 m on the default route. Tag facing, corners,
blur, roll/pitch and extra tags are omitted: even this optimistic sensing model
cannot complete a lap. The landmark candidate reaches B, then loses localization
while turning. Do not extend blind-flight timeouts to hide this result.

Reproduce (the reference commit must exist in the local Git checkout):

```bash
python3 -m policy_integration.turn_sim --seeds 20 --reference-ref d53e2d460ad130ed66c756e9f0ab37f26143e1f8 --output policy_integration/reports/turn_sim_summary.json --run-log policy_integration/reports/turn_sim_runs.jsonl
```

## Inputs needed from the team

1. Sensing owner: actual calibration, camera/body transform, mapped tag poses and
   facing, and capture-time logs proving coverage through the whole maneuver.
2. Simulator owner: measured delays, braking/RC response and deadband; a dynamics
   and detector test that includes tilt, full tag corners and loss/reacquisition.
3. RL owner: checkpoint identity, ordered features, normalization and action
   semantics; compare on the same observations, route and episode budget.
4. Runner owner: shadow adapter, single RC writer, independent watchdog and
   separate landing request/acknowledgment/touchdown states.

Physical stopping margins, bank angles, deployment adapter and race-timer
integration remain unverified. The original stress suite's strong drift and
deadband mission failures also remain part of the evidence.
