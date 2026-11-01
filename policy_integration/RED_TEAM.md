# Red-team plan and evidence

Purpose: try to break the sensing -> policy -> runner contract before judging a
lap or a training improvement. Tests use injected software faults; hardware
checks are clearly marked pending. All limits are prototype values.

| Attack/failure | Required behavior | Evidence/status |
|---|---|---|
| Right/forward or yaw sign reversed | Unit fixtures establish canonical signs; grounded measured motion confirms them | Automated signs pass; real signs pending |
| Tag-map Y-up vs world Z-up | Apply proper rigid transform; transform back before existing lap timer | Adapter/independent timer test passes; measured transform pending |
| Units or tag dimensions wrong | Reject bad frame schema; verify scale against measured tag and distance | Frame test passes; physical scale pending |
| Reflected/scaled pose transform | Reject nonorthonormal matrix or determinant -1 | Automated pass |
| Missing pose briefly | Zero RC after 0.25 s freshness limit; no waypoint advancement | Automated pass |
| Missing pose persistently | Advisory landing after 0.75 s; latch abort | Automated pass; runner delivery pending |
| Frozen video with repeated timestamp | Cannot prove arrival dwell | Automated pass |
| Pose changes under unchanged timestamp | Abort; do not treat re-labelled data as a new fix | Automated pass |
| Large pose jump/tag swap | Abort physically implausible displacement | Automated pass; false-tag cases pending |
| Future/out-of-order timestamps or clock jump | Abort instead of using negative age/dt | Automated pass |
| Old telemetry or Wi-Fi loss | Abort, zero proposed RC, request land; never auto-resume | Automated pass; actual transport/reconnect pending |
| Low battery | Abort and request landing | Injected pass; no intentional low-battery flight |
| NaN/Inf, wrong vector length, missing fields | Reject before generating motion | Automated pass |
| Unknown/wrong observation schema | Reject | Automated pass |
| RL emits invalid range/order/frame, old action, old source pose, missing output | Abort in learned mode | Automated pass |
| Inference hangs | Flight tick consumes only a precomputed packet; stale/missing packet aborts | Packet checks pass; worker implementation pending |
| RL emits extreme but valid normalized actions | Same cap, slew, heading and sensor guards as classic | 1,000 deterministic random action packets checked |
| Starts grounded/disabled | No takeoff or movement produced | Automated pass |
| Reconnect after abort | Abort remains latched; no autonomous new session | Software latch pass; runner pending |
| Still moving inside waypoint | Cannot advance until low speed and dwell | Automated pass |
| Physical pole on a route segment | Reject known route/segment crossing clearance disk | Automated pass; measured footprint/clearance pending |
| Outside height/course limits or excessive speed | Abort and request landing | Automated pass; stopping distance is NOT guaranteed |
| Model or script tries to be second RC writer | Runner enforces one writer; shadow cannot dispatch | Runner implementation pending |
| Return leg loses tags | Hold then land request; cannot declare success | Pose-loss software pass; physical visibility pending |
| Controller completes route without valid lap | Existing timer must independently confirm A-exit/B-entry/A-return | Integrated toy trace confirms timer after transform |
| Wind, glare, blur, radio interference, RC deadband | Characterize in controlled physical tests and simulator randomization | Not simulated here; pending |
| Reward hacking/phase exploitation | Count valid laps separately from cumulative training reward | Proposed RL evaluation; not run |

## Limits of the guard

The policy is a prototype, not a certified safety controller. It cannot prevent
all collisions or guarantee a stop before a boundary. It checks known pole
geometry and current observations, not arbitrary obstacles, future wind, or a
full vehicle braking model. Zero RC is not an emergency motor stop. `request_land`
is not an acknowledged land. The physical runner/operator retain lifecycle control.

## Red-team session sequence

1. Run the automated suite and inspect failures and raw audit findings.
2. Replay an ACTUAL canonical sensor log; inspect signs, freshness and uncertainty.
3. On grounded hardware, cover tags, inject stale timestamps, and mock transport
   loss; confirm the runner has one RC writer and a latched landing workflow.
4. In shadow mode, hand-carry the drone through A/B/A and near the agreed limits.
   Verify proposal direction and independent timer behavior.
5. Only after handoffs are connected and checked, the physical operator performs
   short controlled tests, followed by the measured lap. Record all interventions.
6. Compare classic and RL on equal held-out scenarios and report every failure.

See reports/verification.json and reports/toy_bench.json for this run's software
results. Those reports do not certify hardware readiness.
