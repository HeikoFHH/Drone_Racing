# Turnaround and robustness: execution and teammate handoff

Owner: Maude, policy and integration. Branch: `maude/policy-integration`.
Baseline reviewed: `ea711e996e1f692e58b3d77f0f31f38c751269a2`.
Status: execution plan; the new turn/visibility tests and maneuver changes below
are not implemented. Existing evidence is 37 software tests and 160 seeded runs
of the simplified velocity plant. No physical-flight readiness is claimed.

## The package we will give our teammate

1. A short findings table, with each finding labelled reproduced, inferred, or untested.
2. A reviewed policy change on this branch, without independently rewriting his simulator.
3. Reproducible turn/fault scenarios, saved seeds and before/after results.
4. An agreed sensor/action contract and a list of remaining physical measurements.
5. A clear recommendation: continue development, enter shadow testing, or block flight.

## 1. Establish what is wrong before changing the controller

| Finding | Current evidence | Consequence |
|---|---|---|
| No intentional 180-degree turnaround | Source: the policy holds course +X heading and returns backward | Agree on the actual maneuver before training or deploying it |
| Visibility may fail before B gate entry | Ideal pinhole calculation with the example calibration/route: tag center exits the image at X about 3.066 m; gate entry is X about 3.376 m | Treat the roughly 0.31 m gap as a geometry warning, not a measured flight result |
| Heading guard does not establish physical braking | Fresh estimated error above 0.35 rad suppresses horizontal RC; zero RC is not instantaneous zero velocity | Measure delay and stopping response; test momentum near poles |
| Tilt and optical visibility are not represented by the current plant | CPU model has no roll/pitch/rotor dynamics or rendered tag detector | Do not use its passing results to clear bank-angle or camera-visibility cases |
| Disturbance/deadband limitations remain | Synthetic stronger drift: 0/20 completed; deadband: 6/20 completed | Preserve these failures in the report and calibrate before tuning compensation |

Visibility-calculation assumptions: course X points A-to-B; camera faces +X;
two tag centers are at A/B; lateral route offset is 0.5 m; gate radius is 0.8 m;
the example camera matrix is used. This is not the actual drone's calibration.
Tag facing, full-corner visibility, detection quality, and extra mapped tags are
not included. Recompute from the measured layout and camera/body transform.

## 2. Build the red-team tests we own

Start with isolated faults, then combine them at approach, braking, turning and
reacquisition. Keep true motion, estimated motion and commanded motion separate.

| Test family | Proposed simulation cases | Required evidence |
|---|---|---|
| Heading error/overshoot | Both signs of 30, 60, 90 degrees; 180 degrees; stationary and moving | Bounded heading recovery, displacement, command spikes, visibility and abort reason |
| Braking/turn entry | Sweep approach speed, distance to pole, response lag and command delay | Stop/turn outside measured swept-body clearance; no false gate completion |
| Tag visibility | Camera cone and tag-facing geometry; no tag; late tag; loss during maneuver | Tag quality/age and uncertainty drive continuation or abort; no unsupported blind flight |
| Delayed or false sensing | Video delay, frozen frames, plausible wrong poses, flipped heading and bad units | No motion spike or false lap; report missed detections and false aborts separately |
| Dynamics | Constant drift, gust during braking, asymmetric response, deadband and battery-dependent response | Completion, clearance and recovery measured across fitted ranges |
| Transport/runtime | Delayed/dropped commands, stale telemetry, CPU/inference stall and reconnect | Independent runner watchdog, one RC writer and latched abort; no silent resume |
| Handover | Manual takeover, fallback activation and return to policy | No old action replay, competing writer or abrupt command jump |
| Compounds | Turn + tag loss + delay; braking + gust + command loss | A failed lap remains bounded; an abort request is not counted as a completed landing |

Severity values are test proposals, not operating limits. Fit real ranges from
logs before using them as deployment limits. For development, use fixed seeds;
keep separate held-out seeds for the final comparison and do not train on them.
Roll/pitch, thrust/altitude coupling and physical stopping margins need an
appropriate dynamics model and hardware evidence; label them untested until then.

## 3. Make the shared maneuver decision

Give the teammate the findings and ask for these concrete inputs:

- Exact drone variant, camera calibration, tag sizes/poses/facing and camera/body transform.
- One real sensor log with capture/arrival times, measured velocities and uncertainty.
- His current simulator/estimator interfaces and the RL model's ordered features/actions.
- The runner's behavior when commands stop, packets disappear or a landing request occurs.

Jointly compare fixed-heading backward return with braking followed by a controlled
reorientation and return. Check visibility through the WHOLE path; a tag visible
at each endpoint is insufficient. If neither is observable, change tag coverage
or estimator design before asking the policy to complete the lap. Do not resolve
this by simply increasing the blind-flight timeout or RC cap.

We own the policy proposal and tests. He reviews simulator/estimator compatibility.
The agreed maneuver, shared frame, action order, timing and fallback transitions
must be written in CONTRACT.md before the deployment adapter is finalized.

## 4. Implement and compare the policy change

Add an explicit approach/brake/maneuver/reacquire/return state progression for the
agreed design. Keep phase transitions tied to fresh, trustworthy observations.
Preserve the selected-model/waypoint checks, abort latch and independent deadline.

Use measured response to set the braking envelope. A starting analytical check is
`distance >= speed * total_delay + speed^2 / (2 * braking_acceleration) + margin`.
It assumes a simplified constant-deceleration stop; it is not a validated Tello
guarantee. Margin must include the vehicle footprint, estimation uncertainty and
tracking error. Slewing RC values alone does not guarantee physical acceleration
or jerk limits.

Run old and new policies with identical geometry, fitted parameters, seeds and
time budgets. Report valid laps, lap times among completions, minimum TRUE pole
clearance, boundary/height violations, estimator error, recovery time, command
roughness, aborts and interventions. Keep estimate-timer vs truth-timer agreement
as a separate check. Report all attempts, including link-outage/void attempts.

## 5. Handoff acceptance and physical next step

The teammate should receive a draft PR containing the findings, scenario registry,
replay commands, changed policy, raw results and unresolved items. A green report
must mean its stated requirements passed; generating a report is not acceptance.

Enter grounded/shadow integration only after the shared contract and observation
producer are verified. Physical tests remain operator-run and staged: controlled
hover, braking/one segment, agreed maneuver, then a full lap. Any contact,
unexplained oscillation, boundary violation or loss of control blocks progression.
Zero observed failures in a small sample is not proof of zero risk.

Correct these definitions with the teammate before adopting the larger plan:

- Hover, landing request, command acknowledgment and actual touchdown are distinct outcomes.
- Software fault injection on an airborne drone can create real physical risk.
- Proposed simulated crash percentages are research statistics, not flight-release allowances.
- Use manufacturer guidance and measured operating conditions; an invented wind threshold is not clearance to fly.

No fault injection into a live drone, simulator rewrite, automatic merge, or
physical flight is performed by creating this plan.
