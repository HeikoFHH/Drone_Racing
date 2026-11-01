# Team execution plan

Prepared November 1, 2026. Owner: Maude, flight policy and integration.
Repository: HeikoFHH/Drone_Racing. Work branch: maude/policy-integration.

## Intended outcome

Demonstrate one Tello completing the measured A -> B -> A course using a verified
classic baseline. Then evaluate the RL policy through the same deployment
contract and compare results. The LLM improvement loop runs between experiments,
proposes bounded training changes, and keeps changes only after evaluation.

The current delivery executes the offline software stage. Physical flight,
sensor calibration, live runner wiring, RL training, and RSI automation remain
explicit team handoffs. No motors have been controlled from this environment.

## Responsibilities and required handoffs

| Owner | Owns | Must provide | Does not own |
|---|---|---|---|
| Maude | Flight policy, contract, guards, replay, integration acceptance | Versioned policy, tests, route configuration, pass/fail report | Wi-Fi sockets, unreviewed changes to training |
| RL partner | Simulator, training, checkpoint, deployment preprocessing | Model manifest, exact features/actions, same-course evaluation, inference worker | Direct motor writes or bypassing guards |
| Sensing teammate | Calibration, tag registry/map, localization/filter | Canonical pose/velocity/yaw, capture timestamps, uncertainty, verified map transform | Policy gains or lap completion claims |
| Flight/connection teammate | Drone link, telemetry, lifecycle and sole RC writer | Timestamped telemetry, explicit enable, land acknowledgment, abort latch, shadow hook | Changing controller outputs silently |
| Experiment/RSI teammate, if present | Experiment proposals, logs, evaluation dashboard | One-change experiments, seeds, configs, metrics, accept/reject decision | LLM calls inside real-time control |

These are proposed role boundaries. One teammate may own multiple roles. Names
are intentionally not guessed. The immediate team agreement is the field-level
contract in CONTRACT.md, not a new training algorithm.

## Milestones in dependency order

| Stage | Work now | Evidence required to advance |
|---|---|---|
| 0. Branch and contract | Keep all policy changes on maude/policy-integration; teammates review CONTRACT.md | Each producer/consumer agrees on axes, timestamps, RC order, ownership |
| 1. Offline policy | Run tests, toy route, adversarial cases and source audit | Passing report; known blockers written down; no real-flight claim |
| 2. Real observations | On the ground, unify tag IDs/sizes/map, measure the transform, log canonical JSONL | Tape-measured scale, forward/left/up/yaw sign checks, truthful age/uncertainty |
| 3. Shadow integration | Runner calls policy and logs proposals but sends no policy RC; carry drone around course | Correct proposal signs, phase progression, stale-pose behavior and no competing RC writer |
| 4. Classic physical baseline | Runner honors all guards and land requests; operator tests short hover, one offset waypoint, then full course | Logs/video and authoritative lap timer; no unhandled aborts; 3 consecutive controlled laps as a provisional demo gate |
| 5. RL comparison | Partner supplies compatible model/inference worker; use same course and test budget | Proposed gate: at least 18/20 held-out simulated successes, zero guard-bypass events, then shadow checks before a controlled physical attempt |
| 6. LLM improvement loop | LLM reviews failures and proposes one reward/noise/curriculum change; partner retrains | Candidate beats baseline evaluation under the same seeds, horizon, and budget without worse failure rate |

Stages 2-4 require the team's physical laptop/drone and an operator. This cloud
workspace is not connected to their Wi-Fi dongle or camera. The stage-4 live
adapter is not implemented or validated by the offline prototype.

## First team work session

1. Maude runs the four README commands and shows the contract and red-team report.
2. Sensing owner supplies ONE actual canonical JSONL sample and the measured
   map transform. If any field is unavailable, mark it unavailable and resolve
   its producer; do not invent values to make the API pass.
3. Runner owner adds shadow-only invocation at the existing policy boundary,
   before SDK dispatch, while preserving all lifecycle and loss handling. The
   upstream --no_fly branch exits before policy evaluation, so a shadow hook
   needs deliberate wiring; it will not work merely by importing this package.
4. RL owner confirms whether the 21-feature PPO input can be estimated on the
   drone. Choose verified extra features or sensor-realistic retraining before
   continuing a long training run.
5. Fix current source blockers under their owners' branches. Integrate reviewed
   changes into this policy branch; rerun the contract tests before live use.

For stream-only inspection on a GROUNDED drone after resolving the source config:

```bash
python core/tello_link.py --no_fly
```

This existing command starts/reaps engine processes and requires the upstream
dongle/NetworkManager configuration. It is not a portable Mac setup command or
a way to add a second controller to an airborne drone. Confirm the laptop OS and
dongle support with the runner owner. Avoid running the default takeoff path as
an inspection command: the upstream engine automatically launches after ~20 s.

## Current integration findings

| Finding | Evidence in reviewed base | Resolution owner |
|---|---|---|
| Different tag registries | config/tags.yaml has race IDs 1/2; live tello_tags.py has 5/6/7/10; engine fallback TAG_ID=14 | Sensing + runner |
| Different map paths | Positioner reads config/map.yaml; live legacy engine imports tello_tags.load_map with core/tag_map.json default | Sensing + runner |
| Different horizontal planes | Existing lap timer uses map X/Z; simulator gates use world X/Y | Maude + sensing; explicit adapter supplied, physical calibration pending |
| Different available features | PPO takes 21 simulator-state features; real canonical packet lacks body angular velocity/gravity | RL + sensing |
| Training registration may not reach trainer | train_race.sh imports task in one subprocess then launches an external trainer | RL partner verifies task import inside actual trainer |
| Existing baseline is approach-and-land | tools/tello_policy.py approaches a tag, not a complete lap | Maude; new offline waypoint policy supplied |
| Camera coverage on return leg | New fixed-heading baseline returns backward; tags may leave view | Sensing; measure coverage or jointly add turn maneuver |
| Evaluation duration differs | Conservative toy baseline ~50 s; current Isaac episode 45 s | RL + Maude align horizon before comparison |

The training registration item is a verification requirement, not a reproduced
Isaac error: its external trainer and GPU runtime were not executed here.

## Red-team release gates

Software: no non-finite or oversized action reaches output; stale inputs cannot
complete a waypoint; abort remains latched; only fresh matched RL packets are
accepted; a scored lap must pass the existing timer after the proper transform.

Ground/shadow: prove observation scale, axes, capture-time age and false tag
rejection. Replay actual logs. Demonstrate requested vs transmitted vs
acknowledged landing separately. Test link-loss/reconnect logic with grounded
hardware or a transport mock. No deliberate radio disconnection in flight is
needed for this gate.

Physical: operator verifies the actual route, footprint clearance, landing spot,
local space, and one-drone operation. Test one behavior at a time, then the lap.
Camera lag, motor/radio interference, drift, RC deadband, and stopping distance
are unknown until measured. A passing toy test is insufficient to clear this gate.

## Evaluation and RSI experiment record

Log session/model/config/git IDs, policy mode, source timestamps, observed pose
and uncertainty, proposed/applied RC, guard reasons, runner mode, landing
request/ack/result, independent lap phase/time, and any human intervention.

Primary metrics: valid lap completion fraction, lap time among valid completions,
abort/intervention counts, deadline misses and worst observation age. Count a
fast crash as a failure, not a good lap. Compare equal budgets and horizons.
Use fixed held-out scenarios for reporting; generate separate training scenarios.

RSI iteration: pick one observed failure -> propose one allowed change -> record
config diff -> retrain for the same budget -> evaluate -> accept/reject -> retain
the best verified checkpoint. First candidate experiments: delay/dropout-aware
training, explicit progress rather than sustained proximity reward, and a
curriculum from waypoint reaching to full laps. The current exp(-distance)
reward is a potential lingering incentive; measure whether it is actually
exploited before changing it. The LLM has no authority to change runtime limits.

## Completion status

Implemented: offline policy, explicit transform API, async learned-action packet
contract, replay, toy bench, failure tests, static audit, team plan.

Pending: team contract acceptance, measured sensor packet, real runner patch,
hardware shadow and flights, compatible trained checkpoint, actual LLM loop.

Use `maude/policy-integration` for GitHub publication and team review. Repository
write access has been enabled for this handoff. No merge or physical flight is
authorized by publishing the review branch.
