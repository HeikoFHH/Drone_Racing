# Maude's flight policy and team integration

Branch: `maude/policy-integration`.
Reviewed base: `31c17c25a1c6c0e73a4e8114492fe226ad338878` in `HeikoFHH/Drone_Racing`.

Maude owns the controller, its shared input/output contract, and integration tests.
Her partner owns RL training and the exported model. Roles below are proposed
work assignments; no teammates have been contacted or assigned in an external tool.

## Start here

Open [dashboard.html](dashboard.html) in a browser after cloning or downloading it.
It provides manual forward/back/left/right pulses, classic policy execution,
live synthetic telemetry, camera visibility, and fault injection. GitHub's file
view shows its source; download the file to run it. The same simulation can be
used in the conversation. It has no physical drone connection or trained RL model.
See [DASHBOARD.md](DASHBOARD.md) for controls, parity checks, and model limits.
See [TURN_RESULTS.md](TURN_RESULTS.md) for the reproduced turn/visibility blocker.

From the repository root, run:

```bash
python3 -m unittest policy_integration.test_policy -v
python3 -m policy_integration.bench
python3 -m policy_integration.audit
python3 -m policy_integration.replay policy_integration/examples/observations.jsonl
python3 -m policy_integration.stress_sim --seeds 20 --output policy_integration/reports/stress_sim.json
```

The policy, adapter, toy bench, and replay require only the Python standard library.
The audit and the one existing-lap-timer integration test use upstream `pyyaml`.
None of these commands connects to a drone, takes off, or sends motor commands.
Existing upstream flight commands still run the old controller: this module has
NOT been inserted into `core/tello_dual_video.py`.

Read [EXECUTION_PLAN.md](EXECUTION_PLAN.md) for owners, handoffs, and completion gates.
Read [CONTRACT.md](CONTRACT.md) before another teammate implements an adapter.
Read [RED_TEAM.md](RED_TEAM.md) for adversarial cases and unverified hardware risks.
Read [SIMULATION.md](SIMULATION.md) for the CPU/physics/GPU simulation path and current failures.

## What is implemented

- `policy.py`: classic PD waypoint control, explicit course frame, normalized RL
  action packet handoff, input validation, RC limits and slew, hold/abort behavior,
  pole-path checks, route dwell and completion.
- `adapter.py`: explicit rigid transform from `Positioner.locate()` camera pose to
  course-frame position/yaw. The estimator supplies velocities and uncertainty.
- `replay.py`: canonical sensor log to policy decisions, without flight I/O.
- `bench.py`: small deterministic first-order toy plant with randomized starts and
  noise, plus pose-loss and link-loss fault injections.
- `test_policy.py`: adversarial and interface tests, including the existing lap
  timer after coordinate conversion.
- `audit.py`: source-level checks for current integration blockers.

The toy plant follows the upstream approximate RC-to-speed mapping. It has no
wind, video, real radio, battery dynamics, propeller physics, actuator calibration,
or collisions. Its route completion is a software regression result, not a drone
flight result or an RL training result. The conservative example takes around
50 seconds, longer than the upstream simulator's 45-second episode; align the
evaluation horizon before comparing this baseline with PPO.

The example route is A-offset -> B-offset -> A-offset at constant heading. The
return leg flies backward. The sensing owner must show that localization remains
available on that leg. If it does not, add mapped tag coverage or jointly design
a turn maneuver; the current policy will hold and then request landing on loss.

## Flight runner integration point

```python
from policy_integration import BaselinePolicy

policy = BaselinePolicy()  # instantiate once for a fresh session
# In the existing runner, AFTER a verified adapter and existing lifecycle checks:
decision = policy.step(canonical_observation, now_monotonic_s)
# Shadow phase: log decision.to_dict(); do not send its rc values.
# Later, the runner's only RC writer may dispatch decision.rc.
# The runner must execute/acknowledge decision.request_land; the module cannot.
```

This is an API example, not a completed engine patch. The upstream engine produces
legacy camera-to-tag pose, not `Positioner.locate()` output. Do not pass its
`poses[name]` directly into `PositionerAdapter`.

For RL, select `BaselinePolicy(mode="learned", expected_model_id="<checkpoint-id>")`
at session creation and supply a fresh `ActionSample` from a separate inference
worker on every tick, including the current waypoint index. The selected model
ID is pinned for the session. Do not run
blocking model inference or an LLM inside the flight tick. No checkpoint is included,
and the upstream 21-feature PPO model cannot be plugged in without its feature
adapter and measured action semantics. Use a new session to change policy mode.

## Branch workflow

Use this branch for policy changes. Rebase/integrate teammates' work only after
reviewing their contract changes. Do not modify `master` directly. A draft PR is
the collaboration artifact for team review. Publishing this feature branch does
not authorize merging or enabling physical flight.
