# Start here: Maude's policy branch

**Review status: ready for software review and offline simulation. Physical flight
integration is pending.** Work is isolated on `maude/policy-integration`.

## What this branch changes

Adds `policy_integration/` and an offline test workflow. The existing flight
engine and RL trainer are unchanged. This code never connects to a Tello, sends
RC, or takes off. It computes bounded proposed commands for a future runner.

The policy is a classic PD waypoint baseline plus a common guarded handoff for
asynchronous RL actions. The same Python policy is used in the CPU simulator.
The RL checkpoint, sensor filter and live flight adapter are separate handoffs.

## Run these first

From the repository root with Python 3.10+ and upstream `pyyaml` available:

```bash
python3 -m unittest policy_integration.test_policy policy_integration.test_stress_sim -v
python3 -m policy_integration.stress_sim --seeds 20 --output policy_integration/reports/stress_sim.json
python3 -m policy_integration.audit
```

There are 160 seeded scenario runs. These commands do not run Isaac Lab or train
an RL model. The simulator is an approximate velocity-response model, not Tello
physics. The report generator returning zero means it produced the report; it
does not mean every scenario succeeded.

## What red-teaming found and fixed

1. Continuously delayed video could hold indefinitely. Persistent staleness now
   requests landing after 0.5 s and latches abort.
2. A stalled route had no policy deadline. A 90 s airborne-session bound now
   aborts, and toggling enable cannot reset the deadline.
3. A fresh action could be for the wrong model or previous waypoint. Learned
   mode now pins a selected model ID and requires the current waypoint index.
4. Extremely large JSON integers could overflow validation. They now fail closed.
5. The visual player could retain an old animation callback. Stop/scenario change
   now cancels it, avoiding duplicate playback loops.

## What remains a known failure

The baseline completes 20/20 nominal, mild-drift and 150 ms-delay scenarios.
It completes 0/20 stronger constant-drift cases and 6/20 injected deadband cases.
Those failed missions now reach a bounded session abort; they are not successes.
Do not claim wind robustness or a calibrated RC deadband from these artificial
parameters. Use controlled logs to fit the plant, then compare compensation
changes against the same held-out cases.

## What each teammate must provide

- **Sensing:** a measured map-to-world transform and a real Observation sample,
  with velocity/uncertainty and true capture timestamps. Confirm tag IDs/sizes.
- **Flight runner:** one RC writer, shadow-only invocation first, explicit enable,
  telemetry freshness, latched abort, and landing request/acknowledgment handling.
- **RL partner:** exact feature/action preprocessing, model ID/manifest, compatible
  input features, and a separate inference worker returning ActionSample packets.
- **Maude:** integrate those contracts, replay real logs and evaluate the baseline.

Read CONTRACT.md before coding an adapter. Read SIMULATION.md for the simulator
tiers/results and RED_TEAM.md for the physical checks still pending. The fixed
heading baseline flies backward on the return: tag coverage on that leg must be
demonstrated or the maneuver redesigned with sensing.

## Review / merge conditions

Software review can happen now. Flight deployment requires measured coordinate
signs and scale, actual camera/telemetry age, calibrated margins, observed landing
handling, and a controlled operator test. A green CPU CI check is not flight
approval. Review this branch through a draft PR; do not merge to master directly.
