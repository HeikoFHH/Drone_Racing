# Interactive drone control lab

Open `dashboard.html` locally in a modern browser. It is self-contained and needs
no server, packages, network, GPU or physical drone. GitHub displays HTML source;
download the file to open the interactive version. The conversation version uses
the same source and exported configuration.

## Controls

- **Forward / Backward / Left / Right:** take over from the policy and issue a
  one-second simulated body-relative pulse. Then RC becomes zero, the synthetic
  plant coasts for 0.8 seconds, and simulation time pauses.
- **Stop RC:** immediately sets simulated demand to zero and allows the same
  settling period. It is not a physical emergency stop.
- **Run policy:** starts or resumes the classic waypoint controller. After manual
  takeover, it creates a fresh controller for the current simulated position.
- **Pause simulation:** freezes time and physics. It does not demonstrate hover.
- **New run:** clears the trace, controller state and abort latch.
- **Inject +60° turn:** changes true heading. The next fresh pose lets the policy
  brake/aim; stale sensing can prevent recovery.
- Policy, sensing and fault selections start a new run. Playback changes only
  wall-clock speed, preserving the 50 ms simulation/control tick.

Default sensing approximates two point tags in the example camera image. Choose
ideal pose to isolate controller behavior; it is not an implemented estimator.
Faults activate at five simulated seconds. No trained RL checkpoint is loaded;
the RL owner must supply the feature adapter, model and validated action mapping.

## Implementation and evidence

`dashboard_core.js` previews the **classic** Python policy. Configuration is
generated from `PolicyConfig` and `candidate_config`, rather than independently
tuned browser constants. `test_dashboard.py` compares sequential decisions across
all 16 turn scenarios for both configurations and checks guard cases. Node tests
exercise the actual dashboard script with a DOM harness: manual direction signs,
stop, pause/resume, terminal latch, reset, policy completion under ideal sensing,
camera loss, injected faults, sharp-turn response and SVG widths 320/736/1024.
This is logic verification, not a rendered-browser layout or real-flight test.

Rebuild and verify from the repository root:

```bash
python3 -m policy_integration.build_dashboard --standalone policy_integration/dashboard.html
python3 -m unittest policy_integration.test_policy policy_integration.test_stress_sim policy_integration.test_turns policy_integration.test_dashboard -v
```

The preview uses a noise-free planar first-order velocity plant. The seeded Python
turn comparison adds synthetic measurement noise and randomized starts. Neither
models rotor physics, roll/pitch, thrust coupling or actual radio/landing dynamics.
The drawn cone is local orientation only; tag visibility calculations use the
full point projection. Manual mode has truth-geofence/clearance/time termination,
not the entire autonomous policy envelope or a physical safety system. Telemetry
is synthetic and battery is fixed at 80%. Aborts freeze the preview; no landing is
simulated. “Complete” is the policy state, not certified race-timer completion.
