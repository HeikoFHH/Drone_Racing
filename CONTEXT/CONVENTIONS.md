# Conventions (read first)

## Where things go
- Human-readable descriptions of work done: `docs/`. LLM-intended documents: `CONTEXT/`. Generated figures and the
  replay page that humans look at: `docs/figures/`. Bulk run outputs (checkpoints, logs): `outputs/` (git-ignored).
- Code: simulator `drone_race_sim/`, training `train/`, tests `tests/`, scripts `scripts/`. Maude's `policy_integration/`
  and Tvisha's `race/race_policy.py`, `race/fly_race.py`, `race/sim_race.py` are other people's code: do not edit them to
  make the simulator pass; adapt in `drone_race_sim/controllers.py` instead and report the finding.

## Frames and units (single source of truth: `drone_race_sim/frames.py`)
- Sim frame is `course_flu_v1`: X from Pole A to Pole B, Y left, Z up, metres, m/s; yaw is radians CCW from +X.
- Repo map frame (config/map.yaml, core/positional.py, race/): x right, y up, z out of the tag face. Default
  map->course rotation `R = [[1,0,0],[0,0,-1],[0,1,0]]`, `t = 0` (a test fixture; the real transform must be measured).
- rc sticks are `(right, forward, up, yaw_clockwise)` integers in [-100, 100]. rc yaw positive = clockwise = yaw decreases.
- Control tick 0.05 s (20 Hz). Lap rules: `race/lap_timer.py` is authoritative; `VecLapTimer` must stay in parity.

## Claims policy
- Dynamics and sensor parameters are **placeholders** (`params/*_placeholder.yaml`, each annotated with source and
  confidence) until real flights are recorded. Never describe sim results as predictions for the real drone. Checkpoints
  carry `params_source`.
- Say which sensing mode a result used (`perfect` = truth, `realistic` = camera/tag/estimator model).
- A landing request in the sim ends the episode as outcome `landed` (a safe abort); the descent is not simulated.

## Environment and commands
- Python env: `.venv-sim` (uv, Python 3.11). Run everything from the repo root with `.venv-sim/bin/python`.
- Tests: `.venv-sim/bin/python -m pytest -q` (80+ tests, ~15 s). They include exact parity with
  `policy_integration/stress_sim.py` and `race/lap_timer.py`: a parity failure means the sim changed, not the reference.
- Training: `.venv-sim/bin/python -m train.train --out outputs/sim/<name> --device mps [--sensing realistic]`.
- Full report + figures + replay: `.venv-sim/bin/python scripts/make_report.py --docs-dir docs/figures`.
- zsh: do not write `$VAR:path` in shell strings (modifier); use `${VAR}:path`. Quote args containing `+` (branch `tvisha/policy+sim`).

## Git
- Commits and pushes use the globally configured identity (HeikoFHH, hfh.hoppe@t-online.de). End commit messages with the
  attribution line required by the session. Do not push without being asked; merging branches into master was explicitly
  requested by the owner for `maude/policy-integration` and `tvisha/policy+sim`.
- Maude's branch kept moving after the merge (CI workflow and reports); those later commits are NOT merged.
