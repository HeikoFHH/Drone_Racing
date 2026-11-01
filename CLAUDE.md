# Repo notes for Claude sessions

- LLM-intended documents live in `CONTEXT/` (start with `CONTEXT/README.md`); human-readable descriptions of work live in `docs/`.
- Simulator: `drone_race_sim/` (use `.venv-sim`), tests: `.venv-sim/bin/python -m pytest -q`.
- Do not describe simulator results as predictions for the real drone: all parameters are uncalibrated placeholders.
