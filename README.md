# Drone Race Hacks

Toolkit for flying a single DJI Tello autonomously: camera calibration and AprilTag mapping, flight policies, a Mac-friendly simulator, and a lap timer for a two-pole race. It started at a Harvard i-lab hack and is based on [Action Space](https://github.com/jaromero3rd/Action_Space) and [`jaromero3rd/drone-race-hacks`](https://github.com/jaromero3rd/drone-race-hacks).

**Demo:** <https://heikofhh.github.io/Drone_Racing/> replays a simulated flight to a sign, with a camera view, distance readout and Start/Stop buttons (source in [`docs/sign_demo/`](docs/sign_demo/)).

| | |
|---|---|
| Race rules | [`docs/RACE_RULES.md`](docs/RACE_RULES.md) |
| Simulator guide | [`docs/SIMULATOR_GUIDE.md`](docs/SIMULATOR_GUIDE.md) |
| Flying | [`FLYING.md`](FLYING.md), [`core/README.md`](core/README.md) |
| Camera + map setup | [`calibrate_map.md`](calibrate_map.md) |
| Kept vs removed | [`docs/KEPT_REMOVED.md`](docs/KEPT_REMOVED.md) |
| Isaac Lab race | [`simulation/README.md`](simulation/README.md) (`AS-Tello-Race-v0`) |
| Leaderboard | [`leaderboard/`](leaderboard/) |
| Notes for LLM agents | [`CONTEXT/README.md`](CONTEXT/README.md) |

## What is in the repo

**Policies**: each takes tag detections and Tello telemetry and returns four stick values (right, forward, up, yaw; -100..100).

- [`race/homing_policy.py`](race/homing_policy.py), [`race/homing_control.py`](race/homing_control.py): flies a long course (default 73 m) from tag A to a stop 2-3 m short of tag B, turns 180 degrees, and returns to tag A. No map needed. It starts from whichever tag is in front, validates ranges, aborts on bad height or yaw telemetry, and has a start/stop switch. See [`CONTEXT/HOMING_POLICY.md`](CONTEXT/HOMING_POLICY.md). Tests: `race/test_homing_policy.py`.
- [`race/race_policy.py`](race/race_policy.py), [`race/fly_race.py`](race/fly_race.py): finds a tag, then flies the gates A, B, A for the timed lap.
- [`policy_integration/`](policy_integration/): a conservative waypoint controller with a strict input/output contract, RC limits, hold/abort behaviour, a stress simulator and an audit. It is the safety backup that guards the other policies in the simulator. Start with [`policy_integration/README.md`](policy_integration/README.md).

**Simulators**

- [`drone_race_sim/`](drone_race_sim/): Gymnasium-compatible two-pole race simulator that runs on a Mac without a GPU. Models stick delay, deadband, lag, wind, a camera with tag visibility, a Kalman filter and dead reckoning. It runs all policies side by side, trains a PPO policy, and writes plots and an HTML replay. Parameters are uncalibrated placeholders, so results compare policies in simulation and do not predict real flights.
- [`race/sim_homing.py`](race/sim_homing.py), [`race/sim_race.py`](race/sim_race.py): render the Tello camera image and run the real tag detector on it, with options for wind, heading error, missing or decoy tags, video delay, dropped frames and dead sensors.
- [`simulation/`](simulation/): Isaac Lab environment and PPO scaffold (needs a GPU machine).

**Flight infrastructure**

- [`core/`](core/): Tello link, flight engine, pose estimation, Kalman filter, dongle setup.
- [`cal_and_map/`](cal_and_map/): camera calibration and AprilTag mapping into `config/map.yaml`.
- [`connect-test/tello_basic.py`](connect-test/tello_basic.py): connects to a Tello, sends SDK commands interactively or as a script, and runs a policy from the command line.
- [`race/lap_timer.py`](race/lap_timer.py), [`race/submit_time.py`](race/submit_time.py), [`leaderboard/`](leaderboard/): lap timing and a Cloudflare Worker leaderboard.

**Planning docs** in [`docs/`](docs/): [`SIMULATION_PLAN.md`](docs/SIMULATION_PLAN.md), [`ROBUSTNESS_PLAN.md`](docs/ROBUSTNESS_PLAN.md), [`RECOVERY_PROPOSAL.md`](docs/RECOVERY_PROPOSAL.md) and [`SIM_TEST_REPORT.md`](docs/SIM_TEST_REPORT.md).

The sign-flight code behind the demo (`sign_mission/`: range estimation, landing policy, out-and-back policy, web control page) is on the [`future_policy`](https://github.com/HeikoFHH/Drone_Racing/tree/future_policy) branch.

## Setup

```bash
git clone https://github.com/HeikoFHH/Drone_Racing.git
cd Drone_Racing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e cal_and_map
```

### Simulator and tests (Mac or Linux, no GPU)

```bash
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python -r requirements-sim.txt
.venv-sim/bin/python -m pytest -q
```

Then follow [`docs/SIMULATOR_GUIDE.md`](docs/SIMULATOR_GUIDE.md). The homing policy's camera-in-the-loop sim runs with `python race/sim_homing.py` (options: `--help`).

### Dongle to drone

```bash
python core/tello_dongle_setup.py          # plug one USB Wi-Fi stick at a time
python core/tello_dongle_setup.py --list
```

Pin the NetworkManager profile to the stick interface (`nmcli connection modify TELLO-1 connection.interface-name wlx...`). Put that drone in `config/fleet.yaml` `quick_connect`.

### Camera and map

Follow [`calibrate_map.md`](calibrate_map.md). Race tags default to ids **1** (Pole A, origin) and **2** (Pole B) in `config/tags.yaml`.

### Fly and stream

```bash
python core/tello_link.py --no_fly    # connect and stream only
python core/tello_link.py             # launches the flight engine (see FLYING.md)
```

Positional API: `from positional import Positioner` (with `core/` on `PYTHONPATH`). Live UI while streaming: `http://127.0.0.1:8770/`.

## The race course

One lap, timed:

1. Leave the **Pole A** gate: the timer starts.
2. Enter the **Pole B** gate (turn).
3. Re-enter the **Pole A** gate: the timer stops.

Course config is in `config/race_course.yaml`; map the poles with `cal_and_map` into `config/map.yaml`. Time a recorded track with `python race/time_lap.py ...` (see `--selftest`). Rules: [`docs/RACE_RULES.md`](docs/RACE_RULES.md).

### Leaderboard

Source in `leaderboard/`; deploy steps in [`leaderboard/README.md`](leaderboard/README.md).

```bash
export LEADERBOARD_URL=https://<your-worker>.workers.dev
python race/submit_time.py --team "your-team" --time 12.345 --drone TELLO-1
```

## Layout

```
cal_and_map/         camera calibration + AprilTag mapping
config/              dongles, fleet, tags, map, race course
connect-test/        basic Tello connection and policy runner
core/                link, engine, pose, Kalman, dongle setup
CONTEXT/             notes for LLM agents (conventions, architecture, policies)
docs/                race rules, simulator guide, plans, figures
docs/sign_demo/      demo site served by GitHub Pages
drone_race_sim/      Mac simulator (Gymnasium), controllers, plots, replay
leaderboard/         Cloudflare Worker + static UI
policy_integration/  safety policy, contract, stress sim
presentation/        HTML slide deck
race/                lap timer, homing and gate policies, camera-in-the-loop sims
simulation/          Isaac Lab environment + PPO scaffold
tests/               simulator tests
tools/               range, hover and video checks, PID policy
```

## Safety

Flights are outdoors on a dirt field / parking lot. Keep a catcher ready and land on link loss. This repo sends real RC commands when you run the engine, so treat it like flight software. Try a new policy in simulation first, and never fly one that has only run in the simulator without a supervised first flight.

## License

See [`LICENSE`](LICENSE) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
