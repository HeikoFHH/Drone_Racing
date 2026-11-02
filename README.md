# Drone Race Hacks

Public single-Tello racing toolkit for **Jaime Romero** (`jaromero3rd`), seeded from [Action Space](https://github.com/jaromero3rd/Action_Space) real-world deployment code.

**At Harvard i-labs.** Open hack: build anything with Tellos + this repo. Optional outdoor two-pole race on the **dirt field / parking lot**; optional leaderboard. Iterate in **Isaac Lab** (`simulation/`) before the real course.

| | |
|---|---|
| Race rules | [`docs/RACE_RULES.md`](docs/RACE_RULES.md) |
| Isaac Lab race | [`simulation/README.md`](simulation/README.md) (`AS-Tello-Race-v0`) |
| Kept vs removed | [`docs/KEPT_REMOVED.md`](docs/KEPT_REMOVED.md) |
| Slideshow | [`presentation/index.html`](presentation/index.html) |
| Leaderboard | [`leaderboard/`](leaderboard/) |

## What we built at the hack (SundAI, 1 Nov 2026)

A team fork of this repo ([`HeikoFHH/Drone_Racing`](https://github.com/HeikoFHH/Drone_Racing)): flight policies for a single Tello, a simulator to test them on a Mac, and a demo site.

**Live demo:** <https://heikofhh.github.io/Drone_Racing/> is a replay of the simulated *fly to a 22.5 in sign, stop 0.5 m in front, land* mission, with Start and Stop buttons, a camera view and a live distance readout. It is built from [`docs/sign_demo/`](docs/sign_demo/) and deployed by GitHub Actions on every push to `master`.

| Piece | Who | Where | What it does |
|---|---|---|---|
| Homing policy | Tvisha | [`race/homing_policy.py`](race/homing_policy.py), [`race/homing_control.py`](race/homing_control.py), [`CONTEXT/HOMING_POLICY.md`](CONTEXT/HOMING_POLICY.md) | Flies the 73 m course tag A to tag B and back with no map: stops 2-3 m from each tag, turns a fixed 180 deg, and has a start/stop switch. Adaptive start (uses whichever tag is in front, or spins to find one), validates ranges, aborts on bad height/yaw telemetry or tag loss. 30 unit tests: [`race/test_homing_policy.py`](race/test_homing_policy.py). |
| Gate-racing policy | Tvisha | [`race/race_policy.py`](race/race_policy.py), [`race/fly_race.py`](race/fly_race.py) | Finds a tag, then flies the gates A, B, A using the repo's positioner and lap timer. |
| Camera-in-the-loop sim | Tvisha | [`race/sim_homing.py`](race/sim_homing.py), [`race/sim_race.py`](race/sim_race.py) | Renders the Tello camera image and runs the real tag detector on it. Scenarios: wind and gusts, heading error, misplaced or missing tag, decoy tag, video delay, dropped frames, dead height/yaw sensors. |
| Safety policy + review | Maude | [`policy_integration/`](policy_integration/) | Offline PD waypoint controller with a strict input/output contract, RC limits, abort and hold behaviour, a stress simulator, an audit and a red-team list. It is the **backup** that guards every other policy in our simulator. |
| Connection + basic flight | Bailey | [`connect-test/tello_basic.py`](connect-test/tello_basic.py) | Connects to the Tello, sends SDK commands, runs the policy from the command line and pauses the keep-alive while it flies. |
| Mac simulator | Heiko | [`drone_race_sim/`](drone_race_sim/), [`docs/SIMULATOR_GUIDE.md`](docs/SIMULATOR_GUIDE.md) | Gymnasium-compatible, runs on a Mac without a GPU. Models stick delay, deadband, lag, wind, a camera with tag visibility, a Kalman filter and dead reckoning. Compares all team policies side by side, trains a PPO policy, and draws plots and an HTML replay. |
| Robustness plan | Heiko | [`docs/ROBUSTNESS_PLAN.md`](docs/ROBUSTNESS_PLAN.md), [`docs/RECOVERY_PROPOSAL.md`](docs/RECOVERY_PROPOSAL.md) | Fault cases (tag blackout, sensor and link problems, being pushed off the path) and a proposed recovery layer; tag blackout injection is in the sim. |
| Sign mission | Heiko | branch [`future_policy`](https://github.com/HeikoFHH/Drone_Racing/tree/future_policy) (`sign_mission/`) | Take off, fly to a 22.5 in sign, stop 0.5 m in front, land, in one go. Range estimate from the sign's size with a Kalman filter, landing policy, a second policy that turns 180 deg and flies back, and a web control page (video, sign detected, distance, Start, Stop). Details in `docs/SIGN_MISSION.md` and `docs/SIGN_FLIGHT_OVERVIEW.md` on that branch. |

**Branches.** `master` holds the homing policy, the simulator and the demo site. `future_policy` is the earlier master and also contains the sign mission code and the site's build script. `final-policy` is the branch `master` was set to.

**Run the tests** (Python 3.11, `pip install -r requirements-sim.txt`): `python -m pytest -q` gives 80 passed on `master`.

**What is and is not verified.** The simulator uses placeholder dynamics and camera numbers because we have no recorded real flights, so its results show how policies compare in simulation, not how the real drone will fly. The sign mission has been run only in the simulator and against a fake drone; it has not been flown on hardware. Fly a new policy first in shadow mode (computes commands, sends none) and keep a catcher ready.

## Course (one lap)

1. Leave **Pole A** gate → timer starts  
2. Enter **Pole B** gate (turn)  
3. Re-enter **Pole A** → timer stops  

Config: `config/race_course.yaml`. Map poles with `cal_and_map` into `config/map.yaml`. Time a track with `python race/time_lap.py …` (see `--selftest`).

## Setup

```bash
git clone https://github.com/jaromero3rd/drone-race-hacks.git
cd drone-race-hacks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e cal_and_map
```

### Dongle → drone

```bash
python core/tello_dongle_setup.py          # plug one USB Wi-Fi stick at a time
python core/tello_dongle_setup.py --list
```

Pin the NetworkManager profile to the stick interface (`nmcli connection modify TELLO-1 connection.interface-name wlx…`). Put that drone in `config/fleet.yaml` `quick_connect`.

### Camera + map

Follow [`calibrate_map.md`](calibrate_map.md). Race tags default to ids **1** (Pole A / origin) and **2** (Pole B) in `config/tags.yaml`.

### Fly / stream

```bash
python core/tello_link.py --no_fly    # connect + stream only
python core/tello_link.py             # launches flight engine (see FLYING.md)
```

Positional API: `from positional import Positioner` (with `core/` on `PYTHONPATH`). Live UI when streaming: `http://127.0.0.1:8770/`.

More: [`FLYING.md`](FLYING.md), [`core/README.md`](core/README.md).

## Leaderboard (Cloudflare)

Source lives in-repo under `leaderboard/`. Deploy notes and the KV / secret steps Jaime must run: **[`leaderboard/README.md`](leaderboard/README.md)**.

```bash
export LEADERBOARD_URL=https://<your-worker>.workers.dev
python race/submit_time.py --team "your-team" --time 12.345 --drone TELLO-1
```

## Layout

```
cal_and_map/     camera calib + AprilTag mapping
config/          dongles, fleet, tags, map, race_course
core/            link, engine, pose, Kalman, dongle setup
simulation/      Isaac Lab AS-Tello-Race-v0 + PPO scaffold
tools/           range/hover/video checks, to-origin helper, PID policy
race/            lap timer, submit helper, homing + gate policies, camera-in-the-loop sims
drone_race_sim/  Mac simulator (Gymnasium), controllers, plots, replay
policy_integration/  Maude's safety policy, contract, stress sim
connect-test/    Bailey's basic Tello connection + policy runner
tests/           simulator tests
CONTEXT/         notes for LLM agents (conventions, architecture, policies)
leaderboard/     Cloudflare Worker + static UI
presentation/    self-contained HTML deck (dirt_field venue map)
docs/            race rules, simulator guide, plans, figures
docs/sign_demo/  the demo site served by GitHub Pages
```

### Isaac Lab (sim)

See [`simulation/README.md`](simulation/README.md). Quick path on a GPU box with Action Space’s Isaac install:

```bash
./simulation/scripts/train_race.sh --num_envs 256 --max_iterations 200
```

## Safety

Timed race is **outdoors** on the dirt field / parking lot by Harvard i-labs. Catcher ready. Land on link loss. This repo sends real RC when you run the engine — treat it like flight software.
