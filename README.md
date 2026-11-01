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
race/            lap timer + submit helper
leaderboard/     Cloudflare Worker + static UI
presentation/    self-contained HTML deck (dirt_field venue map)
docs/            race rules, keep/drop notes
```

### Isaac Lab (sim)

See [`simulation/README.md`](simulation/README.md). Quick path on a GPU box with Action Space’s Isaac install:

```bash
./simulation/scripts/train_race.sh --num_envs 256 --max_iterations 200
```

## Safety

Timed race is **outdoors** on the dirt field / parking lot by Harvard i-labs. Catcher ready. Land on link loss. This repo sends real RC when you run the engine — treat it like flight software.
