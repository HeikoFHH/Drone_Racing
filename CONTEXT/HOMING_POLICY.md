# Homing policy: A → 5 m from tag B → turn → 5 m from tag A

Owner: Tvisha (`tvisha/policy+sim`). For the controller: **start / stop is one object, see §1.**

> **Current settings (supersede the numbers further down, which were written for the first version):**
> stop zone **2.0–3.0 m** from each tag (aims for 2.5 m; never flies forward inside 2.0 m of any tag) ·
> flight height **0.8 m** (mount tag centres near 0.8 m) · cruise **1.8 m/s** (`tello_basic --rc-limit` default 36) ·
> turn at the first tag is a **fixed 180°**, then it keeps flying and locks on to the other tag ·
> **adaptive start**: `policy start new` uses whichever race tag is in front (or spins to find one) and
> measures the course itself; `--course-m` / `--tag-size-mm` are optional (defaults 73 m, 487.7 mm) ·
> turns only after 1 s settled inside the zone on a fresh sighting; a tag lost while stopped → spin to
> re-find, never turn blind. Sim: stops 2.43–2.61 m in every case (15/30/73 m, delay, frame loss, fast drone,
> starting backwards). Wind 1 m/s currently lands safely instead of finishing (drift during the start spin).

The 73 m course: tag A (id 1) and tag B (id 2) on poles, 2 ft prints, **facing each other**,
mounted at 1.2 m. The drone takes off in front of A with its nose toward B, flies at a constant
height toward B, stops **5 m short** of the tag, turns, finds A and stops 5 m short of it.
It only changes heading (the sideways stick is never used).

## 1. Start / stop from the controller

```python
from race.homing_control import PolicySwitch, obs_from_detections

switch = PolicySwitch()

def on_start_button():   switch.start()                    # resumes a paused mission
def on_stop_button():    switch.stop()                     # next tick: float in place
def on_new_mission():    switch.start(new_mission=True)    # drone back at A, nose toward B

# control loop, ~20 Hz
obs = obs_from_detections(t, tag_detector.detect(gray), height_m, yaw_deg)
rc, info = switch.step(obs)
send_rc(rc)                                  # send EVERY tick, also while stopped
if info["mode"] in ("land", "done"):
    land()
```

| `info["mode"]` | rc | What the controller does |
|---|---|---|
| `float` | `(0, 0, 0, 0)` | Policy stopped / not started. Tello holds position by itself. Keep sending (no rc for ~15 s → Tello auto-lands). |
| `go` | policy sticks | Send as is. |
| `done` | `(0, 0, 0, 0)` | Mission finished 5 m in front of A. Land (or keep floating). |
| `land` | `(0, 0, 0, 0)` | A safety limit tripped. Land now. Why: `info["abort_reason"]`. |

- Stop works on the very next tick. The buttons can run on another thread (lock inside).
- Start after stop **resumes** the same mission; the paused time is hidden from the policy's timers.
- `start(new_mission=True)` begins again from scratch (the drone must be back at A facing B).
- A tripped safety limit is sticky: only `new_mission=True` clears it.

Inputs: `t` (seconds, monotonic), the `TagDetector.detect(gray)` result from
`cal_and_map/tello_calibration` (unchanged), Tello telemetry `h/100` (m) and `yaw` (deg).
`core/tello_dual_video.listen_state` already parses both into `heights` / `motion["yaw"]`.

## 2. What the policy does (`race/homing_policy.py`)

| State | Behaviour |
|---|---|
| CLIMB | climb to 1.2 m (telemetry h); height is held all flight |
| BLIND | no tag detected → switches on by itself. Far from the tag: fly the leg heading (telemetry yaw) at 1.5 m/s. |
| SPIN | tag should be in detection range but isn't seen: hover and turn a full circle |
| TRACK | target tag seen: yaw to centre it, same speed, brake to stop 5.4 m out |
| HOLD | settle; at B, read where A is from tag B's pose (A is 73 m out of B's face) |
| TURN | turn toward A, slower (yaw stick 35) |
| DONE / ABORT | mode `done` / `land` |

**Validation loop** (every sighting): only the expected tag id counts; one-frame range jumps
> 3 m are rejected; the range trend must be shrinking (window grows with distance: noisy far
away). Flat or growing for 2 s → turn harder onto the tag; for 8 s → land.
Dead-reckoning cross-check flags a range that disagrees with the distance flown.

**Safety limits** (any → `land`): 300 s mission cap · height > 2.5 m · no height or no yaw
telemetry for 2 s · climbing hard with the height not changing (stuck sensor) · left the course
area while blind · never flies forward toward **any** tag closer than 5 m · blind distance assumes
the drone may be 30% faster than estimated, so a missing tag B still stops it > 5 m out.

## 3. Before a real flight (checklist)

1. **`config/tags.yaml` `sizes_mm` for tags 1 and 2 = the measured black square** of the print
   (2 ft sheet with border ≈ 487.7 mm; 2 ft black square = 609.6 mm). It still says 132.08.
   This is the one thing the policy cannot check: a 25% too-large value makes it stop at ~4.1 m.
2. Camera calibration for the drone (`cal_and_map`, `config/camera/<drone>.yaml`).
3. Tags facing each other, centres at 1.2 m (`HomingCfg.height_m`), 73 m apart (`course_m`).
4. Point the drone at B within ~20° at takeoff (the first ~45 m of each leg are flown blind).
5. One calibration flight: forward speed per rc unit (`HomingCfg.mps_per_rc`, placeholder 0.02),
   that telemetry yaw grows on a right turn (`yaw_sign`), Wi-Fi/video range at 73 m.
6. Staged: carry it by hand (policy running, nothing sent) → 15 m → 30 m → 73 m. Catcher ready.

The camera only sees tags: **no obstacle detection.** The space must be empty.

## 4. Simulated results (no wind, 2 ft tags, real TagDetector on rendered frames)

18 scenarios, **0 crashes**. Completed with the drone never inside 5 m (closest 5.03–5.40 m):
nominal (99 s) · start 5/10/20° off course · pole B 3 m sideways · stray tag id 7 · tag B
blocked 10 s · 30% frames lost · 200 ms video delay · height sensor frozen · yaw drift 0.1°/s ·
drone 30% faster / slower than assumed.
Landed safely: yaw telemetry lost · tag B missing · tag B missing + drone 30% faster (6.0 m).
Config errors: tags.yaml 610 mm vs 488 mm print → **4.13 m (inside 5 m)**; tags.yaml left at 132 mm → stops ~20 m short.
Wind: fine to ~1 m/s with gusts; above that it lands safely before finishing (heading-only steering).

## 5. Run it yourself

```bash
uv venv --python 3.11 .venv-sim
uv pip install --python .venv-sim/bin/python -r requirements-sim.txt -r requirements.txt -e cal_and_map
.venv-sim/bin/python -m pytest race/test_homing_policy.py -q        # 17 safety + start/stop tests
.venv-sim/bin/python race/sim_homing.py                              # nominal 73 m run, plot in outputs/sim/
.venv-sim/bin/python race/sim_homing.py --hide-b --mps-per-rc 0.026  # fault injection, see --help
```

| File | What |
|---|---|
| `race/homing_control.py` | **`PolicySwitch` (start/stop/float) + `obs_from_detections`** |
| `race/homing_policy.py` | the policy, validation loop, safety limits |
| `race/test_homing_policy.py` | safety + start/stop tests |
| `race/sim_homing.py` | 73 m simulator: renders camera frames, runs the real TagDetector, fault injection |
| `race/race_policy.py`, `race/fly_race.py`, `race/sim_race.py`, `drone_race_sim/baselines/race_policy_adapter.py` | earlier 4 m gate-race policy + harnesses |
