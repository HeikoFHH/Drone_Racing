# Race rules — two-pole one-lap

## Course

- **One DJI Tello at a time.** No simultaneous multi-drone flights in this hack.
- Two poles, each with a visible AprilTag (`tag36h11`):
  - **Pole A** — tag id from `config/race_course.yaml` (`pole_a_tag`, default **1**). Start / finish.
  - **Pole B** — `pole_b_tag` (default **2**). Turn marker.
- Map the tags into `config/map.yaml` (see `calibrate_map.md`). Example seed map places B ~4 m from A along +X; **remap on site**.

## Lap definition (timed)

Implemented by `race/lap_timer.py`:

1. Enter the **Pole A gate** (horizontal distance ≤ `gate_radius_m`, default 0.8 m).
2. **Leave** Pole A → **timer starts**.
3. Enter the **Pole B gate** (required turn).
4. Re-enter the **Pole A gate** → **timer stops**. That interval is the lap time.

Re-entering Pole A before Pole B resets to “at start” (false start). Height soft bounds are logged via config (`min_height_m` / `max_height_m`); enforce crash/safety rules separately.

## Scoring

- Primary: **lowest valid lap time** (seconds).
- Submit to the public leaderboard (`leaderboard/`) with team name + time.
- Honor system for this hack unless organizers add video review.

## Safety

- Fly only in the cleared course volume. Keep a catcher ready.
- Link loss: land; do not continue a timed lap on dead reckoning without a tag.
