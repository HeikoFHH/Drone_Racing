# Kept vs removed from Action_Space

Source tree: Jaime’s local `action-space` checkout (`drone_deployment_infra/` + related).
This package is a **sibling directory** (`~/drone-race-hacks`), not an in-place rewrite of Action_Space.

## Kept (single-Tello race tooling)

| Area | Paths |
|------|--------|
| Wi‑Fi dongle path | `core/tello_dongle_setup.py`, `config/dongles.json`, stick-pinned connect in `core/tello_link.py` / `test_link.py` / `tello_dual_video.py` |
| Command APIs | SDK + RC send path in `core/tello_dual_video.py`, `tools/*` flight helpers |
| Positional access | `core/positional.py`, `positional_kalman_filter.py`, `tello_tags.py`, `config/map.yaml`, `config/tags.yaml` |
| Calibrated camera | `cal_and_map/`, `config/camera/`, `calibrate_map.md` |
| Tracks / viz | `core/extract_track.py`, `build_flight_viz.py`, `recordings/`, `outputs/` |
| Link tests | `tools/tello_range_test.py`, `tello_hover_test.py`, `tello_video_check.py` |
| Config plumbing | `core/paths.py`, `core/fleet.py` (still reads `quick_connect`, defaulted to one drone) |

## Added for the hack

| Area | Paths |
|------|--------|
| Race rules + lap timer | `docs/RACE_RULES.md`, `config/race_course.yaml`, `race/` |
| HTML deck | `presentation/index.html` |
| Cloudflare leaderboard | `leaderboard/` |

## Removed / not copied

| Area | Why |
|------|-----|
| `swarm_ops/` | Multi-drone orchestration / formation / swarm policy |
| `simulation_infra/` | Isaac Lab multi/single room sim — out of race-hack scope |
| `drone_gui/`, `drone_telemetry/` | Fleet GUI / extra telemetry not required for race seed |
| `core/team_features.py`, `room_policy.py`, `residual_policy.py` | Multi-drone / room PPO deploy helpers |
| `drone_deployment_tests/tello_swarm_to_origin.py`, RL deploy scripts | Multi-drone / policy-central stack |
| Lab `config/camera/tello_*.yaml`, full `dongles.json` fleet | Replaced with templates so the public repo does not bake one lab’s hardware map; re-run setup on site |
| Attack/defense fleet roles, `fly_count: 8` | Slimmed to `roles.race` + one `quick_connect` entry |

Action_Space working tree left untouched (aside from reading it as the seed).
