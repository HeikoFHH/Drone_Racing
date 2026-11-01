# tools

Single-drone flight helpers and link tests (from Action Space `drone_deployment_tests/`, swarm scripts dropped).

| File | What |
|------|------|
| `tello_policy.py` | PID approach policy imported by the flight engine |
| `tello_range_test.py` | Walk-range test (no takeoff) |
| `tello_hover_test.py` | Hover link test |
| `tello_video_check.py` | H.264 path diagnostic |
| `tello_to_origin.py` | Fly to map origin and land |
| `rc_rate_bench.py` | RC rate bench |

Which drone connects: `config/fleet.yaml` `quick_connect` + present sticks in `config/dongles.json`.
