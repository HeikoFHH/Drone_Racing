# config

| File | What |
|------|------|
| `tags.yaml` | AprilTag family, sizes, detector gates. Race origin = tag 1. |
| `race_course.yaml` | Pole A / B tag ids, gate radius, height soft bounds |
| `map.yaml` | Tag poses in Pole A frame (remap on site) |
| `dongles.json` | USB Wi‑Fi stick → Tello (from `core/tello_dongle_setup.py`) |
| `fleet.yaml` | Single-drone `quick_connect` + calib identity |
| `camera/` | Per-drone calibration YAML (`default.yaml` fallback) |
