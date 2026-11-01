"""Fly to a sign (AprilTag print, 22.5 in wide), stop 0.5 m in front of it, and land.

Modules
  spec        sign size, stop distance, camera model (config/camera/default.yaml)
  common      dataclasses shared by everything (observations, estimates, actions)
  range_est   SignRangeEstimator: multi-cue distance fusion + Kalman filter + near-field dead reckoning
  policies    ApproachPolicy (fly to sign, stop 0.5 m) and OutAndBackPolicy (also turn 180 deg and fly back)
  landing     LandingPolicy (stop, settle, descend, land)
  runner      MissionRunner: takeoff -> policy -> landing, one rc writer, start/stop, watchdogs
  sim         SimDrone: simulated Tello with a rendered camera (same interface as the real drone)
  tello       TelloDrone: the real drone over the Tello SDK (live mode)
  server      web control interface (video, detection + distance, Start / Stop)
"""
