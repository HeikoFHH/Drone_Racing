# Send this to the person with the drone

This adds a local website with real Tello video, hold-to-move forward/back/left/right/
up/down, Take off, Land, Run policy and Stop policy. It runs on **the computer
connected to the drone's Wi-Fi**, not on ChatGPT's server. The URL is
`http://127.0.0.1:8780/` on that computer. It is not a remotely hosted public link.

## Install and try the buttons without a drone

```bash
git clone --branch maude/policy-integration https://github.com/HeikoFHH/Drone_Racing.git
cd Drone_Racing
python3 -m venv .venv-live
source .venv-live/bin/activate
python -m pip install -r live_control/requirements.txt
python -m pip install -e cal_and_map
python -m live_control.bridge
```

Open `http://127.0.0.1:8780/`. The DEMO label and synthetic image identify this mode.
Click Take off, select Maude's baseline or Tvisha's guarded RacePolicy, then Run
policy. Stop policy cancels its output and requests zero RC; manual direction
buttons also take over. One policy run is allowed per session; restart after
landing and review to run another. No trained PPO checkpoint is loaded.

## Connect the real camera first

Close every other Tello app/runner, including `core/tello_link.py`,
`core/tello_dual_video.py`, `race/fly_race.py` and any phone controller. Do not
launch the old link script alongside this dashboard: it can auto-launch and would
compete for command ownership. Join the selected Tello's Wi-Fi in system settings.

```bash
python -m live_control.bridge --real
```

This is camera-only. It sends SDK `command` and `streamon`, never takeoff. The
forward camera and state packets must arrive before flight can be enabled.
The bridge exclusively binds UDP 8889, 8890 and 11111. On a multi-radio Linux rig,
select the drone adapter with `--interface <actual-interface>`. On Mac or a
single-radio laptop, normal routing usually suffices; `--bind-ip <local-drone-WiFi-IP>`
selects the local address. Interface binding may require OS privileges. No Wi-Fi
profiles, firewall rules or privileges are changed by the bridge.

## Manual flight

Stop the camera-only bridge and launch:

```bash
python -m live_control.bridge --real --allow-flight
```

Open the same URL. Starting this command does not take off. Take off is a separate
operator action and needs fresh video/telemetry, battery at least 25%, and ground-
consistent reported height. These are prototype limits, not validated aircraft
operating specifications. Do a controlled operator-run hover/manual test before
autonomous testing. Keep the flight area clear and the drone in direct view.

## Run the actual repository policies on the real drone

`live_control/policies.py` calls the existing `BaselinePolicy` or
`race/race_policy.py::RacePolicy`; it does not copy or rewrite either controller.
Race mode uses the existing `LapTimer` and `FilteredPositioner`. Race RC passes
through Maude's existing action-packet guards, cap/slew, fresh-pose, heading/braking,
clearance and abort latch. Prototype race settings reduce cruise to 0.16 m/s and
cap sticks at 20; neither setting is a hardware calibration. No simulator-only
relaxed pose age or invented position is used for real control.

Real policy mode requires the sensing owner to supply a JSON setup file. Paths
are relative to that file. `live_control/setup.example.json` is **intentionally
not verified** and has no guessed map-to-floor transform or latency. Replace its
nulls with measured values and confirm the specific camera, tag sizes/map/course,
camera-body mounting assumption and signs before setting the verification field.
The camera config currently in the repo is copied from a different drone; selecting
that file does not by itself establish calibration for your aircraft.

```bash
python -m live_control.bridge --real --allow-flight --policy-setup /absolute/path/to/verified-setup.json
```

Take off manually. Allow the estimator to settle, select a policy, then click Run
policy. The page explains rejected starts (missing/stale pose, uncertainty,
transform/route validation, etc.). Stop policy immediately cancels autonomous
intent; the next 20 Hz worker tick requests zero RC. Manual takeover does the same.
It does not instantly remove momentum or land; Land is a separate button. A policy
abort/completion requests landing and cannot be silently resumed.

## Fault behavior and limits

- One worker writes all SDK commands; HTTP handlers only change its intent.
- A manual movement renewal expires after 0.3 seconds. A held button renews it.
- Browser heartbeat loss for one second, telemetry older than 0.5 seconds, decoded
  camera frames older than one second, low battery, a worker gap over 0.25 seconds,
  or the 90-second session cap requests landing and latches autonomous movement off.
- Button release, pointer cancel, focus loss and hidden tab cancel movement/policy.
- Landing has separate request and acknowledgment fields. Touchdown is always
  unconfirmed here. Failed radio delivery is possible. No endless RC keepalive
  continues after a landing request. No new takeoff is automatically retried.
- Real frame timestamps are decode timestamps minus an operator-supplied measured
  video-delay upper bound. State age is receipt age: Tello state has no common
  capture timestamp. These do not establish measured end-to-end response latency.
- Tag loss retains the accepted fix's position/time; fresh video never relabels an
  old position as new. Filter covariance supplies uncertainty. Bad/no tag coverage
  can prevent or abort a run. No world geofence is claimed for manual mode.
- HTTP is loopback-only with origin/Host checks and a per-launch command token.
  Another browser cannot seize the operator session. A public/team viewing URL
  would require a separate authenticated relay/video transport; none is exposed.
- No real flight or rendered-browser visual QA was performed in this workspace.

## Check the package

```bash
python -m unittest live_control.test_bridge live_control.test_policies -v
node --check live_control/app.js
```

The SDK protocol reference is Ryze's [SDK 2.0 guide](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf).
Policy integration was reviewed against teammate sources on master at
`dd419df` (RacePolicy and simulator guard work). `connect-trial` PR #2 is separate
test code and is not launched by this bridge.
