# Shared policy contract v1

## Coordinates and clocks

Canonical `frame = course_flu_v1`: right-handed world X from A toward B, world Y
left of that direction, world Z up from the ground. Positions are metres,
velocities metres/second. Yaw is radians counterclockwise from world +X.
Camera optical axes are right/down/forward; those are not the world axes.

`Positioner.locate()` returns `xyz` and `R` as camera pose in the origin-tag map.
The sensing owner must provide a verified proper rigid transform:

`p_world = R_world_map @ p_map + t_world_map`

`R_world_map` must be orthonormal with determinant +1. `t_world_map` includes the
height of the origin tag relative to the floor. No automatic axis swap or guessed
tag height is used by the adapter. Hand-carry the unpowered drone forward, left,
and up and rotate it left; verify the four canonical signs in the logs. Verify
pose scale against a tape measure and the BLACK tag-square edge measurement.

The adapter assumes camera +Z is drone forward and camera position approximates
vehicle position. Measure/correct camera-body extrinsics if those approximations
are not covered by the agreed clearance margin.

`now_s`, pose capture time, telemetry capture time, and learned action generation
time use the SAME monotonic clock. Account for camera transport/decode delay in
the pose timestamp; do not timestamp an old image as fresh on each control tick.
Repeated frame timestamps are allowed temporarily, but do not advance arrival dwell.
Missing pose: keep its ORIGINAL timestamp until it becomes stale. Never refresh
a missing fix with a new timestamp or invent a zero velocity / covariance.

## Observation fields

| Field | Producer | Meaning |
|---|---|---|
| `schema_version` | All adapters | Integer 1 |
| `frame` | Sensing | Literal `course_flu_v1` |
| `pose_time_s` | Sensing | Effective image capture time, including known transport delay |
| `telemetry_time_s` | Flight runner | Effective telemetry sample time |
| `position_w` | Sensing/estimator | Three finite world coordinates |
| `velocity_w` | Estimator | Three finite world linear velocities |
| `yaw_rad` | Sensing/estimator | World heading, CCW positive |
| `position_std_m` | Estimator | Conservative maximum position standard deviation over axes |
| `velocity_std_mps` | Estimator | Conservative maximum velocity standard deviation over axes |
| `battery_pct` | Flight runner | Finite 0..100, not unknown or substituted |
| `link_ok` | Flight runner | Boolean based on current transport/telemetry state |
| `airborne` | Flight runner | Boolean; not inferred by this policy |
| `enabled` | Operator/runner | False by default; explicit session enable |

Failure or uncertainty must be represented truthfully. The existing upstream
training log is NOT this schema; its `tag`, `vel`, `fresh`, and wall-clock `t`
must not be re-labelled as a canonical observation without a validated conversion.

## Output

`Decision.rc = (right, forward, up, yaw_clockwise)`, four SDK-scale integers.
Positive means right strafe, forward, up, and clockwise yaw. SDK range is +/-100;
this prototype applies a smaller +/-20 limit and a fractional slew limit.
The horizontal right/forward pair is scaled together to magnitude 20 before
rounding, and shares one slew interpolation fraction. Rounding can increase the
vector norm by at most sqrt(0.5); it does not change the individual +/-20 cap.
Zero RC requests stabilization; it does not guarantee zero physical velocity.

Fresh large heading errors enter `brake` while measured horizontal speed exceeds
the example turn-entry threshold. This opposes measured velocity, then enters
`aim` at low speed. It is not a proven stopping-distance or collision guarantee.
Optional `heading_targets_w` supplies one look-at landmark per waypoint. `None`
preserves fixed +X heading and backward return. The landmark candidate is for
review; sensing must prove whole-maneuver coverage before any deployment choice.
The same heading/braking and horizontal vector guards wrap learned action packets.

`mode`, `reason`, `waypoint_index`, and `request_land` must be logged with each
command. A landing request is an instruction to the flight runner; it is not proof
that a land command was transmitted, acknowledged, or completed. During link loss,
the runner cannot assume it can deliver RC or land. It must latch abort and land
when communication is recovered, consistent with the existing lifecycle.

An abort stays latched. Do not construct a new policy to bypass an abort in flight.
After landing and review, explicitly create a new session. Takeoff, landing SDK
I/O, acknowledgment, manual intervention, and transport watchdog belong to the
flight runner. Exactly ONE component may write RC commands for a given drone.

## Timing and current example limits

Target flight runner rate: 20 Hz. A gap above 0.2 seconds aborts. Pose older than
0.25 seconds holds at zero RC; pose older than 0.75 seconds requests landing.
Continuously delayed frames that remain over the freshness limit for 0.5 seconds
also request landing; repeated but persistently late updates cannot hold forever.
Telemetry older than 0.25 seconds, low battery, excess uncertainty, invalid data,
or outside-geofence observation requests landing. These are prototype parameters,
not validated Tello operating specifications. Set limits from measured logs.

An independent 90-second session deadline begins at the first airborne
observation. Disabling/re-enabling the policy does not reset it. It aborts a
stalled flight instead of waiting indefinitely. Change this example bound only
after agreeing on actual course/vehicle limits with the flight owner.

## RL handoff

`ActionSample` carries four normalized values in [-1,1], a model ID, the exact
action-order tuple, canonical frame ID, action generation time, and the timestamp
of the pose used to infer that action. It is produced asynchronously. It must be
at most 0.1 seconds old, not future-dated, and match the current pose and waypoint
index. The constructor pins an explicit `expected_model_id`; packets for another
checkpoint are rejected. Stale or
invalid packets abort the learned mode; there is no silent policy switch in flight.

The RL owner must supply a model manifest with:

1. Training git SHA, algorithm/checkpoint identity and file hash.
2. Ordered observation features, units, frames, normalization statistics and clip rules.
3. Action semantics, order, range, controller rates, and fixed-course-heading compatibility.
4. Simulator dynamics/noise/delay settings, evaluation seeds and completion evidence.
5. Exact preprocessing and deterministic-inference settings for deployment.

The current PPO scaffold concatenates 21 values: vectors to A (3), B (3), world
linear velocity (3), BODY angular velocity (3), BODY projected gravity (3), phase
one-hot (5; AT_START, OUTBOUND, TURNED, FINISHED, IDLE), elapsed lap time (1).
The current policy Observation does not supply all of these. Do not invent missing
angular rates or gravity. Either extend sensing with verified estimates and a
checkpoint-specific adapter, or retrain on the sensor-realistic shared interface.

The simulator's own controller dependencies must establish the exact four action
axes and signs. Identical vector length does not establish compatibility. The
new packet is a proposed deployment contract, not evidence the old PPO matches it.

## Lap timing

The repository timer uses map X/Z as horizontal and the simulator uses world X/Y.
Pass verified map coordinates into the existing timer, using the inverse rigid
transform when observations are canonical. Keep the timer authoritative for a
scored lap. Policy route completion alone is not a valid race result.
