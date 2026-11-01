# Sim-to-real and robustness plan for the Tello race policy

Status: **planning document, nothing here is implemented yet.** All `drone_race_sim/*` modules named below are *proposed* (see the Mac-sim plan, milestones M0-M6). Line references (`file:line`) point at the repo as of the merge commit; numbers marked *(unverified)* are proposals or external knowledge that no file in this repo confirms.

## 1. Purpose, scope, relation to the Mac-sim plan

The Mac-sim plan defines a vectorized PyTorch Tello simulator (`drone_race_sim/`: `dynamics.py`, `sensors.py`, `estimator.py`, `safety.py`, `randomization.py`, `curriculum.py`), a calibration protocol, an asymmetric-actor-critic PPO setup and a staged real-world protocol. It randomizes *nominal* uncertainty (gains, lag, noise, latency). This document adds the layer above that:

- a **catalogue of faults and off-nominal events** (not just parameter noise) and how each is injected in the sim;
- how to **schedule them in training** (curriculum, ADR, failure-seeking sampling) without producing an over-cautious policy;
- a **fixed benchmark** to measure robustness and gate releases;
- a **safe protocol to reproduce the same faults on the real drone** and feed the results back into the sim's severity ranges.

Out of scope: the nominal sim, the calibration fitting itself, and the engine's Linux-specific plumbing. This plan depends on M1-M3 of the Mac-sim plan (env, sensors, estimator, safety shim) and extends M3 (DR/curriculum).

### 1.1 Sim-to-real gaps this plan addresses

| Gap | Evidence in the repo | Mitigation here |
|---|---|---|
| Link drops within ~2 s of motors spinning (EMI) | `FLYING.md:123-131`; engine declares link lost after 1.5 s without 8890 packets (`core/tello_dual_video.py:373`), sends `rc 0 0 0 0` and lands on reconnect when `LAND_ON_LOSS` (`:57`, `:383-385`) | Split link faults into survivable blips (< 1.5 s) and void-attempt outages; F-C scenarios; truncation not failure (section 3.4) |
| Race rule: link loss = land, no dead reckoning | `docs/RACE_RULES.md:31` | Policy never rewarded for continuing blind; "uninterrupted lap" is a separate metric |
| Narrow camera cone, tags often not visible | `config/camera/default.yaml` (about 55 deg x 43 deg FOV, per Mac-sim plan); `config/tags.yaml:14-22` gate (`min_side_px 12`, `max_view_angle_deg 90`) | T-series tag-loss scenarios; fault-aware observations |
| Engine "fresh" window and search override tuned for approach, not race | `fresh = age <= 0.4` (`:421-428`), `SEARCH_AFTER_S = 0.4` (`:65`, `:507-516`) | Racing-specific threshold in the safety shim; scenarios SW-3, T-1 test the interaction |
| Loop slower and more jittery than 20 Hz | `stop.wait(0.05)` after tick work (`:629`); `tools/rc_rate_bench.py:1-14`; Kalman self-test dt 0.074 s | SW-1, SW-2 loop-rate drops and stalls; policy observes its own dt |
| Engine cannot see race tags 1/2 | `core/tello_tags.py:18-25` (origin 10, ids 5/6/7/10); legacy `TAG_ID = 14` (`:69`); `config/race_course.yaml:3-4` uses 1 and 2 | Deployment integration item (section 6); ghost/wrong-ID scenarios T-6 |
| Systematic range error (hard-coded focal 871 vs calibrated 914.5) | `CAL_F_AT_960 = 871.0` (`:71`) vs `config/camera/default.yaml` | S-8 range scale error; part of nominal DR, stressed here |
| Training log hides faults (fudged height, no packet age/tag id/R) | height forced to 0.5 when flying and raw low (`:441-442`); train row fields (`:614-627`) | Logging requirements (section 5.3) |
| No fault injection, noise or latency in the Isaac scaffold; privileged obs; lap update inside obs step | `tello_race_env_cfg.py:26,35` (60 Hz), `tello_race_env.py:155` | New sim only; do not port the scaffold's ordering |
| Lap logic depends on a pose estimate that can flip or drop | `race/lap_timer.py:87-130`; KF gate `positional_kalman_filter.py:44-54` | Estimate-driven vs truth-driven timer comparison (metric M-7) |

## 2. Scenario catalogue

### 2.1 Conventions

- **ID prefixes:** `P` path/pose disturbance, `T` tag/vision, `S` sensors, `C` connection, `A` actuation/dynamics, `E` environment/course, `SW` software/system, `X` compound.
- **Priority:** P0 = required before any real free flight; P1 = required before racing at speed / a full lap; P2 = nice to have.
- **Severity levels** (used in training ranges and the benchmark): `S0` nominal, `S1` mild (inside calibrated spread), `S2` hard (about 2 sigma outside), `S3` extreme (expected to cost the lap but must end safely). Ranges below give `S1 .. S3`; training samples between `S0` and the ADR-controlled level.
- **Injection point:** where in the sim pipeline the fault acts. Every fault is a *time window* `(t_start, duration, params)` drawn at reset (or triggered mid-episode), applied through per-env tensor masks.
- **Expected behaviour** describes what a good policy should do, used to define pass/fail.
- All parameter ranges are **proposals**; those depending on real hardware are re-fit from flight data (section 5.4).

### 2.2 P: steering the drone off the optimal path

Purpose: the policy must recover from states the PPO exploration would rarely visit, and from the states real flights will produce (wind, delay, estimator jumps).

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| P-1 | Spawn pose/velocity | Hand-launched start, takeoff drift, after a search sweep | `course.spawn()`: position offset from gate A up to 0.6-1.5-3.0 m, heading uniform 0..180/360 deg, initial velocity up to 0.3-0.6-1.0 m/s, height 0.5-1.5 m | Re-acquires gate, completes lap | lap success, lap time penalty vs nominal | P0 |
| P-2 | Position kick (mid-episode) | Gust, collision with a person, estimator reset | `faults.kick_position`: instantaneous displacement 0.3-1.0-2.0 m at random time, direction uniform | Replans toward next gate; no overshoot into pole | recovery time (back within 0.5 m of nominal line), max deviation | P0 |
| P-3 | Velocity kick | Gust, propwash, push | `faults.kick_velocity`: delta-v 0.3-0.8-1.5 m/s, 1-3 per episode, rate-limited | Damps out within 2 s | settling time, pole proximity | P0 |
| P-4 | Heading kick | Yaw drift, bump | delta-yaw 20-60-120 deg | Re-aims; no spin loop | recovery time | P0 |
| P-5 | Action override episodes | Safety shim takes over (search sweep, hover, cap) then hands back | `faults.override_action`: replace policy action with (hover / search sweep per `SEARCH_YAW=25`, `SEARCH_LEG_S=2.0` at `:66-67` / random) for 0.5-3 s, then release | Resumes from arbitrary state after override | recovery time, no oscillation at handover | P0 |
| P-6 | DART noise in executed actions | Actuation noise, imitation of imperfect pilots | Gaussian noise on rc before dynamics, sigma 0.05-0.15-0.3 (normalized), temporally correlated (OU, tau 0.3 s) | Same lap time within 10% at S1 | lap time degradation | P1 |
| P-7 | Off-line start | Wrong course side | spawn behind/beside gate A, facing away | Turns around, enters gate A cleanly | lap success | P1 |
| P-8 | Mid-lap respawn at random phase | Covers phases of `LapPhase` (`race/lap_timer.py:25-29`) | spawn into AT_START / OUTBOUND / TURNED with consistent timer state | Completes remaining legs | per-phase success | P1 |
| P-9 | Near-pole spawn | Drifted toward obstacle | spawn 0.3-0.6 m from pole, velocity toward it 0-0.5 m/s | Brakes/sidesteps; no collision | collision rate | P1 |

Note: P-2..P-5 are applied to the **true** state, not the estimate, so the estimator must also recover (it sees the jump only through the next tag fix or the packet velocity).

### 2.3 T: tag and vision faults

Injection module: `sensors.TagSensor` (visibility, detection, pose noise) with fault masks from `faults.py`. "Hidden" means the detector returns no fix for that tag.

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| T-1 | Both tags hidden for N s | Tag outside the 55-deg cone after a turn, occlusion, drops | mask both tags for N = 0.5/2/5 s at random time (and always at least once per lap in stage 2+) | Short gap: keep flying on the estimator; long gap: stop translating, search (yaw sweep), re-acquire; **never** fly dead-reckoned into a pole | recovery time, collision rate, time-to-reacquire | P0 |
| T-2 | One tag permanently hidden (A or B) | Fallen tag, occlusion by another drone, bad facing | mask one tag for the whole episode | Complete the lap with the visible tag only, or fail safe (hover/land) when the turn cannot be confirmed; attempt marked void is acceptable, crash is not | safe-landing rate, success rate when still feasible | P1 |
| T-3 | Both tags never visible | Tag facing wrong (the example `config/map.yaml` has normals perpendicular to A-B; Mac-sim plan finding 4) | mask all | Policy must not fly blind; fall back to hover/land | no-crash rate, time-to-safe-state | P0 |
| T-4 | Partial occlusion | Pole, person, other drone | Random fix dropout bursts: Markov with p(drop) 0.3-0.5-0.8 per detection opportunity, mean burst 3-10-30 frames; plus size reduction (apparent side px * 0.5-0.8) | Degrades gracefully | success vs dropout rate | P1 |
| T-5 | Tag appears late | Entered the cone after a turn | hide a tag until a distance/time threshold | Search/steer toward last-known bearing | time-to-first-fix | P1 |
| T-6 | Wrong / ghost tag IDs | Wrong ID read, other course's tags, reflections | inject spurious fixes with a random ID at random pose (rate 0.01-0.05-0.2 per s) and swap-ID events; estimator must reject by chi-square gate (`gate_chi2 = 16.27`, `max_rejects = 3`, `positional_kalman_filter.py:50-52`) | Ignored or gated; no pole-hop in estimate | accepted-ghost rate, estimate error | P0 |
| T-7 | Pole map error | Tags moved, `config/map.yaml` stale ("remap on site", `docs/RACE_RULES.md`) | per-episode offset of believed pole position vs true: 0.1-0.3-0.6 m, plus tag normal error 5-20-40 deg | Gate pass still triggered by the *estimate-driven* timer; lap success under truth timer reported separately | estimate-timer vs truth-timer disagreement | P1 |
| T-8 | Glare / saturation | Low sun on a white tag | detection probability drops for views facing the sun: p *= 0.2-0.5 inside a random heading sector; added blur | Pick another approach angle if possible | success rate | P2 |
| T-9 | Motion blur | Fast flight, yaw rate | blur length = (speed or yaw rate) * exposure; the detection LUT in `camera.py` takes blur as an input | Policy limits yaw rate/speed when it needs the tag | tag-visible fraction vs speed | P1 |
| T-10 | Low light / noise | Dusk, shade | detection decision margin scale 0.5-0.8 (gate `min_decision_margin 10`, `config/tags.yaml:15`), noise sigma factor 1.5-3 | Same as T-4 | success rate | P2 |
| T-11 | Steep view angle | Approach along the tag plane | allow detections only up to 60/70/80 deg view angle; increase flip probability with angle | Prefer a frontal approach when it costs little | flip rate, success | P1 |
| T-12 | Ambiguity flips / outliers | Pose ambiguity at range (`positional_kalman_filter.py:5-6`) | with p 0.01-0.05-0.15 add displacement 0.3-1.0-2.0 m; bursts of 1-3 frames | Estimator gates; policy does not lurch | estimate jump size, max rc spike | P0 |
| T-13 | Frozen frame / frame drops | Decoder stall, CPU load (`tools/rc_rate_bench.py` models load) | repeat the previous frame's fix for 0.2-1-3 s (fix timestamps look fresh but data stale) | Fault indicator catches stale data (section 3.5) | accepted-stale-fix rate | P1 |
| T-14 | Systematic range scale error | 871 vs 914.5 focal (`:71`) | per-episode scale 0.95-1.05-1.12 on measured range | Not fly into poles from range bias | collision rate | P1 |

### 2.4 S: sensor faults (state packet, IMU, optical flow, height)

Injection module: `sensors.StatePacketSensor` and `estimator.RaceStateEstimator`. Packet fields follow the engine's parsing (`tello_dual_video.py:126-165`); units/quantization are *unverified* (Mac-sim calibration step E0).

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| S-1 | State packet freeze / dropout | Link glitch, 8890 stream stalls | hold last packet for 0.3/1/2 s (age increases); the engine's own link-lost threshold is 1.5 s (`:373`) | Uses `packet_age` indicator; reduce commands, rely on tags | recovery, overshoot | P0 |
| S-2 | Stale but plausible data | Out-of-order UDP, buffered burst | deliver packets delayed by 0.1-0.3-0.8 s in bursts | Same as S-1 | state estimate error | P1 |
| S-3 | Velocity bias / scale error | Optical flow bias on dirt/asphalt | bias 0.03-0.08-0.2 m/s, scale +/-15-30-50%, persisting per episode | Tag fixes dominate when visible; modest drift when not | position drift during T-1 | P0 |
| S-4 | Velocity spikes | Flow outliers, vibration | spikes of 0.5-1.5 m/s for 1-3 samples, rate 0.02-0.1 per s | Not over-react (low-pass in obs or learned) | action spike magnitude | P1 |
| S-5 | Height glitch / quantization | ToF/baro error; engine itself fudges height to 0.5 when low (`:441-442`) | quantize 0.01-0.1 m; glitch spike to 0 or +1 m for 1-3 samples; latch value for 1 s | Hold altitude, no climb/dive reaction | max height excursion | P0 |
| S-6 | ToF failure / ground mismatch | Reflective ground, tall grass | height reads constant offset 0.2-0.5 m, or drifting | Keep within soft bounds `min_height_m 0.4`/`max_height_m 2.5` (`config/race_course.yaml:8-9`) | height-bound violations | P1 |
| S-7 | Yaw drift / gyro bias | IMU bias, magnetic anomaly | yaw rate bias 0.2-1-3 deg/s; heading step 10-30 deg | Re-anchor on tag fix (heading complementary filter) | heading error, recovery | P0 |
| S-8 | Optical-flow failure over featureless ground | Wet/uniform dirt, low light | velocity measured = 0 or noise-only for 1-5 s while true speed continues; drone slides under wind | Rely on tags; slow down | drift, collision rate | P1 |
| S-9 | Pitch/roll bias | Bad IMU calibration | camera pitch error 3-8-15 deg (changes visibility geometry) | Tolerates a shifted image | tag-visible fraction | P2 |
| S-10 | Timestamp jitter | Receiver thread contention | per-sample timestamp jitter 10-50-150 ms | Observation uses its own arrival times | filter innovation | P1 |
| S-11 | Latency spikes (video/state) | WiFi retries, CPU stall | extra latency 0.2-0.5-1.0 s for 0.5-2 s | Does not oscillate on stale info | oscillation amplitude | P0 |

### 2.5 C: connection faults

Two regimes, because the engine treats them differently. **Regime 1 (blips):** gaps under 1.5 s, the engine keeps flying and the policy must cope. **Regime 2 (outage):** at least 1.5 s of no packets: the engine sends `rc 0 0 0 0`, sets `abort` and lands on reconnect (`:373-397`), and `docs/RACE_RULES.md:31` says do not continue the lap. Regime 2 is a **void attempt**, not a policy failure.

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| C-1 | rc packet loss (random) | UDP drops (`send_rc`, `:114-115`, is fire-and-forget) | p 0.01-0.05-0.15 per tick, previous rc persists | Smooth | lap time degradation | P0 |
| C-2 | rc loss bursts | WiFi interference | bursts of 3-10-30 ticks (0.15-1.5 s); drone holds last command (or auto-hover: *unverified*) | Continuing last velocity is dangerous: policy keeps actions small near poles | collision rate | P0 |
| C-3 | Action delay increase | WiFi retries | extra delay 0.05-0.2-0.5 s | Lower gain, no limit cycle | oscillation, lap time | P0 |
| C-4 | Video dropout | Video link separate from control | no frames for 0.3-1-3 s while state packets continue (tags unseen, state ok) | Behaves as T-1 plus fault flag | recovery time | P0 |
| C-5 | State-vs-video desync | Different latencies, clock drift | video latency - state latency offset 0.1-0.3-0.6 s | Not trust fused estimate blindly | estimate error | P1 |
| C-6 | Packet reordering / duplication | UDP | reorder 1-3 packets | Robust parser | NaN/negative-dt count | P1 |
| C-7 | WiFi disassociation, short | Dongle/AP hiccup | whole link down 0.5-1.4 s, then back (below the 1.5 s threshold) | Resume; no jump command on return | action jump after reconnect | P0 |
| C-8 | Link loss with reconnect (outage) | EMI after motors spin (`FLYING.md:123-131`) | outage 1.5-3-8 s, Poisson hazard 0-0.05-0.2 /s; at onset engine emits `rc 0 0 0 0` and aborts | Not applicable to policy; episode **truncated as void**; shim lands. Evaluate "fraction of laps without outage" as a hardware metric | outage rate, safe-landing rate | P0 |
| C-9 | EMI correlation | Drop probability rises with motor effort | hazard multiplier proportional to mean |rc|^2 (a mild incentive for smooth, lower thrust; *unverified* coupling) | Prefers moderate commands | hazard vs effort | P2 |
| C-10 | Wrong-dongle / interface binding failure | `SO_BINDTODEVICE` mismatch (`:92`) | out of sim scope; covered by engine tests | n/a | n/a | P2 |

### 2.6 A: actuation and dynamics faults

Injection module: `dynamics.TelloDynamics` parameter overrides (gain, tau, deadband, delay, saturation) as time-varying functions.

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| A-1 | Battery sag | Voltage drop over flight | gain *(1 - k*t), k so total 0.05-0.15-0.30 over the episode; lag tau * 1-1.3-1.8 | Adapts via history/feedback | lap time vs battery | P0 |
| A-2 | Gain/lag step change | Mode switch, temperature | step at random time: gain x 0.7-1.3, tau x 0.6-1.6 | Re-settles | recovery time | P1 |
| A-3 | Asymmetric damage | Bent prop, motor wear | per-axis gain asymmetry (left vs right, forward vs back) 5-15-30%, constant yaw/roll bias | Compensates drift | path deviation | P1 |
| A-4 | Steady wind | Outdoor field | world-frame wind 0-1.5-3 m/s scaled by `k_wind` 0.05-0.4 (Mac-sim plan) | Crabs into wind, keeps gate | deviation, lap time | P0 |
| A-5 | Gusts | Turbulence | OU gust sigma 0.3-0.8-1.5 m/s, tau 0.5-2 s; occasional 2 s step of 2-4 m/s *(S3)* | Recovers | recovery time | P0 |
| A-6 | Ground effect / propwash | Low altitude, direction reversals | up-gain jitter +/-10-20%, height-dependent below 0.5 m, 0.3 s lag bump after a reversal | Avoids low flight | height excursion | P2 |
| A-7 | Payload / mass change | Added battery/tape | gain x 0.85-0.7, tau x 1.3-1.8 | Same as A-1 | success | P2 |
| A-8 | Deadband shift | Firmware/calibration | deadband 4-12-20 (the PID code already creeps at 8 to beat it: `tools/tello_policy.py:150-154`) | Avoid dithering near zero command | action chatter | P1 |
| A-9 | Saturation / rate cap | rc cap lowered by the safety shim | cap 25/35/60/100 rc, slew limit | Same lap with lower speed; train with cap sampled | lap time vs cap | P0 |
| A-10 | Takeoff drift / start height error | Takeoff settles at 0.5-1.4 m | start height N(0.9, 0.3) | Corrects height | height excursion | P1 |

### 2.7 E: environment and course

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| E-1 | Pole spacing/bearing | "remap on site" | spacing 3-7 m (default 4 m), bearing +/-45 deg, per episode | Generalizes | success vs geometry | P0 |
| E-2 | Pole height / tag height | Different poles | tag height 0.9-1.5 m; gate heights per `config/race_course.yaml` | Adjusts altitude | success | P1 |
| E-3 | Tag facing | Facing perpendicular to the A-B line (Mac-sim finding 4) | normal orientation +/-45 deg, and a "worst" setting where B is seen at 80-90 deg | Chooses an approach path with visibility or reports infeasible | feasibility rate, success | P0 |
| E-4 | Static obstacles | Cones, tripods, catcher | cylinders radius 0.1-0.3 m at random positions, terminate on contact; unobservable in actor obs at first (tests only safety margins) | P2: not required to avoid unseen obstacles; test shim stop | collision rate | P2 |
| E-5 | Other drones / people | Hack-night crowd | moving cylinder crossing the course 0.5-1.5 m/s; also occludes tags (T-4) | Hover/avoid when safety shim triggers | near-miss rate | P2 |
| E-6 | Boundary breach | Drift or wind | out-of-bounds radius from course midpoint 8 m (scaffold) tightened to 5-6 m; shim must stop at boundary | Safe return or land | boundary violations | P1 |
| E-7 | Uneven lighting | Shadows across tags | per-tag detection multiplier 0.3-1.0 | See T-8/T-10 | success | P2 |
| E-8 | Terrain / low-texture ground | Dirt field, asphalt | links to S-3/S-8 | See S-series | drift | P1 |

### 2.8 SW: software and system faults

| ID | Perturbed | Real-world cause | Injection | Expected behaviour | Metrics | Pri |
|---|---|---|---|---|---|---|
| SW-1 | Control loop rate drop | GIL contention with decoders/recorder/detector (`tools/rc_rate_bench.py:1-14`) | loop period 0.05 -> 0.08-0.12-0.20 s sustained for 2-10 s | Uses observed dt; no windup | action smoothness | P0 |
| SW-2 | CPU stall | GC, disk flush | single tick delay 0.3-1.0 s (policy not called, last rc held) | Hold safe; recover | overshoot | P0 |
| SW-3 | Search override hijack | Engine search triggers at `SEARCH_AFTER_S = 0.4` (`:65`, `:507`) while racing | shim threshold 0.4 s vs 2-3 s; train with the shim in the loop at the deployment threshold | Policy and shim are consistent (policy knows search takes over) | time in search, lap time | P0 |
| SW-4 | NaN / inf / out-of-range inputs | Parser error, missing field | with p 1e-4 per tick set a random obs element to NaN/inf/1e6 | Shim sanitizes (clip, replace with last-valid, hover on repeated failure); the policy sees finite values only | NaN-output count (must be 0) | P0 |
| SW-5 | Estimator divergence / reset | KF gate rejects too many fixes then resets (`max_rejects = 3`, `lost_after = 0.4`: `positional_kalman_filter.py:52-54`) | force reset with covariance inflation at random times; random initial offset 0.5-2 m | Recovers on next fix | recovery time | P0 |
| SW-6 | Wrong units / frame | cm vs m, y-up vs z-up (Mac-sim finding 3) | **test, not training**: replay logged data through the observation builder with swapped units; unit tests assert shape/range of every obs dimension | n/a | unit tests pass | P0 |
| SW-7 | Out-of-distribution detection | Policy faces unseen state | online: Mahalanobis distance of obs to training stats (or ensemble disagreement); shim switches to hover/land above threshold | Safe fallback before a crash | OOD trigger precision/recall | P1 |
| SW-8 | Policy file/runtime error | ONNX load/inference exception | engine catches, hovers, lands | Fails safe | n/a (integration test) | P1 |
| SW-9 | Clock jumps | NTP, suspend | dt negative or huge | Clamp dt in estimator | no NaNs | P1 |

### 2.9 X: compound and adversarial cases

| ID | Composition | Why | Pri |
|---|---|---|---|
| X-1 | T-1 (both tags hidden 2 s) + A-4 (wind 1.5 m/s) + S-11 (latency spike 0.5 s) | Realistic worst-hour combination | P0 |
| X-2 | C-2 (rc burst loss) + A-5 (gust) + P-3 (kick) | Control-side stack | P1 |
| X-3 | S-3 (velocity bias) + T-1 + E-8 (featureless ground) | Estimator drift stack | P1 |
| X-4 | T-7 (pole map error) + T-12 (flips) + C-5 (desync) | Pose/lap-logic stack | P1 |
| X-5 | A-1 (battery) + SW-1 (loop drop) + C-3 (delay) | Slow/lagging loop stack | P1 |
| X-6 | Random k-of-n composition: sample 2-4 faults from the catalogue at S1-S2 each | Coverage | P0 |
| X-7 | **Adversarial search:** optimize fault parameters (timing, severity) to minimize policy return (CMA-ES or random search over the schedule vector, 200 candidates per round on the fixed seed set). Top 1% failures go to the failure-seed replay buffer (section 3.3) | Finds holes randomization misses | P1 |

## 3. Training integration

### 3.1 Extending the curriculum

The Mac-sim plan defines stages 0-4 (perfect state, dynamics DR, sensors/estimator, full DR + wind + random spawn + link loss, higher rc cap). Add robustness stages after stage 3 and keep earlier stages as a mixture:

| Stage | Adds | Fault mix per episode | Advance when |
|---|---|---|---|
| 3 (existing) | Full nominal DR, wind, random spawn yaw, geometry DR | none beyond DR | success >= 80% over 2000 eps |
| R1 | P-1..P-5 (spawn/kicks/override), C-1/C-3, A-1/A-9 | 40% nominal, 60% one S1 fault | R1 success >= 70% on benchmark, no collision rate > 2% |
| R2 | T-1/T-4/T-6/T-12, S-1/S-3/S-5/S-7/S-11, C-2/C-4/C-7, SW-1/SW-2/SW-4/SW-5 | 30% nominal, 50% one fault S1-S2, 20% two faults | meets thresholds (section 4.4) at S1, >= 50% success at S2 |
| R3 | Compound X-1..X-6, remaining P1/P2 faults, E-series geometry/tag facing | 25% nominal, 35% one, 25% two, 15% three | thresholds at S2 for the P0 set |
| R4 | ADR on all severity ranges + failure-seed replay (X-7) | 20% nominal, rest ADR-controlled | benchmark plateau; no regression on the nominal set |

Rules: (a) earlier stages keep 20% of environments always sampling the earlier level (anti-forgetting, as in the Mac-sim plan); (b) **always keep at least 20-25% purely nominal episodes** in every stage so the lap-time objective is not traded away (section 3.7); (c) the rc cap is a randomized quantity (A-9) rather than a stage constant after R1.

### 3.2 Scheduling faults

- **Per-episode parameters** (persist all episode): S-3 bias, T-7 map error, T-14 scale, A-3 damage, A-4 wind mean, E-1..E-3 geometry. Sampled at reset.
- **Mid-episode events** (windows or impulses): T-1, T-13, S-1, S-5, S-7 steps, C-2/C-4/C-7, P-2..P-5, A-2/A-5, SW-1/2/5. Each env holds a `FaultSchedule`: up to `K = 6` events `(type, t_start, duration, severity)` drawn at reset; events can overlap. This keeps the vectorized step branch-free: all events evaluate to boolean masks of shape `(E,)`.
- **Event timing:** uniform over the active lap window, with a bias toward critical moments (just after leaving gate A, during the turn at B, within 1.5 m of a pole): 50% uniform, 50% critical-biased.
- **Probability schedule:** per-fault activation probability ramps linearly with `adr_level` from 0 to its target `p_max` (default 0.15 per fault family per episode, up to 4 faults).
- **ADR (automatic domain randomization):** each severity parameter keeps a range `[lo, hi]` widened by `delta` when the rolling success rate on episodes containing that fault family is >= 85%, narrowed when < 55%. One range per (family, parameter). Logged to TensorBoard per family. ADR operates only on parameters flagged `adr: true` in the scenario yaml (section 6.3).
- **Failure-seeking sampling:** after each evaluation round, episodes with failure (collision, out-of-bounds, timeout while a fault was active) store `(env seed, fault schedule, geometry sample)` in a buffer (cap 20k). 10% of resets in R4 replay a stored failure with a small perturbation of the schedule. Adversarial search (X-7) feeds the same buffer.
- **Seed hygiene:** the benchmark seeds (section 4.1) are never inserted into training sampling.

### 3.3 Reward and termination for recovery

- **Do not penalize unavoidable faults.** While a fault flagged `unavoidable` is active (e.g. both tags hidden by T-1 with no fix available), suppress the visibility penalty and the time penalty is still applied (lap time is real) but the *progress-shaping* term uses the estimator's belief, not truth, so there is no incentive to peek at privileged information.
- **Recovery bonus:** for each disturbance event (P-series, T-1, S-7, etc.) record the "pre-event" deviation `d0` from the nominal line; give a one-time `+r_rec * clip((d_max - d_t)/d_max, 0, 1)` when the drone returns within `0.5 m` (or `15 deg`) after the event, scaled by `1/(1 + t_recovery)`. Keep `r_rec` small (1-2 reward units vs +5 for hitting B and +10 for finishing: Mac-sim reward design) so recovery is not farmed.
- **Anti-farming:** limit recovery bonuses to one per event and zero for faults that were the agent's own doing (deviation exceeding `d0` without an active fault).
- **Hard penalties unchanged:** collision, ground contact, boundary breach stay at the Mac-sim values; faults never excuse them. Rationale: a crash during a fault is what the real drone will do.
- **Link loss (C-8) is a truncation, not a terminal failure:** bootstrap the value from the last state at truncation and add no negative reward; log it separately as `void_outage`. Rationale: the engine lands on reconnect (`:383-385`) and `docs/RACE_RULES.md:31` voids the attempt, so the policy cannot influence it. Sub-1.5 s blips (C-7) are **not** truncated.
- **Safe-landing success:** when a scenario marks "end safely is the goal" (T-3, T-2 infeasible, C-8, S-6), reward `+3` for reaching hover at height 0.4-1.5 m inside the boundary within 5 s and zero progress reward; reward comes from `safety` shim state, not the policy. This teaches a hover-and-wait habit without hijacking nominal races.
- **Penalty for unnecessary hesitation:** time penalty continues during recovery, preventing "hover forever" under faults.

### 3.4 Asymmetric critic: privileged fault flags

Critic-only additions to the Mac-sim critic (about 160 dims): active-fault one-hot per family (about 30 dims), fault severity scalars, time-to-fault-end, true estimator error (position/heading), true link state, current wind/dynamics parameters (already there), pending events. The actor never sees these; this gives faster value learning under faults without leaking labels.

### 3.5 Fault awareness in the actor observation

Add only quantities the **real stack can compute**, appended to the 25-dim frame (Mac-sim obs). Proposed (about +10 dims, each also logged by the engine):

| Indicator | Computation on the real drone | Detects |
|---|---|---|
| `fix_age_A`, `fix_age_B` | time since last accepted fix per tag (engine already tracks `last_fresh_t`, `:351`, `:445`) | T-1, T-3, C-4 |
| `packet_age` | time since last 8890 state packet (`link["t"]`, `:373`) | S-1, C-7 |
| `kf_reject_count`, `nis` | rejected fixes in a row and normalized innovation (`gate_chi2`) | T-6, T-12 |
| `pos_std` | KF position std (`position_variance`) | drift (S-3, S-8) |
| `vg_vs_tag_resid` | packet velocity minus tag-derived velocity when both available | S-3, S-4, S-8 |
| `loop_dt`, `rc_send_ok` | measured period; send errors | SW-1, SW-2, C-1 |
| `height_resid` | height vs integrated vertical velocity over 1 s | S-5, S-6 |
| `frozen_frame` | identical frame hash / detection pose within epsilon for N frames | T-13 |
| `rc_cap` | current cap from the shim | A-9 |

Rule: an indicator enters the observation only if the sim computes it from the same inputs and code path as the engine (`estimator.RaceStateEstimator` is shared sim/deploy in the Mac-sim plan). Indicators are standardized with running stats; during training they are present in 100% of episodes, including nominal ones, so their *absence of change* is also informative.

### 3.6 History vs recurrence

Keep the Mac-sim choice (frame stack 4 + last 3 actions, MLP) for R1-R2. Re-evaluate a GRU at R3 if either (a) T-1 5 s gaps have recovery time > 2x S1 gaps despite the indicators, or (b) S-3/S-8 drift cases need longer memory of the last good fix than 4 frames (0.2 s). A cheap alternative first: extend history to 8 frames and add `time-since-last-fix` and `position at last fix in body frame` as explicit inputs. A GRU adds ONNX state I/O, a hidden-state reset policy after link loss, and more tests, so it is a measured decision, not a default.

### 3.7 Avoiding over-conservative policies

- Nominal share >= 20-25% in every stage (section 3.1).
- Evaluate lap time degradation at S0 *every* evaluation; a stage fails if nominal median lap time worsens more than 10% against the best stage-3 checkpoint.
- Do not stack faults in the reward: severe faults give less shaping, not more caution bonuses.
- Entropy schedule: keep `ent_coef` at 0.003 floor during R-stages; a collapsing entropy usually precedes brittle recovery.
- Report Pareto plot (lap time at S0 vs success at S2); select checkpoints on the Pareto front, not best-robustness-only.
- A safety shim cap that is too tight (A-9) caps the policy's lap time on purpose; benchmark lap time at the *same* cap as the PID baseline.

### 3.8 Interaction with the safety shim and fallback policy

- The shim runs **inside** the training loop (Mac-sim plan section 4) at the deployment thresholds: action cap and slew, search override (racing threshold 2-3 s of no tags, not the engine's 0.4 s, `:65`), height guards, link-loss handling, NaN sanitation (SW-4).
- The policy gets a `shim_active` flag (observation) and the shim's action is the executed action; the buffer stores both the policy action and the executed action. PPO updates on the policy action only when `shim_active == False`; otherwise the transition is masked from the policy-gradient loss (but still used for the critic), so the policy is not trained on actions it did not control.
- **Fallback policy:** `engine_replica`/`race_pid` baselines (Mac-sim plan section 8). When the OOD detector (SW-7) or repeated sanitation triggers, the shim hands control to a deterministic hover-then-land routine, **not** to the PID approach. The PID baseline is an evaluation reference and an optional warm start, not a flight fallback.
- Test both directions of handover (P-5): the policy must not produce a command spike on the first tick after control returns.

### 3.9 Regression suites

- **Unit tests** per fault: each injector applied to a known trajectory produces the specified mask/magnitude/duration (`tests/test_faults.py`).
- **Determinism:** same seed + same schedule yields identical rollouts.
- **Benchmark regression:** every checkpoint that is a release candidate runs the full suite (section 4) and compares to the last accepted checkpoint; a drop beyond tolerance in any P0 scenario blocks release.
- **Nominal guard:** S0 lap time and success (section 4.4).
- **Failure-seed suite:** the replay buffer's top failures are re-run each release; the count of still-failing seeds must not increase.

## 4. Evaluation

### 4.1 Benchmark suite

`benchmarks/robustness.yaml` defines scenarios x severities x seeds:

- Scenarios: all P0 and P1 items from section 2 plus X-1..X-6 (about 45 scenarios).
- Severities: `S0, S1, S2, S3` (S0 once per seed set as nominal control).
- Seeds: 64 fixed seeds per cell (course geometry, spawn, noise streams); a held-out set of 64 more used only at release time.
- Cells: about 45 x 4 x 64 = 11.5k episodes, runnable in a few minutes at 4096 envs on the M-series (*throughput unverified*).
- Each cell runs with: trained policy, `race_pid` baseline (waypoint PID with the same safety shim), and a "hover-only" control (checks that scenarios are not trivially "pass").

### 4.2 Metrics (per episode, aggregated per cell)

| ID | Metric | Definition |
|---|---|---|
| M-1 | Lap success | truth timer reaches `FINISHED` (`race/lap_timer.py:29`) without crash, within 30 s; separate rate for estimate-driven timer |
| M-2 | Lap time degradation | median(lap time at severity) / median(lap time at S0) |
| M-3 | Recovery time | seconds from fault end/impulse until deviation < 0.5 m and heading error < 15 deg |
| M-4 | Max deviation | max distance from the nominal line during/after the fault |
| M-5 | Crash cause histogram | pole, ground, ceiling, boundary, lost-tag timeout, timeout |
| M-6 | Safe-landing rate | for "end safely" scenarios: reaches hover within bounds, no crash |
| M-7 | Timer disagreement | rate where truth and estimate-driven timers disagree on lap/false-start |
| M-8 | Action roughness | mean |delta a|, max rc spike, saturation fraction |
| M-9 | Void-outage rate | fraction of episodes truncated by C-8 (reported, not penalized) |
| M-10 | OOD trigger quality | precision/recall of the OOD detector vs actual failures |

### 4.3 Reporting

Per-scenario table (rows = scenario, columns = S1/S2/S3 x {success, lap time ratio, recovery time, crash cause}); a heat map across the grid; delta vs previous accepted checkpoint; `race_pid` column alongside for comparison. Reports are generated by `eval/robustness_report.py` to `reports/<ckpt>/robustness.md` and TensorBoard scalars `robust/<scenario>/<severity>/<metric>`.

### 4.4 Proposed pass/fail thresholds (to be revised after real data)

| Gate | Condition |
|---|---|
| Nominal (S0) | success >= 95%; median lap time <= 0.9 x `race_pid` median |
| P0 at S1 | success >= 90% (P-series, A-series), >= 80% (T/S/C series); crash rate <= 2% |
| P0 at S2 | success >= 60%; crash rate <= 8%; safe-landing rate >= 95% for end-safely scenarios |
| P0 at S3 | crash rate <= 20%; safe-state reached >= 80% when the lap is infeasible |
| Recovery | median M-3 <= 2.5 s at S1 impulses; <= 5 s at S2 |
| Link/void | C-8 handled by shim: 100% safe landing in simulation |
| NaN | zero NaN actions across the whole suite (SW-4) |
| Compound | X-1..X-6 at S1: success >= 60% |
| Regression | no P0 cell drops by more than 5 points versus the last accepted checkpoint; S0 lap time within +10% |
| vs baseline | the policy beats `race_pid` success in at least 80% of P0 cells at S2 |

### 4.5 CI-style regression

A script `scripts/run_robustness_ci.sh` (on demand locally, nightly if a runner exists): unit tests, a reduced benchmark (16 seeds, S1 and S2 only, about 2-5 min), failure-seed replay, and a throughput check. The full benchmark runs at release candidates. Output is a single `PASS/FAIL` with the failing cells.

## 5. Sim-to-real validation

### 5.1 Principles

- Reproduce faults on the real drone with **software-side injection first** (no hardware risk), physical tests second, and only after the shadow-mode and tethered stages of the Mac-sim plan (S0-S1).
- Every real fault trial has a logged ground-truth fault schedule so results are comparable with the sim cell of the same ID.
- Nothing in this section raises the rc cap beyond what is calibrated (Mac-sim plan risks).

### 5.2 Engine-side fault injection shim (proposed)

A small module `core/fault_injection.py`, enabled only by an explicit CLI flag (`--inject faults.yaml`) and loud on start. It wraps the points where data enters/leaves the engine:

| Hook | Where in the engine | Faults it can inject |
|---|---|---|
| rc output | before `send_rc` (`:114`; flight call at `:597`, link-down hold at `:393`) | drop, delay (ring buffer), burst loss, rc noise (P-6, C-1..C-3, SW-2) |
| Tag fix input | where `poses.get(name)` is read (`:417-428`) | hide tag/id, freeze frame, add flips/ghost fix, pole map offset (T-1, T-3, T-6, T-12, T-13) |
| State packet input | where heights/velocity/yaw are read (`:126-165`, `:435-442`) | freeze, delay, bias/scale, spike, height glitch (S-1..S-7, S-11) |
| Loop pacing | `stop.wait(0.05)` (`:629`) | extra delay, stall (SW-1, SW-2) |
| Link state | around `connected = ...` (`:373`) | simulated blips below the threshold (C-7) |

Safety rules for injection: never inject C-8-style outages artificially (the dongle's real behaviour is what matters); injected faults never suppress the engine's own safety layers (`LAND_ON_LOSS`, height guard, emergency stop); an on-screen/console banner lists active faults; a kill switch (Ctrl-C and `pkill -f tello_link.py`, `FLYING.md:133-139`) ends the flight.

### 5.3 Logging requirements

Extend the train log (proposed engine patch, same as the Mac-sim plan's M0 spec): raw height (not the `0.5` fudge at `:441-442`), packet age and arrival time, full raw packet, tag id, pixel center/size, R, decision margin, tag age, loop dt, `t_send`, wifi state, rc as sent and rc as injected, fault ID/severity/active flag, shim state. Everything needed to recompute the section 3.5 indicators offline must be present, so a flight can be replayed through the sim's estimator.

### 5.4 Staged real-world protocol

| Stage | Setup | Faults | Go/no-go |
|---|---|---|---|
| R-S0 shadow (no takeoff) | carry drone by hand along the course, policy outputs logged only | T-1, T-3 (hide tags in software), S-1, S-5, S-7, SW-1, SW-4, T-6/T-12 (software), C-1..C-4 | no NaN, indicators fire when expected, estimator recovery within 2x sim, time between fault and indicator <= 0.3 s |
| R-S1 hover (tethered / catcher) | hover at 1 m, no course | A-9 (caps), C-1/C-3 (rc loss/delay), S-5 (height glitch in software), SW-2 stalls, P-3 small manual pushes | no oscillation growth, returns to hover within 3 s of each fault |
| R-S2 segments | A->B at cap 25 -> 40 | T-1 (software, 1-2 s), S-3 (velocity bias, software), C-2 bursts, P-2/P-4 kicks (software shift of the pose estimate), A-4 natural wind (record speed) | no pole contact, safe landing on every abort; no more than 1 void outage in 5 runs, else stop and fix hardware (EMI, antenna) |
| R-S3 laps | full lap at the calibrated cap | X-1, X-2 (software combos), natural wind/battery; physical tag occlusion (hand-held card) | success and lap time within the tolerance of the matching sim cells (section 5.5) |
| Physical-only tests | Catcher always ready | Physical tag cover (T-1/T-4), battery at 100/60/30% (A-1), different time of day (T-8/T-10), moved pole (T-7/E-3 with new map) | each recorded with weather and battery |

Safety rules (all stages): catcher ready (`docs/RACE_RULES.md`, `README.md:91`); battery >= 50% at start; wind < 3 m/s for flights beyond hover; one drone flying; abort criteria written on the run sheet (any pole contact, any unplanned oscillation, an outage); emergency stop procedure as in `FLYING.md:133-139`; no fault stacking beyond what the stage lists; stop a stage if the same fault fails twice.

### 5.5 Feeding real results back

For each scenario ID run in reality, compute the same metrics (M-1..M-9) from the logs and compare with the sim cell of matching severity:

- **Severity calibration:** adjust the severity ranges so the real outcome lies within the sim's [10th, 90th] percentile band for the cell; if the real outcome is worse than the sim's 10th percentile, widen that scenario's range (or fix the sim model) and retrain at the next R-stage.
- **Rate calibration:** measured fault rates (outage hazard, dropout rate, flip rate, loop-rate dip frequency) replace the guessed Poisson hazards/probabilities (C-8, T-6/T-12, SW-1); keep the ADR bounds at 1.5-2x the observed values.
- **Gap log:** `reports/sim2real_gaps.md` lists each scenario, sim vs real metric, decision (accepted / widened range / model fix needed).
- **Go/no-go for the next real stage:** the previous stage met its gate in the table above, no unexplained gap > 2x on any P0 metric, and the real void-outage rate is documented.

## 6. Implementation plan

### 6.1 New modules and files (all proposed)

```
drone_race_sim/
  faults.py        FaultSchedule (K events per env), injectors, severity scaling
  scenarios.py     scenario registry, yaml loader, family -> injector mapping
  adr.py           per-parameter range state, update rule, TensorBoard logging
  failure_buffer.py  stores failing (seed, schedule, geometry) and replays
  indicators.py    fault-awareness obs terms (shared with deploy via estimator)
  overrides.py     action-override and kick utilities (P-series)
  params/faults.yaml        default severities (S1..S3), families, adr flags
eval/
  robustness_suite.py       benchmark runner (scenario x severity x seeds)
  robustness_report.py      tables, heat maps, regression diff
benchmarks/robustness.yaml  fixed grid and seed lists
core/fault_injection.py     engine-side shim (section 5.2), flag-gated
tests/
  test_faults.py, test_scenarios.py, test_adr.py, test_indicators.py,
  test_fault_replay_determinism.py, test_engine_fault_shim.py
scripts/run_robustness_ci.sh
docs/ROBUSTNESS_PLAN.md     (this file)
```

### 6.2 Interfaces

```python
# drone_race_sim/faults.py
@dataclass
class FaultEvent:
    kind: str            # e.g. "tag_hide", "rc_burst_loss", "vel_bias"
    t_start: Tensor      # (E,) seconds since lap start
    duration: Tensor     # (E,)  (0 for impulses)
    severity: Tensor     # (E,)  in [0, 1] mapped through params/faults.yaml
    target: Tensor       # (E,) tag id / axis / sensor index

class FaultInjector:
    def reset(self, env_ids, level, rng) -> None: ...           # sample schedules
    def active(self, t) -> dict[str, Tensor]: ...               # per-kind (E,) masks
    # one hook per pipeline stage, each a pure tensor function
    def on_action(self, rc, dt) -> Tensor: ...                  # C-1..C-3, P-5, P-6, SW-2
    def on_state(self, true_state, dt) -> TrueState: ...        # P-2..P-4, A-*
    def on_tag_fix(self, fix, t) -> Fix: ...                    # T-*
    def on_packet(self, pkt, t) -> Packet: ...                  # S-*
    def on_clock(self, dt) -> Tensor: ...                       # SW-1, SW-2, SW-9
    def info(self) -> dict[str, Tensor]: ...                    # privileged flags for the critic
```

Pseudocode of the step order (fault hooks marked `*`):

```
rc_raw   = shim(policy(obs))                       # cap, slew, search override, sanitize
rc       = injector.on_action(rc_raw, dt)       *  # drop / delay / noise / override
true     = dynamics.step(rc, params(t))             # params include A-series faults
true     = injector.on_state(true, dt)          *  # kicks, damage steps
fix      = injector.on_tag_fix(tag_sensor(true), t)       *
pkt      = injector.on_packet(packet_sensor(true), t)     *
est      = estimator.update(fix, pkt, dt_measured)        # same code as deploy
ind      = indicators(est, fix, pkt, dt_measured)
obs      = obs_builder(est, ind, history)
reward, done = rewards(true, est, injector.info())        # critic-only flags
```

### 6.3 Scenario config schema (example)

```yaml
# params/faults.yaml  (excerpt; severities are proposals)
version: 1
families:
  tag_hide:
    id: T-1
    priority: P0
    applies_to: [tag_sensor]
    event: window                    # window | impulse | episode
    parameters:
      duration_s:  {S1: 0.5, S2: 2.0, S3: 5.0, adr: true}
      targets:     {value: [both]}   # [A] | [B] | [both]
      timing:      {mode: mixed, critical_fraction: 0.5}
    unavoidable: true                # suppress visibility penalty (3.3)
    end_safely: false
  rc_burst_loss:
    id: C-2
    priority: P0
    applies_to: [action]
    event: window
    parameters:
      burst_ticks: {S1: 3, S2: 10, S3: 30, adr: true}
      hold_mode:   {value: last_rc}  # last_rc | zero (open question 4)
  link_outage:
    id: C-8
    priority: P0
    applies_to: [link]
    event: window
    parameters:
      hazard_per_s: {S1: 0.02, S2: 0.05, S3: 0.2, adr: false}   # measure first
      duration_s:   {S1: 1.5,  S2: 3.0,  S3: 8.0}
    outcome: truncate_void           # not a policy failure
  vel_bias:
    id: S-3
    priority: P0
    applies_to: [packet]
    event: episode
    parameters:
      bias_mps:  {S1: 0.03, S2: 0.08, S3: 0.20, adr: true}
      scale_err: {S1: 0.15, S2: 0.30, S3: 0.50}

curriculum_mix:                      # fraction of episodes per stage (3.1)
  R1: {nominal: 0.40, one_fault: 0.60}
  R2: {nominal: 0.30, one_fault: 0.50, two_faults: 0.20}
  R3: {nominal: 0.25, one_fault: 0.35, two_faults: 0.25, three_faults: 0.15}
  R4: {nominal: 0.20, adr: 0.80}
```

### 6.4 Milestones (add to the Mac-sim M0-M6; one engineer, rough)

| Milestone | Content | Effort | Depends on |
|---|---|---|---|
| RB0 | Finalize catalogue/severity table; `faults.py` skeleton, `FaultSchedule`, masks, unit tests | 2 days | M1 (env) |
| RB1 | P-series, A-series, C-1..C-3/C-7, SW-1/2/4 injectors + stage R1 training + benchmark runner v0 | 3-4 days | M2 (PPO), M3 (sensors/estimator/shim) |
| RB2 | Tag/sensor injectors (T-1,T-4,T-6,T-12,T-13, S-1..S-7,S-11), indicators in obs, critic flags, recovery reward, stage R2 | 4-5 days | M3 |
| RB3 | Compound, E-series, ADR, failure buffer, adversarial search, stage R3-R4, full benchmark + report | 4-5 days | RB1-RB2 |
| RB4 | `core/fault_injection.py` engine shim, logging extension, shadow-mode tests (R-S0), replay of real logs through estimator | 3-4 days | M5 (deploy wrapper), M4 (real logs) |
| RB5 | Real fault trials R-S1..R-S3, sim-vs-real gap log, severity recalibration, retrain | ongoing | RB4, flights |

Total: about 3 weeks after the Mac-sim MVP; P0 scenarios alone (RB0-RB2 + R-S0) about 10 working days.

### 6.5 Risks

- **Fault space too large:** training signal diluted. Mitigation: P0 first, mix caps (section 3.1), ADR per family, benchmark-driven pruning.
- **Over-conservative policy** (section 3.7).
- **Sim faults unrealistic:** severity ranges rest on guesses until real logs exist; mitigated by the real-feedback loop (5.5) and conservative S3.
- **Link outages dominate real flights** (EMI, `FLYING.md:123-131`): robustness work cannot fix hardware; measure the outage rate early with `tools/tello_hover_test.py` and fix the dongle/antenna first.
- **Engine shim bugs mask or create faults:** the shim has its own tests (`test_engine_fault_shim.py`) and logs injected vs real values.
- **Vectorization cost:** K = 6 events per env with branch-free masks is cheap but adds many small tensor ops; check throughput at RB1 against the M0 gate (>= 30k steps/s, *unverified*).
- **Policy exploits sim fault models** (for example learning to hide behind simulated dropouts): check with held-out fault parameters in the benchmark's held-out seeds.
- **Engine integration gap** (tags 1/2 not known to the engine, Mac-sim finding 1) blocks real fault trials; it is on the deploy critical path.

### 6.6 Open questions

1. Real outage rate and duration after motors spin on the actual rig? (Decides C-8 hazard and whether laps are possible at all.)
2. Standard or EDU Tello? EDU accepts the video-shrink commands (`FLYING.md:123-131`), which changes video latency and dropout characteristics.
3. How does the Tello behave when `rc` stops arriving mid-motion (hold last command vs brake)? Needed for C-2 `hold_mode`. *Unverified.*
4. Units and quantization of the packet velocity/height fields (cm/s vs dm/s) and the packet rate. *Unverified.*
5. Which faults can physically be reproduced safely at the venue (cover a tag, move a pole, wind fans)?
6. Is an OOD detector acceptable in the flight path, or should uncertainty come only from the indicators?
7. Race-day expectations: is a void lap acceptable (a retry) or must the policy always finish?
8. Is the separate Action Space repo's tag-sensor/Kalman/motion-fit code available to calibrate several of these models?

## 7. Prioritized checklist

**P0: before any real free flight**
- [ ] Freeze catalogue and severity table (`params/faults.yaml`); review with the team.
- [ ] `faults.py` skeleton with `FaultSchedule`, masks and unit tests; deterministic replay.
- [ ] P-1..P-5 (off-path spawns, position/velocity/heading kicks, action overrides) and recovery metrics.
- [ ] T-1 / T-3 (tag loss), T-6 / T-12 (ghost tags, flips), S-1 / S-3 / S-5 / S-7 / S-11, C-1..C-4 / C-7 / C-8 (outage as void truncation), A-1 / A-4 / A-5 / A-9, SW-1 / SW-2 / SW-4 / SW-5.
- [ ] Fault-awareness indicators in the observation, computed by the shared estimator; critic fault flags.
- [ ] Shim in the training loop at deployment thresholds, with masked policy-gradient on shim-controlled steps.
- [ ] Benchmark v0 (P0 scenarios x S1/S2 x 64 seeds) with `race_pid` comparison and thresholds from 4.4.
- [ ] Engine logging extension (raw height, packet age, tag id/R, injected vs sent rc) and unit-checked observation builder (SW-6).
- [ ] Real stage R-S0 (shadow) and R-S1 (hover) fault trials; measure the real outage rate.

**P1: before a full lap / racing**
- [ ] Remaining T/S/C/A/E P1 scenarios, compound X-1..X-6, ADR, failure buffer, adversarial search (X-7).
- [ ] OOD detector and hand-over tests (SW-7, P-5).
- [ ] Engine fault-injection shim (`core/fault_injection.py`) and R-S2/R-S3 trials.
- [ ] Sim-vs-real gap log and severity recalibration; GRU decision gate (3.6).
- [ ] CI-style regression script and release gates.

**P2: as time allows**
- [ ] Lighting/glare/low light (T-8, T-10, E-7), propwash/ground effect (A-6), payload (A-7), obstacles/crowd (E-4, E-5), EMI-effort coupling (C-9), pitch/roll bias (S-9).
