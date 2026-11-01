# Proposal: automatic recovery for the flight policy

Date: 2026-11-01. Status: **proposal, not implemented** (apart from the measurement hook described in section 2).
Built on [ROBUSTNESS_PLAN.md](ROBUSTNESS_PLAN.md) (scenario IDs such as T-1 or A-4 refer to it),
[SIM_TEST_REPORT.md](SIM_TEST_REPORT.md) and Maude's turn/visibility work on branch `maude/policy-integration`
(not merged). All numbers come from the simulator with placeholder parameters; they show where the policies are weak,
not how the real drone behaves.

## 1. What is missing today

The drone should recover on its own when it loses sight of the tags or when something else goes wrong. Today the
three pieces that exist do not work together:

| Piece | What it does when tracking is lost | Gap |
|---|---|---|
| Tvisha's racer (`RacePolicy`) | stops, turns on the spot toward where the tag was last seen, resumes when a tag appears; lands after 30 s | no knowledge of where it drifted to; sweep direction is a guess; no check that the re-found position is trustworthy |
| Learned policy (PPO) | keeps flying on dead reckoning | never trained with blackouts; flies on after losing the tags and can hit a pole |
| Maude's baseline and the guard | aborts and lands (guard: hover after 2.5 s without a fix, land after 4 s) | safe, but turns every recoverable loss into a landing, even for policies that would have recovered |

## 2. Measured evidence (simulator, realistic sensing, 30 seeds)

I injected a total tag blackout at t = 6 s (mid-outbound) of 0, 1, 2, 4 and 8 s and ran each policy to a 70 s limit.
`lap` = true timer finished, `est-only` = only the estimate-driven timer finished, `landed` = safe landing request.

| Blackout | Tvisha's racer | PPO (realistic) | Maude (bridged) | Guard + racer | Guard + PPO |
|---|---|---|---|---|---|
| none | 21 lap, 9 est-only | 30 lap | 30 landed | 19 lap, 8 est-only, 3 landed | 30 landed |
| 1 s | 24 lap, 6 est-only | 30 lap | 30 landed | 22 lap, 5 est-only, 3 landed | 30 landed |
| 2 s | 26 lap, 4 est-only | 30 lap | 30 landed | 30 landed | 30 landed |
| 4 s | 26 lap, 4 est-only | **17 lap, 12 pole crashes**, 1 est-only | 30 landed | 30 landed | 30 landed |
| 8 s | 26 lap, 4 est-only | 16 lap, 13 est-only, 1 pole crash | 30 landed | 30 landed | 30 landed |

Reading it:
1. Tvisha's stop-and-search recovers from any blackout length tried, and finishes laps.
2. The learned policy has no recovery: a 4 s blackout makes it hit a pole in 12 of 30 runs.
3. The guard prevents every crash but lands everything that is blind for 2 s or longer. It also cancels the racer's working recovery.
4. Earlier findings that matter for recovery: the learned policy turns 65 degrees away from the tags and flies blind up to 6.3 s on purpose;
   the racer passes 0.32 m from a pole at the start; the estimate-driven lap timer and the true timer disagree by up to 1.2 s;
   Maude's strong-drift and deadband cases fail (0 of 100 and 26 of 100).

Limits of this evidence: one fault type (total tag blackout), one injection time, one scenario, placeholder dynamics.

## 3. Proposed additions

One new layer between the primary policy and the existing guard, plus changes to the policies themselves. Order of authority:
**guard limits (land) > recovery layer > primary policy**, with Maude's PD route still the backup for invalid or stalled output.

### 3.1 Recovery state machine (new, the core of the proposal)

```
NOMINAL --fix age > 0.4 s--> COAST --fix age > 1.0 s--> BRAKE --speed < 0.05 m/s--> REACQUIRE
   ^                           |  fix back                                              |  3 consistent fixes
   |                           v                                                        v
   +------- RESUME <-------- (NOMINAL)                                             RELOCALIZE (verify)
   +------- ramp speed 0.2 -> 1.0 over 2 s <------------------------------------------+
REACQUIRE --budget 8 s--> RETRACE (fly back to the last good fix, slowly) --budget 6 s--> HOLD_AND_LAND
```

| State | What the drone does | Why |
|---|---|---|
| COAST | primary keeps flying, but under the speed governor (3.3) | short gaps (detector 12 Hz, narrow field of view) are normal |
| BRAKE | counter-velocity from the packet velocity, no translation, then stop; this is Maude's brake-before-turn rule | zero stick is not zero speed; turning at speed drifts into poles |
| REACQUIRE | rotate toward the tag the dead-reckoned estimate says should be visible, then sweep +/-60 degrees starting toward that bearing, 25 stick units yaw, 0.5 s dwell per heading | informed search is faster than blind sweeping; slow yaw limits blur and the 0.15 s video delay |
| RELOCALIZE | hover until 3 consistent fixes agree within 0.3 m and the filter uncertainty drops below 0.25 m; if the fix differs from the dead-reckoned pose by more than 0.5 m, log a "jump" and reset the policy's history | stops the drone from acting on one wrong fix (flip, ghost tag) |
| RESUME | continue at the waypoint that matches the lap phase, speed ramped over 2 s | avoids a command jump on return |
| RETRACE | fly slowly back toward the last pose where tags were seen (stored breadcrumbs), then scan again | the tag was last visible there, so the field of view will contain it |
| HOLD_AND_LAND | hover, then land request | bounded budget: about 20 s total from first loss |

The guard's pose-age landing (4 s) is relaxed only while the recovery layer is active and only up to its budget; this needs
Maude's contract v1.1 (already proposed as decision D4 in the simulation plan).

### 3.2 Visibility-aware heading and course (policy change)

The drone should choose its heading so a mapped tag stays in the 55 degree field of view along the whole route, not just at the ends.
Both Maude's pinhole check (tag B leaves the image at X of about 3.07 m, gate entry at 3.38 m) and my sensor model show this fails for
the fixed-heading route. Proposals: (a) nose on B for the whole lap, as Tvisha does, plus an explicit turn at the end with the
brake-first rule; (b) a heading selector that scores candidate headings by predicted tag visibility over the next 1.5 m; (c) a cheap
hardware change: put tags on several faces of each pole or add one relay tag near each end, so a tag is always in view while turning.
(c) is the only option that fixes the geometry rather than working around it.

### 3.3 Uncertainty-aware speed governor

Limit speed so the drone can always stop inside the clearance: `v_max = sqrt(2 * a_brake * max(0, d_clear - 0.1 - 2 * position_std))`,
reduced further as the fix gets older (`x clamp(1 - fix_age / 3 s, 0.2, 1)`), where `d_clear` is the distance to the nearest pole
minus the 0.25 m crash radius. This directly targets the 12 pole crashes of the learned policy and the racer's 0.32 m pass at the start.
`a_brake` must be measured; until then use 0.5 m/s^2 as a placeholder.

### 3.4 Estimator and sensor health monitor

A small monitor that outputs `OK / DEGRADED / BAD` and the reason, from: fix age per tag; filter innovation statistics and rejected-fix count
(wrong poses, T-12); velocity vs fix mismatch (optical-flow trouble, S-3/S-8); repeated identical frames (T-13); packet age (S-1);
loop period (SW-1, SW-2); NaN or out-of-range inputs (SW-4). It drives the COAST transition, is logged every tick, and is given to
learned policies as an observation.

### 3.5 Disturbance and actuation fixes for the PD baseline

Maude's baseline fails strong drift (0 of 100) and deadband 12 (26 of 100) because it has no integral term and sends commands below the
deadband. Add an integral term with anti-windup (Tvisha's racer already has one) and a minimum stick value while far from the target.
Kick recovery (P-2 to P-4): treat a sudden position or velocity jump as a disturbance and run BRAKE, re-aim, RESUME.

### 3.6 Command-path watchdogs

Sanitize inputs; slow down when the loop period exceeds 0.12 s and hover above 0.3 s; treat a frozen video stream as a blackout; hover when
the state packet is older than 0.5 s; on link loss follow the existing rule (latch, land when the link returns). Exactly one component
writes the sticks.

### 3.7 Learned-policy changes

Recommended order: first let the recovery layer handle faults and keep the policy for NOMINAL flying (fast, interpretable, safe), then make the
policy itself more robust:
- reward for keeping a tag in view (the plan's -0.05 per step unseen over 0.5 s) to remove the blind-flying habit;
- wider odometry drift in domain randomization (packet bias up to 0.1 m/s, scale 15%);
- fault-awareness inputs: time since each tag was last seen, last-seen bearing, recovery state, shim flag;
- fault episodes in training (30 percent): blackouts of 0.5 to 8 s, position and velocity kicks, pose outliers, latency spikes;
- recovery bonus for relocalizing within 8 s without pole proximity; a safe landing is neutral, a crash is not.

## 4. How we will know it works

Proposed acceptance tests (simulator first; thresholds are proposals, to be revised with real data):

| Test (ROBUSTNESS_PLAN ID) | Pass condition for every policy behind the recovery layer |
|---|---|
| Tag blackout 0.5, 1, 2, 4 s (T-1) | 0 crashes; at least 90 percent true laps |
| Tag blackout 8 s (T-1) | 0 crashes; at least 80 percent relocalize and finish, rest land safely |
| Tag never visible (T-3) | hover and land within 20 s, no motion beyond 0.5 m |
| Position kick 0.5 to 1.0 m, velocity kick 0.8 m/s (P-2, P-3) | recovered within 3 s, no pole contact |
| Pose outliers, ghost tags (T-6, T-12) | no jump of more than 0.3 m in commanded direction; no false lap |
| Latency spike 0.5 s, frozen frames 1 s (S-11, T-13) | no oscillation; resumes within 2 s |
| Strong drift, deadband 12 (A-4, A-8) | Maude's baseline at least 90 percent (today 0 and 26 percent) |
| Link outage 3 s (C-8) | latched landing request, no motion on return |
| Compound: blackout + wind + latency (X-1) | 0 crashes |

Also track: time to relocalize, minimum true pole distance, count of guard false landings (today 2 to 3 of 30), and the
difference between the estimate and true lap time. Regression: add these cells to the report script and to CI.

## 5. Plan

| Phase | Work | Effort |
|---|---|---|
| 1 | recovery state machine as a controller wrapper in the simulator, speed governor, health monitor, guard coordination, fault injectors (blackout done; add kicks, frozen frame, latency spike, outliers), tests and report cells | about 1 week |
| 2 | visibility-aware heading and breadcrumb retrace; integral and deadband fix for the PD baseline; retrain the learned policy with the fault curriculum | about 1 week |
| 3 | port to the real stack as a shadow-mode wrapper around the policy; physical checks | needs the Linux laptop and measured braking, delay and camera pose |

Real-flight rule: no software fault injection on an airborne drone before shadow mode passes. Safe real tests: carry the drone by hand and
cover a tag with paper (a physical blackout), then tethered hover with a covered tag, then a short segment with a catcher.

## 6. Decisions I need from you

1. Should I implement phase 1 now in the simulator (nothing touches the real flight code)?
2. Merge Maude's newer branch (brake-before-turn, landmark heading option, turn simulator, dashboard, offline CI workflow)? The CI workflow only runs
   offline unit tests with read-only permissions, but it is a new workflow on your repository.
3. May we change the course (tags on more pole faces or relay tags)? It solves the visibility problem at the root.
4. Contract v1.1 with Maude: allow a stale pose while the recovery layer is active.
5. When can braking distance, command delay and camera pose be measured on the real drone? The speed governor and the braking limit depend on them.

## 7. Related work in other branches

- `maude/policy-integration` (not merged): brake-before-turn and landmark heading, 960 comparison runs showing that a turn controller alone does not fix
  localization, and the same tag-visibility geometry as in section 3.2.
- `connect-trial` (not merged): `tello_basic.py`, a small SDK tool for connecting and sending basic commands with a +/-30 stick cap and automatic landing on
  exit. Useful for the first real link and hover checks, not a policy.
