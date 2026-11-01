"""Sign mission: estimator, policies, landing, runner safety, live backend (against a fake drone), control server.

Scope: normal conditions and light wind (0.5 m/s). No extreme-condition tests.
"""

import json
import socket
import threading
import time
import urllib.request
from dataclasses import replace

import numpy as np
import pytest

from sign_mission.common import RawObs
from sign_mission.drone import RealClock, VirtualClock
from sign_mission.landing import LandingCfg, LandingPolicy
from sign_mission.policies import OutAndBackPolicy, PolicyCfg
from sign_mission.range_est import SignRangeEstimator, observation_from_corners
from sign_mission.runner import MissionRunner, RunnerCfg
from sign_mission.sim import Scenario, SignRenderer, SimDrone, sign_pose_in_camera
from sign_mission.spec import INCH, SIGN_WIDTH_M, SignSpec, load_camera

CAM, SIGN = load_camera(), SignSpec()


def mission(policy="approach", seed=0, runner_cfg=None, **kw):
    clk = VirtualClock()
    sd = SimDrone(Scenario(**kw), clock=clk, seed=seed)
    r = MissionRunner(sd, cfg=runner_cfg)
    ok, msg = r.start(policy)
    assert ok, msg
    r.thread.join(300)
    return r.status(), sd.truth(), r, sd


# ------------------------------------------------------------------------------------ spec / estimator
def test_sign_is_22_5_inches_and_geometry_limits():
    assert abs(SIGN_WIDTH_M - 22.5 * INCH) < 1e-9 and abs(SIGN_WIDTH_M - 0.5715) < 1e-6
    assert 0.70 < CAM.min_range_fully_visible(SIGN_WIDTH_M) < 0.80      # the sign stops fitting in the image at ~0.74 m


def _obs(d, sigma, rng, lateral=0.0):
    rend = SignRenderer(CAM, SIGN, 1)
    R, t = sign_pose_in_camera(np.array([0.03, 0, 1.0]), 0.0, np.array([d, lateral, 1.0]))
    dst = rend.project(R, t)
    return observation_from_corners(dst + rng.normal(0, sigma, dst.shape), 60.0, 1, CAM, SIGN.width_m), float(t[2])


def test_estimator_is_accurate_and_the_filter_beats_single_frames():
    rng = np.random.default_rng(1)
    est, raw_err, filt_err = SignRangeEstimator(SIGN, CAM), [], []
    for k in range(150):
        o, truth = _obs(8.0, 1.0, rng)
        e = est.step(k * 0.05, [o], 0.0)
        raw_err.append(o.z_pnp - truth)
        if k > 15:
            filt_err.append(e.range_m - truth)
    sd = lambda v: float(np.std(v))
    assert sd(filt_err) < 0.6 * sd(raw_err)                       # the filter removes the frame-to-frame noise ...
    assert abs(np.mean(filt_err)) < 0.15 and abs(np.mean(raw_err)) < 0.15   # ... what is left is a ~1 % pose bias at 1 px corner noise, 8 m


def test_estimator_rejects_an_outlier_and_dead_reckons_when_the_sign_is_too_close():
    rng = np.random.default_rng(2)
    est = SignRangeEstimator(SIGN, CAM)
    for k in range(40):
        o, _ = _obs(5.0, 0.3, rng)
        e = est.step(k * 0.05, [o], 0.0)
    bad, _ = _obs(9.0, 0.3, rng)                          # one frame with a wildly wrong distance
    e = est.step(40 * 0.05, [bad], 0.0)
    assert abs(e.range_m - 5.0) < 0.2 and "outlier" in e.note
    # sign lost close to the camera (cut by the image): the filter keeps going on the closing speed
    est2 = SignRangeEstimator(SIGN, CAM)
    for k in range(20):
        o, _ = _obs(1.0, 0.2, rng)
        est2.step(k * 0.05, [o], 0.0)
    e2 = est2.step(20 * 0.05 + 1.0, [], closing_mps=0.2)
    assert e2.tracking and e2.source == "near_field" and e2.range_m < 0.9


# ------------------------------------------------------------------------------------------ policy 1
@pytest.mark.parametrize("kw", [
    dict(distance_m=3.0), dict(distance_m=8.0), dict(distance_m=12.0),
    dict(distance_m=8.0, lateral_m=1.0), dict(distance_m=8.0, lateral_m=-1.0),
    dict(distance_m=8.0, heading_error_deg=15), dict(distance_m=8.0, heading_error_deg=-15),
    dict(distance_m=8.0, mps_per_rc=0.004), dict(distance_m=8.0, mps_per_rc=0.016),       # true drone half / twice as fast as assumed
    dict(distance_m=8.0, wind_mps=(0.0, 0.5)), dict(distance_m=8.0, wind_mps=(0.0, -0.5)),   # light cross-wind
    dict(distance_m=8.0, rc_deadband=4), dict(distance_m=8.0, corner_noise_px=0.8), dict(distance_m=8.0, video_delay_s=0.2)])
def test_approach_stops_half_a_metre_in_front_and_lands(kw):
    s, truth, r, sd = mission("approach", **kw)
    end = s["result"]["truth_at_end"]
    assert s["phase"] == "LANDED", s["reason"]
    assert abs(end["depth_to_sign_m"] - 0.5) < 0.10, end
    assert not truth["collided"] and truth["phase"] == "landed" and truth["z"] == 0.0
    assert abs(s["result"]["final_range_m"] - end["depth_to_sign_m"]) < 0.10      # the drone's own estimate is right too


def test_approach_light_wind_along_the_axis_does_not_hit_the_sign():
    for wind in ((0.3, 0.0), (-0.3, 0.0)):            # 0.3 m/s tail- and headwind (0.5 m/s tailwind still reaches the sign while landing: documented)
        s, truth, *_ = mission("approach", distance_m=8.0, wind_mps=wind)
        assert s["phase"] == "LANDED" and not truth["collided"] and s["result"]["truth_at_end"]["depth_to_sign_m"] > 0.35


def test_full_loop_with_the_real_detector_on_rendered_images():
    s, truth, r, sd = mission("approach", distance_m=3.0, perception="render")
    assert s["phase"] == "LANDED" and abs(s["result"]["truth_at_end"]["depth_to_sign_m"] - 0.5) < 0.10 and not truth["collided"]


# ------------------------------------------------------------------------------------------ policy 2
@pytest.mark.parametrize("kw", [dict(distance_m=4.0), dict(distance_m=8.0), dict(distance_m=8.0, lateral_m=1.0),
                                dict(distance_m=8.0, heading_error_deg=15), dict(distance_m=8.0, mps_per_rc=0.016)])
def test_out_and_back_turns_180_and_returns_to_the_start(kw):
    s, truth, r, sd = mission("out_and_back", **kw)
    end, pol = s["result"]["truth_at_end"], r.policy
    assert s["phase"] == "LANDED", s["reason"]
    assert abs(pol.turned_deg - 180.0) < 6.0
    assert (end["x"] ** 2 + end["y"] ** 2) ** 0.5 < 0.6                              # back within 0.6 m of the take-off spot
    assert abs(s["result"]["out_distance_m"] - s["result"]["returned_m"]) < 0.3     # "the same distance again"
    assert not truth["collided"]


def test_single_sign_policy_two_never_searches_for_a_second_tag():
    s, truth, r, sd = mission("out_and_back", distance_m=6.0)
    names = [e["msg"] for e in s["events"]]
    seq = [e for e in r.policy.events]
    after_turn = [x for x in seq if "TURN" in x or "RETURN" in x or "STOP2" in x]
    assert after_turn and not any("SEARCH" in x for x in seq[[i for i, x in enumerate(seq) if "TURN" in x][0]:])


# ---------------------------------------------------------------------------------- stop and landing
def test_stop_button_mid_flight_stops_and_lands_without_touching_the_sign():
    clk = VirtualClock()
    sd = SimDrone(Scenario(distance_m=10.0), clock=clk)
    r = MissionRunner(sd)
    orig = clk.sleep

    def sleep(dt):
        orig(dt)
        if clk.now() > 14.0:
            r.stop()
    clk.sleep = sleep
    r.start("approach")
    r.thread.join(120)
    s, tr = r.status(), sd.truth()
    assert s["phase"] == "LANDED" and "operator" in s["reason"] and tr["plane_distance_m"] > 1.0 and not tr["collided"] and tr["z"] == 0.0


def test_landing_policy_sequence_brake_settle_descend_land():
    lp, t, seen = LandingPolicy(LandingCfg()), 0.0, []
    h = 1.0
    for _ in range(600):
        a = lp.step(RawObs(t=t, height_m=h, speed_mps=0.0))
        if not seen or seen[-1] != a.state:
            seen.append(a.state)
        if a.rc[2] < 0:
            h = max(0.0, h - 0.25 * 0.05)
        if a.command == "land":
            break
        t += 0.05
    assert seen == ["BRAKE", "SETTLE", "DESCEND", "LAND"] and a.command == "land" and a.rc == (0, 0, 0, 0) and h <= 0.36


def test_landing_policy_lands_at_once_if_the_height_does_not_fall():
    lp, t = LandingPolicy(LandingCfg(stuck_s=1.0)), 0.0
    for _ in range(400):
        a = lp.step(RawObs(t=t, height_m=1.0, speed_mps=0.0))
        t += 0.05
        if a.command == "land":
            break
    assert a.command == "land" and "not falling" in a.info["reason"]


# --------------------------------------------------------------------------------------- safety nets
def test_a_sign_behind_the_drone_is_found_by_turning():
    s, truth, r, sd = mission("approach", distance_m=8.0, heading_error_deg=180)
    assert s["phase"] == "LANDED" and abs(s["result"]["truth_at_end"]["depth_to_sign_m"] - 0.5) < 0.1 and not truth["collided"]


def test_policy_aborts_and_lands_when_the_sign_is_never_found():
    s, truth, r, sd = mission("approach", distance_m=60.0)       # too far to detect: 8.7 px wide, below the 12 px limit
    assert s["phase"] == "ABORTED" and s["reason"] == "sign_not_found" and truth["z"] == 0.0


def test_runner_refuses_low_battery_and_an_unconfirmed_live_start():
    clk = VirtualClock()
    sd = SimDrone(Scenario(battery_pct=10.0), clock=clk)
    r = MissionRunner(sd)
    r.start("approach")
    r.thread.join(30)
    assert r.status()["phase"] == "ERROR" and "battery" in r.status()["reason"] and sd.phase == "ground"
    sd.live = True
    ok, msg = MissionRunner(sd).start("approach", confirmed=False)
    assert not ok and "confirm" in msg


def test_runner_never_sends_a_stick_above_its_cap():
    clk = VirtualClock()
    sd = SimDrone(Scenario(distance_m=6.0, mps_per_rc=0.002), clock=clk)
    seen = []
    orig = sd.send_rc
    sd.send_rc = lambda *a: (seen.append(max(abs(v) for v in a)), orig(*a))[1]
    r = MissionRunner(sd, cfg=RunnerCfg(rc_cap=30))
    r.start("approach")
    r.thread.join(300)
    assert seen and max(seen) <= 30


def test_shadow_mode_runs_the_policy_but_never_flies():
    clk = VirtualClock()
    sd = SimDrone(Scenario(distance_m=6.0), clock=clk)
    sent = []
    sd.send_rc = lambda *a: sent.append(a)
    sd.takeoff = lambda: sent.append("takeoff")
    r = MissionRunner(sd, cfg=RunnerCfg(shadow=True, shadow_max_s=8.0))
    r.start("approach")
    r.thread.join(60)
    assert sent == [] and r.status()["phase"] == "LANDED" and r.status()["detected"] in (True, False)


# ------------------------------------------------------------------------- live backend vs a fake Tello
class FakeTello:
    def __init__(self, cmd_port, state_port):
        self.cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd.bind(("127.0.0.1", cmd_port))
        self.cmd.settimeout(0.2)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_port, self.h, self.log, self.stop = state_port, 0, [], threading.Event()
        threading.Thread(target=self._cmds, daemon=True).start()
        threading.Thread(target=self._state, daemon=True).start()

    def _cmds(self):
        while not self.stop.is_set():
            try:
                data, src = self.cmd.recvfrom(256)
            except socket.timeout:
                continue
            c = data.decode()
            self.log.append(c)
            if c.startswith("rc"):
                continue
            if c == "takeoff":
                self.h = 80
            if c == "land":
                self.h = 0
            self.cmd.sendto(b"ok", src)

    def _state(self):
        while not self.stop.is_set():
            self.out.sendto(f"pitch:0;roll:0;yaw:10;vgx:0;vgy:0;vgz:0;h:{self.h};bat:77;".encode(), ("127.0.0.1", self.state_port))
            time.sleep(0.05)


def test_live_backend_against_a_fake_tello_and_stop_lands():
    from sign_mission.tello import TelloDrone
    fake = FakeTello(28889, 28890)
    d = TelloDrone(drone_ip="127.0.0.1", drone_cmd_port=28889, local_cmd_port=28891, local_state_port=28890)
    try:
        assert d.connect() and abs(d.telemetry().battery_pct - 77) < 1e-6 and d.telemetry().yaw_deg == 10
        r = MissionRunner(d, cfg=RunnerCfg(require_video=False),
                          landing_cfg=LandingCfg(stuck_s=0.8, brake_max_s=0.4, settle_s=0.2))
        assert not r.start("approach", confirmed=False)[0]                      # live needs the operator's confirmation
        assert r.start("approach", confirmed=True)[0]
        t0 = time.time()
        while time.time() - t0 < 10 and r.status()["phase"] != "FLYING":
            time.sleep(0.05)
        assert r.status()["phase"] == "FLYING" and fake.h == 80
        time.sleep(0.6)
        r.stop()
        r.thread.join(30)
        s = r.status()
        assert s["phase"] == "LANDED" and fake.h == 0
        cmds = fake.log
        assert cmds.index("takeoff") < cmds.index("land") and any(c.startswith("rc") for c in cmds) and "emergency" not in cmds
        assert cmds[-1].startswith("rc 0 0 0 0") or "land" in cmds
    finally:
        r.close()
        d.close()
        fake.stop.set()


# ---------------------------------------------------------------------------------------- the web server
def test_control_server_start_stop_status_and_video():
    from sign_mission.server import serve
    app = serve(SimDrone(Scenario(distance_m=5.0, perception="analytic"), clock=RealClock(20.0)), port=18770, block=False)
    base = "http://127.0.0.1:18770"
    get = lambda p: json.loads(urllib.request.urlopen(base + p).read())

    def post(p, b=None):
        req = urllib.request.Request(base + p, data=json.dumps(b or {}).encode(), headers={"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req).read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())
    try:
        time.sleep(0.8)
        s = get("/api/status")
        assert s["mode"] == "simulation" and s["phase"] == "IDLE" and s["detected"] and 4.5 < s["range_m"] < 5.5
        assert urllib.request.urlopen(base + "/frame.jpg").read()[:2] == b"\xff\xd8"
        assert b"Sign flight control" in urllib.request.urlopen(base + "/").read()
        assert post("/api/start", {"policy": "nope"})["ok"] is False
        assert post("/api/start", {"policy": "approach"})["ok"] is True
        assert post("/api/start", {"policy": "approach"})["ok"] is False             # already running
        time.sleep(1.2)
        assert post("/api/stop")["ok"]
        t0 = time.time()
        while time.time() - t0 < 30 and get("/api/status")["phase"] not in ("LANDED", "ABORTED", "ERROR"):
            time.sleep(0.2)
        s = get("/api/status")
        assert s["phase"] == "LANDED" and s["truth"]["collided"] is False and s["truth"]["z"] == 0.0
    finally:
        app.httpd.shutdown()
        app.runner.close()
