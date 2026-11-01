"""Contract and adversarial checks. Run: python -m unittest policy_integration.test_policy -v"""

from dataclasses import replace
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from .adapter import PositionerAdapter, rotation
from .bench import run
from .policy import ACTION_ORDER, ActionSample, BaselinePolicy, Observation, PolicyConfig, ZERO
from .replay import main as replay


def observation(now=1.0, **changes):
    base = Observation(now, now, (1.0, 0.5, 1.0), (0.0, 0.0, 0.0), 0.0,
                       0.03, 0.05, 80.0, True, True, True)
    return replace(base, **changes)


class ContractTests(unittest.TestCase):
    def test_disabled_and_grounded_never_move_or_takeoff(self):
        for change in ({"enabled": False}, {"airborne": False}):
            with self.subTest(change=change):
                d = BaselinePolicy().step(observation(**change), 1.0)
                self.assertEqual(d.rc, ZERO)
                self.assertFalse(d.request_land)

    def test_forward_and_sideways_signs(self):
        cfg = PolicyConfig(waypoints_w=((2, 0.5, 1),))
        self.assertGreater(BaselinePolicy(cfg).step(observation(), 1).rc[1], 0)
        cfg = PolicyConfig(waypoints_w=((1, 0.9, 1),))
        self.assertLess(BaselinePolicy(cfg).step(observation(), 1).rc[0], 0)

    def test_positive_yaw_requires_clockwise_correction_and_no_translation(self):
        d = BaselinePolicy().step(observation(yaw_rad=0.8), 1.0)
        self.assertEqual(d.rc[:2], (0, 0))
        self.assertGreater(d.rc[3], 0)

    def test_vertical_sign(self):
        cfg = PolicyConfig(waypoints_w=((1, 0.5, 1.5),))
        self.assertGreater(BaselinePolicy(cfg).step(observation(), 1).rc[2], 0)

    def test_motion_damping(self):
        cfg = PolicyConfig(waypoints_w=((1, 0.5, 1),))
        d = BaselinePolicy(cfg).step(observation(velocity_w=(0.4, 0, 0)), 1)
        self.assertLess(d.rc[1], 0)

    def test_nonfinite_wrong_shape_and_schema_abort(self):
        changes = [{"yaw_rad": math.nan}, {"battery_pct": math.inf}, {"position_w": (1, 2)},
                   {"position_w": (None, 0, 1)}, {"velocity_w": None}, {"frame": "tag_frame"},
                   {"schema_version": 2}, {"link_ok": 1}, {"position_std_m": -1},
                   {"battery_pct": 101}, {"schema_version": True}]
        for change in changes:
            with self.subTest(change=change):
                d = BaselinePolicy().step(observation(**change), 1)
                self.assertEqual(d.rc, ZERO)
                self.assertTrue(d.request_land)
                self.assertEqual(d.reason, "invalid_observation")

    def test_link_loss_is_latched(self):
        policy = BaselinePolicy()
        d = policy.step(observation(link_ok=False), 1)
        self.assertEqual(d.reason, "link_lost")
        self.assertEqual(policy.step(observation(now=1.05), 1.05).rc, ZERO)
        self.assertTrue(policy.step(observation(now=1.1), 1.1).request_land)

    def test_low_battery(self):
        self.assertEqual(BaselinePolicy().step(observation(battery_pct=10), 1).reason, "low_battery")

    def test_short_pose_loss_holds_long_loss_requests_land(self):
        policy = BaselinePolicy()
        d = policy.step(observation(pose_time_s=0.7), 1)
        self.assertEqual((d.mode, d.rc), ("hold", ZERO))
        for k in range(1, 12):
            now = 1 + k * 0.05
            d = policy.step(observation(now=now, pose_time_s=0.7), now)
        self.assertEqual(d.mode, "abort")
        self.assertTrue(d.request_land)

    def test_fresh_pose_recovers_from_brief_hold(self):
        policy = BaselinePolicy()
        policy.step(observation(pose_time_s=0.7), 1)
        self.assertEqual(policy.step(observation(now=1.05), 1.05).mode, "track")

    def test_continuously_delayed_video_cannot_hold_forever(self):
        policy = BaselinePolicy()
        for i in range(15):
            now = 1 + i * 0.05
            d = policy.step(observation(now, pose_time_s=now-0.3), now)
        self.assertEqual(d.reason, "persistent_pose_stale")
        self.assertTrue(d.request_land)

    def test_telemetry_loss_requests_land(self):
        self.assertEqual(BaselinePolicy().step(observation(telemetry_time_s=0.7), 1).reason, "sensors_timed_out")

    def test_clock_future_backwards_and_gap(self):
        self.assertEqual(BaselinePolicy().step(observation(pose_time_s=1.01), 1).reason, "future_timestamp")
        for now in (1, 0.9, 1.3):
            policy = BaselinePolicy()
            policy.step(observation(), 1)
            self.assertEqual(policy.step(observation(now=now), now).reason, "control_clock_or_deadline")

    def test_out_of_order_measurement(self):
        policy = BaselinePolicy()
        policy.step(observation(), 1)
        self.assertEqual(policy.step(observation(now=1.05, pose_time_s=0.99), 1.05).reason, "out_of_order_timestamp")

    def test_pose_jump_and_relabelled_frozen_frame(self):
        for change, reason in [({"position_w": (4, 0.5, 1)}, "implausible_pose_jump"),
                               ({"pose_time_s": 1, "position_w": (1.1, 0.5, 1)}, "pose_changed_without_new_frame")]:
            with self.subTest(reason=reason):
                policy = BaselinePolicy()
                policy.step(observation(), 1)
                self.assertEqual(policy.step(observation(now=1.05, **change), 1.05).reason, reason)

    def test_uncertainty_fence_and_speed(self):
        for changes, reason in [({"position_std_m": 0.5}, "estimator_uncertain"),
                                ({"velocity_std_mps": 0.5}, "estimator_uncertain"),
                                ({"position_w": (9, 0.5, 1)}, "geofence"),
                                ({"position_w": (1, 0.5, 3)}, "geofence"),
                                ({"velocity_w": (2, 0, 0)}, "excess_speed")]:
            with self.subTest(reason=reason):
                d = BaselinePolicy().step(observation(**changes), 1)
                self.assertEqual(d.reason, reason)
                self.assertEqual(d.rc, ZERO)

    def test_route_and_runtime_segment_clearance(self):
        with self.assertRaises(ValueError):
            PolicyConfig(waypoints_w=((-0.5, 0, 1), (0.5, 0, 1)))
        d = BaselinePolicy().step(observation(position_w=(0, -0.5, 1)), 1)
        self.assertEqual(d.reason, "path_near_pole")

    def test_invalid_config(self):
        for change in ({"rc_limit": 101}, {"rc_limit": 3.4}, {"kp": math.nan},
                       {"x_bounds": (2, 1)}, {"waypoints_w": ()}, {"waypoints_w": ((0, 0, 1),)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                PolicyConfig(**change)

    def test_no_arrival_while_still_moving(self):
        policy = BaselinePolicy()
        for i in range(20):
            now = 1 + i * 0.05
            policy.step(observation(now, position_w=(0, 0.5, 1), velocity_w=(0.2, 0, 0)), now)
        self.assertEqual(policy.waypoint_index, 0)

    def test_one_frozen_frame_cannot_prove_arrival(self):
        policy = BaselinePolicy()
        for i in range(7):
            now = 1 + i * 0.05
            policy.step(observation(now, pose_time_s=1, position_w=(0, 0.5, 1)), now)
        self.assertEqual(policy.waypoint_index, 0)

    def test_final_waypoint_requires_dwell_then_latches_complete(self):
        policy = BaselinePolicy(PolicyConfig(waypoints_w=((1, 0.5, 1),)))
        self.assertFalse(policy.step(observation(), 1).request_land)
        for i in range(1, 9):
            now = 1 + i * 0.05
            d = policy.step(observation(now), now)
        self.assertEqual(d.mode, "complete")
        self.assertEqual(d.rc, ZERO)
        self.assertTrue(d.request_land)

    def test_learner_requires_valid_fresh_matching_packet(self):
        base = ActionSample((0.1, 0.2, 0, 0), 1, 1, "test-model")
        invalid = [None, replace(base, values=(math.nan, 0, 0, 0)),
                   replace(base, values=(1.1, 0, 0, 0)), replace(base, values=(0,)),
                   replace(base, generated_at_s=0.8), replace(base, generated_at_s=1.1),
                   replace(base, observation_pose_time_s=0.9), replace(base, frame="other"),
                   replace(base, action_order=tuple(reversed(ACTION_ORDER))), replace(base, model_id="")]
        for action in invalid:
            with self.subTest(action=action):
                d = BaselinePolicy(mode="learned", expected_model_id="test-model").step(observation(), 1, action)
                self.assertEqual(d.reason, "candidate_failed")
                self.assertEqual(d.rc, ZERO)

    def test_learned_policy_cannot_bypass_cap_heading_or_link_guard(self):
        sample = ActionSample((1, 1, 1, 1), 1, 1, "test-model")
        for change in ({}, {"yaw_rad": 1}, {"link_ok": False}):
            d = BaselinePolicy(mode="learned", expected_model_id="test-model").step(observation(**change), 1, sample)
            self.assertTrue(all(abs(v) <= 20 for v in d.rc))
            if change:
                self.assertEqual(d.rc[:2], (0, 0))

    def test_fractional_slew_at_high_tick_rate(self):
        policy = BaselinePolicy(PolicyConfig(waypoints_w=((3, 0.5, 1),)))
        for i in range(40):
            now = 1 + i * 0.001
            d = policy.step(observation(now), now)
        self.assertGreater(d.rc[1], 3)

    def test_random_normalized_actions_are_bounded(self):
        rng = random.Random(42)
        policy = BaselinePolicy(mode="learned", expected_model_id="fuzz")
        for i in range(1000):
            now = 1 + i * 0.05
            sample = ActionSample(tuple(rng.uniform(-1, 1) for _ in range(4)), now, now, "fuzz")
            d = policy.step(observation(now), now, sample)
            self.assertTrue(all(type(v) is int and abs(v) <= 20 for v in d.rc))
            self.assertFalse(d.request_land)

    def test_huge_json_numbers_fail_closed_without_exception(self):
        for field in ("yaw_rad", "pose_time_s", "battery_pct"):
            with self.subTest(field=field):
                d=BaselinePolicy().step(observation(**{field:10**500}),1)
                self.assertEqual(d.reason,"invalid_observation")
                self.assertEqual(d.rc,ZERO)

    def test_learned_model_identity_and_goal_are_pinned(self):
        for sample in (ActionSample((0,0,0,0),1,1,"other-model"),
                       ActionSample((0,0,0,0),1,1,"chosen",waypoint_index=1)):
            with self.subTest(sample=sample):
                d=BaselinePolicy(mode="learned",expected_model_id="chosen").step(observation(),1,sample)
                self.assertEqual(d.reason,"candidate_failed")
        with self.assertRaises(ValueError):
            BaselinePolicy(mode="learned")

    def test_session_timeout_cannot_be_reset_by_disable_toggle(self):
        cfg=PolicyConfig(max_session_s=0.5)
        policy=BaselinePolicy(cfg)
        policy.step(observation(),1)
        policy.step(observation(now=1.2,enabled=False),1.2)
        d=policy.step(observation(now=1.6),1.6)
        self.assertEqual(d.reason,"session_timeout")
        self.assertTrue(d.request_land)


class AdapterTests(unittest.TestCase):
    def test_map_to_world_and_roundtrip(self):
        adapter = PositionerAdapter(((1, 0, 0), (0, 0, -1), (0, 1, 0)), (0, 0, 1.2))
        self.assertEqual(adapter.point_to_world((2, -0.2, -0.5)), (2, 0.5, 1.0))
        for actual, expected in zip(adapter.point_to_map((2, 0.5, 1)), (2, -0.2, -0.5)):
            self.assertAlmostEqual(actual, expected)

    def test_positioner_camera_heading(self):
        # Identity map->world; camera right=-Y, down=-Z, forward=+X.
        adapter = PositionerAdapter(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
        result = {"xyz": [1, 0.5, 1], "R": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]}
        obs = adapter.adapt(result, pose_time_s=1, telemetry_time_s=1, velocity_w=(0, 0, 0),
                            position_std_m=0.03, velocity_std_mps=0.05,
                            battery_pct=80, link_ok=True, airborne=True)
        self.assertEqual(obs.yaw_rad, 0)
        self.assertFalse(obs.enabled)

    def test_reflections_scaling_and_bad_transforms_rejected(self):
        for r in (((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
                  ((2, 0, 0), (0, 1, 0), (0, 0, 1)), ((1, 0),)):
            with self.assertRaises(ValueError):
                rotation(r)


class EndToEndTests(unittest.TestCase):
    def test_toy_route_and_independent_existing_lap_timer(self):
        from race.lap_timer import CourseConfig, LapTimer
        summary, poses = run(3)
        self.assertTrue(summary["complete"])
        timer = LapTimer(CourseConfig("test", 1, 2, 0.8, 0.4, 2.5), {1: (0, 0, 0), 2: (4, 0, 0)})
        for now, (x, y, z), _ in poses:
            # Existing timer uses tag-map X/Z horizontal, not canonical world X/Y.
            timer.update(now, (x, z - 1.2, -y))
        self.assertEqual(timer.phase.name, "FINISHED")
        self.assertGreater(timer.lap_time_s, 0)

    def test_injected_losses_abort_instead_of_complete(self):
        for fault in ("pose_loss", "link_loss"):
            summary, _ = run(0, fault)
            self.assertFalse(summary["complete"])
            self.assertNotEqual(summary["terminal_reason"], "timeout")

    def test_replay_valid_and_malformed_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "log.jsonl"
            path.write_text(json.dumps({"now_s": 1, "observation": observation().to_dict()}) + "\n")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(replay([str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue())["reason"], "classic_pd")
            path.write_text('{"now_s": 1}\n')
            with redirect_stderr(io.StringIO()):
                self.assertEqual(replay([str(path)]), 2)


if __name__ == "__main__":
    unittest.main()
