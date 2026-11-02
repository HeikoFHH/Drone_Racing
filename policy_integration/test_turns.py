"""Turn-direction, momentum and camera-availability regression checks."""

from dataclasses import replace
import math
import random
import unittest

from .policy import ActionSample,BaselinePolicy,PolicyConfig,ZERO
from .test_policy import observation
from .turn_sim import Camera,TurnScenario,candidate_config,geometry,run


class TurnTests(unittest.TestCase):
    def test_camera_front_behind_and_outside_cone(self):
        c=Camera()
        self.assertTrue(c.visible((0,0,1),0,(4,0,1.2)))
        self.assertFalse(c.visible((0,0,1),0,(-4,0,1.2)))
        self.assertFalse(c.visible((0,0,1),0,(4,-3,1.2)))
        self.assertTrue(c.visible((0,0,1),math.pi,(-4,0,1.2)))

    def test_example_geometry_gap_is_not_a_universal_constant(self):
        self.assertAlmostEqual(geometry()["unobserved_gap_before_gate_m"],0.31,places=2)
        self.assertLess(geometry(lateral_offset=0.2)["unobserved_gap_before_gate_m"],0)

    def test_camera_matches_example_calibration(self):
        from pathlib import Path
        import yaml
        source=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/camera/default.yaml').read_text())
        c=Camera()
        self.assertEqual((c.width,c.height),(source['image_width'],source['image_height']))
        self.assertEqual((c.fx,c.fy,c.cx,c.cy),(source['camera_matrix'][0][0],source['camera_matrix'][1][1],source['camera_matrix'][0][2],source['camera_matrix'][1][2]))

    def test_braking_opposes_world_velocity_before_yaw(self):
        cfg=PolicyConfig(waypoints_w=((3,0.5,1),))
        obs=observation(yaw_rad=math.pi/2,velocity_w=(0.2,0,0))
        d=BaselinePolicy(cfg).step(obs,1)
        self.assertEqual(d.mode,'brake')
        self.assertEqual(d.rc[3],0)
        right,forward=d.rc[:2]
        world_x=math.cos(obs.yaw_rad)*forward+math.sin(obs.yaw_rad)*right
        self.assertLess(world_x,0)

    def test_low_speed_heading_correction_does_not_drive_to_goal(self):
        d=BaselinePolicy().step(observation(yaw_rad=math.pi/2,velocity_w=(0.01,0,0)),1)
        self.assertEqual(d.mode,'aim')
        self.assertEqual(d.rc[:2],(0,0))
        self.assertNotEqual(d.rc[3],0)

    def test_brake_entry_drops_the_old_body_frame_command(self):
        p=BaselinePolicy(PolicyConfig(waypoints_w=((3,0.5,1),)))
        for k in range(10):
            now=1+k*0.05
            d=p.step(observation(now,velocity_w=(0.16,0,0)),now)
        self.assertGreater(d.rc[1],10)
        d=p.step(observation(1.5,yaw_rad=math.pi/2,velocity_w=(0.16,0,0)),1.5)
        self.assertEqual(d.mode,'brake')
        self.assertEqual(d.rc[1],0)
        self.assertLess(d.rc[0],0)

    def test_arrival_cannot_advance_with_wrong_heading(self):
        p=BaselinePolicy(PolicyConfig(waypoints_w=((1,0.5,1),)))
        for k in range(12):
            now=1+k*0.05
            d=p.step(observation(now,yaw_rad=math.pi/2),now)
        self.assertFalse(d.request_land)
        self.assertEqual(d.mode,'aim')

    def test_look_at_is_explicit_opt_in(self):
        obs=observation()
        self.assertEqual(BaselinePolicy().step(obs,1).rc[3],0)
        d=BaselinePolicy(candidate_config()).step(obs,1)
        self.assertGreater(d.rc[3],0)

    def test_horizontal_cap_preserves_world_direction_when_yawed(self):
        yaw=math.radians(-60)
        cfg=PolicyConfig(waypoints_w=((3,0.5,1),),heading_targets_w=((2,0.5-math.sqrt(3),1),))
        p=BaselinePolicy(cfg)
        for k in range(20):
            now=1+k*0.05
            d=p.step(observation(now,yaw_rad=yaw),now)
        right,forward=d.rc[:2]
        world_y=math.sin(yaw)*forward-math.cos(yaw)*right
        self.assertLess(abs(world_y),0.6)
        self.assertLessEqual(math.hypot(right,forward),20.71)

    def test_learned_horizontal_vector_is_also_bounded(self):
        p=BaselinePolicy(mode='learned',expected_model_id='fixture')
        for k in range(20):
            now=1+k*0.05
            action=ActionSample((1,1,0,0),now,now,'fixture')
            d=p.step(observation(now),now,action)
        self.assertLessEqual(math.hypot(*d.rc[:2]),20.71)

    def test_horizontal_slew_stays_bounded_across_random_direction_changes(self):
        p=BaselinePolicy(mode='learned',expected_model_id='fixture')
        rng=random.Random(19)
        for k in range(2000):
            now=1+k*0.01
            action=ActionSample((rng.uniform(-1,1),rng.uniform(-1,1),0,0),now,now,'fixture')
            d=p.step(observation(now),now,action)
            self.assertLessEqual(math.hypot(*d.rc[:2]),20.71)

    def test_heading_wrap_takes_short_direction(self):
        p_w=(1,0.5,1)
        angle=math.radians(-179)
        point=(p_w[0]+math.cos(angle),p_w[1]+math.sin(angle),1)
        cfg=PolicyConfig(waypoints_w=((3,0.5,1),),heading_targets_w=(point,))
        d=BaselinePolicy(cfg).step(observation(yaw_rad=math.radians(179)),1)
        self.assertLess(d.rc[3],0)
        self.assertEqual(d.mode,'track')

    def test_heading_configuration_rejects_missing_or_invalid_targets(self):
        for headings in ((),((math.nan,0,1),)):
            with self.assertRaises(ValueError):
                PolicyConfig(heading_targets_w=headings)

    def test_candidate_oracle_route_finishes_with_truth_confirmation(self):
        result=run(TurnScenario('oracle'),1,True)
        self.assertEqual(result['outcome'],'complete')
        self.assertTrue(result['truth_lap_finished'])
        self.assertFalse(result['landing_simulated'])

    def test_camera_limited_routes_do_not_claim_a_lap(self):
        for candidate in (False,True):
            result=run(TurnScenario('camera',camera_limited=True),0,candidate)
            self.assertEqual(result['outcome'],'abort')
            self.assertFalse(result['truth_lap_finished'])
            self.assertEqual(result['reason'],'sensors_timed_out')

    def test_hidden_turn_combination_aborts(self):
        result=run(TurnScenario('hidden',yaw_kick_deg=60,hide_from_s=5,hide_duration_s=1),0,True)
        self.assertEqual(result['outcome'],'abort')
        self.assertFalse(result['truth_lap_finished'])

    def test_simulation_is_reproducible(self):
        scenario=TurnScenario('kick',yaw_kick_deg=-60,command_delay_s=0.15)
        self.assertEqual(run(scenario,4,True),run(scenario,4,True))


if __name__=='__main__': unittest.main()
