import importlib.util
import unittest

from .bridge import Controller,ControlError,DemoTransport,ZERO
from .policies import DemoPose


class PolicyIntegrationTests(unittest.TestCase):
    def start(self,name='baseline'):
        t=DemoTransport();c=Controller(t,True,pose_source=DemoPose(t))
        for now in (0,.05,.1):c.tick(now)
        self.c=c;self.t=t;self.seq=0
        self.send('heartbeat',.1);self.send('takeoff',.1);c.tick(.15);c.tick(.2)
        self.send('run_policy',.2,policy=name)
    def send(self,action,now,**args):
        self.c.accept({'action':action,'session':'policy-integration-12345','seq':self.seq,**args},now);self.seq+=1
    def test_baseline_calls_existing_policy_and_stop_cancels_next_tick(self):
        self.start()
        from policy_integration.policy import BaselinePolicy
        self.assertIsInstance(self.c.policy_session.policy,BaselinePolicy)
        for n in range(5,25):
            now=n*.05;self.send('heartbeat',now);self.c.tick(now)
        self.assertTrue(self.c.policy_active);self.assertNotEqual(self.c.rc,ZERO)
        self.send('stop_policy',1.21);self.c.tick(1.25)
        self.assertFalse(self.c.policy_active);self.assertEqual(self.c.rc,ZERO)
        with self.assertRaises(ControlError):self.send('run_policy',1.26,policy='baseline')
    def test_manual_takeover_cancels_autonomy(self):
        self.start();self.c.tick(.25);self.send('move',.26,direction='up');self.c.tick(.3)
        self.assertFalse(self.c.policy_active);self.assertEqual(self.c.rc,(0,0,20,0))
    def test_policy_fault_latches_landing_and_has_no_resume(self):
        self.start();self.t.p=[20,.5,1];self.c.tick(.25);self.c.tick(.3)
        self.assertTrue(self.c.land_latched);self.assertFalse(self.c.policy_active)
        self.assertIn('land',self.t.sent);self.assertEqual(self.c.rc,ZERO)
    def test_demo_baseline_finishes_and_requests_land(self):
        self.start()
        for n in range(5,1800):
            now=n*.05;self.send('heartbeat',now);self.c.tick(now)
            if self.c.land_latched:break
        self.assertTrue(self.c.land_latched);self.assertIn('route_complete',self.c.message)
    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('tello_calibration'),'Install hardware policy dependencies')
    def test_teammates_actual_race_policy_and_maude_guard_are_used(self):
        self.start('race')
        from race.race_policy import RacePolicy
        from policy_integration.policy import BaselinePolicy
        self.assertIsInstance(self.c.policy_session.policy,RacePolicy)
        self.assertIsInstance(self.c.policy_session.guard,BaselinePolicy)
        for n in range(5,60):
            now=n*.05;self.send('heartbeat',now);self.c.tick(now)
            self.assertLessEqual(max(abs(v) for v in self.c.rc),20)
        self.assertTrue(self.c.policy_active)
        self.send('stop_policy',3.);self.c.tick(3.)
        self.assertFalse(self.c.policy_active);self.assertEqual(self.c.rc,ZERO)


if __name__=='__main__':unittest.main()
