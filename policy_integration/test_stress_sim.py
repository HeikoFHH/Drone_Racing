"""Scenario semantics and visual-data export, independent of hardware."""

import json
from pathlib import Path
import tempfile
import unittest

from .build_view import build
from .stress_sim import SCENARIOS, Scenario, run, suite


class ScenarioTests(unittest.TestCase):
    def test_delayed_observations_still_require_a_truth_lap(self):
        for delay in (0, 0.15):
            result=run(Scenario('delay','delay',delay_s=delay),3,trace=True)
            self.assertEqual(result['outcome'],'complete')
            self.assertTrue(result['truth_lap_finished'])
            self.assertLess(result['max_pose_age_s'],0.25)

    def test_losses_abort_without_claiming_landing_or_lap(self):
        for scenario in SCENARIOS:
            if scenario.id not in ('pose-loss','link-loss','delay-300'):
                continue
            result=run(scenario,trace=True)
            self.assertEqual(result['outcome'],'abort')
            self.assertFalse(result['truth_lap_finished'])
            self.assertEqual(result['trace'][-1][11],'abort')

    def test_export_is_bounded_and_contains_trace_and_scenario_controls(self):
        report=suite(seeds=1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'report.json').write_text(json.dumps(report))
            size=build(root/'report.json',root/'view.html')
            fragment=(root/'view.html').read_text()
            self.assertLess(size,1_000_000)
            self.assertNotIn('__TRACE_DATA__',fragment)
            self.assertIn('id="drone-scenario"',fragment)
            self.assertIn('id="drone-time"',fragment)
            self.assertNotIn('<html',fragment.lower())


if __name__=='__main__':
    unittest.main()
