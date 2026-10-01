"""Regression checks for withdrawal of full-circle body scanning."""
import unittest
import json
from pathlib import Path
from dataclasses import replace
from apiaviz.research.confirmed_exploration import ConfirmedExplorationController
from apiaviz.research.panoramic_confirmation import PanoramicConfirmationController
from apiaviz.research.familiarity_controller import SensorView,Settings,wrap


class BoundedScanningTests(unittest.TestCase):
    def test_recorded_corner_failures_choose_supported_turn_in_both_phases(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/bounded_corner_scores.json').read_text())
        for row in fixture['records']:
            for phase in (-1, 1):
                with self.subTest(world=row['world'], method=row['method'], phase=phase):
                    p = ConfirmedExplorationController(dict(scale=row['scale']),
                        replace(Settings(),scan_step=5.), phase)
                    sensor,calls = self.sensor({v['heading']:v['familiarity'] for v in row['samples']})
                    scan = p._scan(sensor,row['base'],True)
                    self.assertTrue(scan['supported'])
                    self.assertEqual(scan['target'],row['expected_heading'])
                    self.assertEqual(len(calls),25)
                    self.assertTrue(all(abs(wrap(h-row['base']))<=60 for h in calls))

    def sensor(self, scores, budget=100):
        calls=[]
        def observe(h):
            if len(calls)>=budget:return None,h,.05*len(calls)
            calls.append(h)
            return scores.get(h,.1),h,.05*len(calls)
        return SensorView(0,0,0,observe,lambda h:(True,h,.05*len(calls)),lambda:None,lambda _:None),calls

    def test_withdrawn_controller_cannot_be_enabled(self):
        with self.assertRaisesRegex(RuntimeError,'Full-circle scanning was removed'):
            PanoramicConfirmationController(dict(scale=1))

    def test_all_supported_scan_resolutions_and_phases_stay_bounded(self):
        for base in (0.,173.,-177.):
            for step in (5.,10.):
                for phase in (-1,1):
                    p=ConfirmedExplorationController(dict(scale=1),replace(Settings(),scan_step=step),phase)
                    sensor,calls=self.sensor({})
                    sensor.heading=base
                    scan=p._scan(sensor,base,True)
                    self.assertFalse(scan['supported'])
                    self.assertTrue(all(abs(wrap(h-base))<=60+1e-8 for h in calls))
                    self.assertLessEqual(len(calls),25)

    def test_reorientation_checks_for_corner_beyond_nearby_local_peak(self):
        p=ConfirmedExplorationController(dict(scale=1),replace(Settings(),scan_step=5.))
        sensor,calls=self.sensor({0.:.95,45.:1.})
        scan=p._scan(sensor,0.,True)
        self.assertTrue(scan['supported']);self.assertEqual(scan['target'],45.)
        self.assertEqual(len(calls),25);self.assertLessEqual(max(map(abs,calls)),60.)

    def test_budget_exhaustion_does_not_return_partial_confirmation(self):
        p=ConfirmedExplorationController(dict(scale=1))
        sensor,calls=self.sensor({},budget=3)
        self.assertIsNone(p._scan(sensor,0,True));self.assertEqual(len(calls),3)

    def test_tracking_checks_remain_local(self):
        p=ConfirmedExplorationController(dict(scale=1))
        sensor,calls=self.sensor({0.:.95,50.:1.})
        self.assertEqual(p._scan(sensor,0,False)['target'],0)
        self.assertEqual(set(calls),{-10.,0.,10.})


if __name__=='__main__':unittest.main()
