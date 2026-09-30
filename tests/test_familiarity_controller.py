import unittest
from dataclasses import replace

import numpy as np

from apiaviz.research.active_navigation import Observations, evaluate
from apiaviz.research.familiarity_controller import (
    ControllerAdapter, FamiliarityController, Settings, acquisition_calibration)


SENSOR = dict(speed_m_s=.1, yaw_speed_deg_s=180., observation_s=.05,
              time_budget_s=360., observation_budget=2600)
EVALUATION = dict(max_steps=120, kick_before_step=16, recovery_radius_m=.1,
                  recovery_consecutive_steps=3)
ROUTE = np.column_stack([np.arange(0, 6.01, .1), np.zeros(61)])


def trial(field, phase=1, lateral=.5, heading=0., kick=0., settings=None, scale=.3,
          sensor_settings=None):
    policy = FamiliarityController(dict(scale=scale), settings, phase)
    scorer = lambda p, hs: -np.array([field(p, h) for h in hs])
    result = evaluate(ROUTE, np.zeros(60), scorer, ControllerAdapter(policy),
        dict(lateral=lateral, heading=heading, kick=kick), sensor_settings or SENSOR,
        EVALUATION, [-10, 20, -10, 10])
    return result, policy


def useful_field(p, h):
    return .6*np.exp(-p[1]**2/1.2) + .3*np.cos(np.radians(h))


class FamiliarityTests(unittest.TestCase):
    def test_calibration_is_acquisition_only_and_handles_identical_codes(self):
        calibration = acquisition_calibration(np.ones((5, 8)))
        self.assertTrue(calibration['degenerate'])
        self.assertEqual(calibration['additional_observations'], 0)
        self.assertGreater(calibration['scale'], 0)
        with self.assertRaises(ValueError):
            acquisition_calibration(np.ones((1, 8)))

    def test_recovery_uses_spatial_signal_for_both_search_phases(self):
        for phase in (-1, 1):
            result, policy = trial(useful_field, phase=phase)
            self.assertTrue(result['recovered'], (phase, result['termination']))
            self.assertTrue(any(t['reason']=='repeated_translation_improvement' for t in policy.transitions))
            self.assertEqual(result['corrective_resets'], 0)

    def test_aligned_route_is_not_stopped_for_lack_of_positive_gradient(self):
        result, _ = trial(useful_field, lateral=0.)
        self.assertTrue(result['reached_nest'], result['termination'])

    def test_flat_high_and_low_familiarity_end_bounded_search(self):
        for value in (.05, .95):
            result, policy = trial(lambda p, h: value)
            self.assertEqual(result['termination'], 'uninformative_views')
            self.assertLess(result['steps'], 30)
            self.assertFalse(result['recovered'])
            self.assertTrue(any(t['after']=='cast' for t in policy.transitions))

    def test_heading_only_field_does_not_claim_translation_improvement(self):
        _, policy = trial(lambda p, h: .6 + .3*np.cos(np.radians(h)))
        comparisons = [d['comparison'] for d in policy.decisions if 'comparison' in d]
        self.assertTrue(comparisons)
        self.assertLess(max(abs(c['normalized_change']) for c in comparisons), 1e-12)
        self.assertFalse(any(t['reason']=='repeated_translation_improvement' for t in policy.transitions))

    def test_score_offset_and_scaled_calibration_preserve_actions(self):
        baseline, _ = trial(useful_field)
        transformed, _ = trial(lambda p,h: 2.5*useful_field(p,h)-.7, scale=.75)
        self.assertEqual(baseline['termination'], transformed['termination'])
        np.testing.assert_allclose([r['position'] for r in baseline['trace']],
                                   [r['position'] for r in transformed['trace']], atol=1e-12)

    def test_temporal_comparisons_match_heading_and_actual_move_distance(self):
        result, policy = trial(useful_field)
        observations = [e for e in result['events'] if e['kind']=='observation']
        for decision in policy.decisions:
            if 'comparison' not in decision: continue
            c = decision['comparison']
            self.assertAlmostEqual(c['distance_m'], .1)
            self.assertAlmostEqual(c['normalized_change'], (c['after']-c['before'])/.3)
            self.assertTrue(any(abs(e['familiarity']-c['after'])<1e-12 and
                                abs((e['heading']-c['reference_heading']+180)%360-180)<1e-9
                                for e in observations))

    def test_adapter_hides_geometry_and_displacement(self):
        seen = []
        class Spy:
            def step(self, sensor, step):
                for attr in ('position', 'route', 'scorer', 'events', 'obstacles', 'endpoint'):
                    self_test.assertFalse(hasattr(sensor, attr))
                seen.append(sensor.distance)
                return True
        self_test = self
        result = evaluate(ROUTE, np.zeros(60), None, ControllerAdapter(Spy()),
            dict(lateral=0.,heading=0.,kick=.5), SENSOR,
            dict(EVALUATION,max_steps=20), [-10,20,-10,10])
        np.testing.assert_allclose(seen, np.arange(20)*.1)
        self.assertTrue(result['disturbance_applied'])

    def test_observation_and_time_limits_prevent_unseen_actions(self):
        for limits, reason in ((dict(SENSOR, observation_budget=2),'observation_budget'),
                               (dict(SENSOR,time_budget_s=.01),'time_budget')):
            result, _ = trial(useful_field, sensor_settings=limits)
            self.assertEqual(result['termination'], reason)
            self.assertEqual(result['steps'], 0)
            self.assertLessEqual(result['time_s'], limits['time_budget_s'])
            self.assertLessEqual(result['observations'], limits['observation_budget'])

    def test_repeated_runs_are_exactly_deterministic(self):
        first, policy = trial(useful_field)
        second, repeated = trial(useful_field)
        self.assertEqual(first, second)
        self.assertEqual(policy.decisions, repeated.decisions)

    def test_unannounced_displacement_can_be_recovered_from_observations(self):
        result, policy = trial(useful_field, lateral=0., kick=.5)
        self.assertTrue(result['disturbance_applied'])
        self.assertTrue(result['recovered'], result['termination'])
        self.assertEqual(result['corrective_resets'], 0)
        after_kick = next(d for d in policy.decisions if d['step']==EVALUATION['kick_before_step'])
        self.assertLess(after_kick['comparison']['normalized_change'], 0.)

    def test_opposite_equal_peaks_are_treated_as_ambiguous(self):
        result, policy = trial(lambda p,h: .6+.3*np.cos(2*np.radians(h)),
                              settings=replace(Settings(),scan_extents=(180.,)))
        self.assertEqual(result['termination'], 'uninformative_views')
        self.assertFalse(any(t['reason']=='supported_direction' for t in policy.transitions))

    def test_settings_reject_invalid_thresholds_and_angles(self):
        for changes in (dict(trend_alpha=0),dict(cast_angle=0),dict(scan_extents=(60,20)),
                        dict(improvement=-.1),dict(check_every=1.5)):
            with self.assertRaises(ValueError): replace(Settings(), **changes)


if __name__ == '__main__': unittest.main()
