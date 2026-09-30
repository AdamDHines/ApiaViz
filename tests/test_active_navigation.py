import unittest
import numpy as np

from apiaviz.research.active_navigation import (Active, Exhaustive, Observations,
                                               Straight, evaluate, segment_collision)
from apiaviz.research.regime_experiments import bank_indices, bent_route


SETTINGS = dict(speed_m_s=.1, yaw_speed_deg_s=180., observation_s=.05,
                time_budget_s=360., observation_budget=2600)
EVAL = dict(max_steps=30, kick_before_step=6, recovery_radius_m=.1,
            recovery_consecutive_steps=3)
ACTIVE = dict(min_turn_deg=2., max_turn_deg=30., scan_threshold=.25,
              poor_views=3, scan_cooldown_steps=6, scan_offsets=[-60., -30., 30., 60.])


class SensorFunction:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, position, headings):
        self.calls.append((np.array(position).copy(), np.array(headings).copy()))
        return -np.array([self.value(h) for h in headings])


class ActiveNavigationTests(unittest.TestCase):
    def test_teaching_budget_is_matched_without_duplicates(self):
        centre, spread = bank_indices(77, 'centre'), bank_indices(77, 'spread_equal')
        self.assertEqual(len(centre), len(spread))
        self.assertEqual(len(set(spread)), 77)
        np.testing.assert_array_equal(centre // 9, spread // 9)
        self.assertEqual(set(spread % 9), set(range(9)))
        self.assertEqual(len(bank_indices(77, 'corridor3')), 231)

    def test_exhaustive_scans_pay_for_actual_yaw_and_choose_observed_heading(self):
        scorer = SensorFunction(lambda h: 1. if h == 20 else .5)
        sensor = Observations(scorer, [0, 0], 0., SETTINGS)
        self.assertTrue(Exhaustive().step(sensor, 1))
        self.assertEqual(sensor.count, 13)
        self.assertEqual(sensor.heading, 20)
        self.assertAlmostEqual(sensor.rotation_deg, 60 + 120 + 80)
        self.assertAlmostEqual(sensor.time, 13 * .05 + 260 / 180)
        self.assertEqual([x[1][0] for x in scorer.calls], list(range(60, -61, -10)))

    def test_observation_budget_blocks_unobserved_candidates(self):
        scorer = SensorFunction(lambda h: .5)
        settings = dict(SETTINGS, observation_budget=2)
        sensor = Observations(scorer, [0, 0], 0., settings)
        self.assertFalse(Exhaustive().step(sensor, 1))
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual(sensor.termination, 'observation_budget')

    def test_time_budget_prevents_extra_observation(self):
        scorer = SensorFunction(lambda h: .5)
        sensor = Observations(scorer, [0, 0], 0., dict(SETTINGS, time_budget_s=.38))
        self.assertIsNone(sensor.observe(60))  # .333 s turn + .05 s sensing
        self.assertEqual(len(scorer.calls), 0)
        self.assertLessEqual(sensor.time, .38)

    def test_active_uses_current_view_then_scans_only_after_three_poor_views(self):
        scorer = SensorFunction(lambda h: 0.)
        sensor = Observations(scorer, [0, 0], 0., SETTINGS)
        controller = Active(ACTIVE, dict(low=0., range=1.))
        for i in (1, 2):
            controller.step(sensor, i)
            self.assertEqual(sensor.count, i)
            self.assertEqual(sensor.scan_bouts, 0)
        controller.step(sensor, 3)
        self.assertEqual(sensor.count, 7)  # 3 current views, 4 extra scan views
        self.assertEqual(sensor.scan_bouts, 1)

    def test_collision_checks_segment_interior(self):
        obstacle = [dict(x=.5, y=0., conservative_radius_m=.1)]
        self.assertTrue(segment_collision([0, 0], [1, 0], obstacle))
        self.assertFalse(segment_collision([0, .2], [1, .2], obstacle))

    def test_perturbation_is_not_counted_as_walked_distance_or_reset(self):
        route = np.column_stack([np.arange(0, 2.1, .1), np.zeros(21)])
        result = evaluate(route, np.zeros(20), None, Straight(),
                          dict(lateral=0., heading=0., kick=.5), SETTINGS, EVAL,
                          [-1, 5, -1, 5], timed=True)
        self.assertTrue(result['disturbance_applied'])
        self.assertEqual(result['corrective_resets'], 0)
        self.assertAlmostEqual(result['path_length_m'], result['steps'] * .1)
        self.assertFalse(result['recovered'])
        self.assertFalse(result['reached_nest'])

    def test_generated_route_has_fixed_step_and_no_direct_heading_solution(self):
        route, heading = bent_route([[2.3, 8], [2.3, 6], [3, 4.8], [5, 4.7], [7.3, 4.7]])
        np.testing.assert_allclose(np.linalg.norm(np.diff(route, axis=0), axis=1), .1, atol=1e-12)
        result = evaluate(route, heading, None, Straight(), dict(lateral=0., heading=0., kick=0.),
                          SETTINGS, dict(EVAL, max_steps=200), [-.16, 10.24, -.23, 10.09], timed=False)
        self.assertFalse(result['reached_nest'])


if __name__ == '__main__': unittest.main()
