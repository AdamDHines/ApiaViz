import unittest
from unittest.mock import patch

import numpy as np

from apiaviz.research.visual_stall_recovery import VisualStallRecovery
from apiaviz.research import motor_feedback, stall_navigation
from apiaviz.research.visual_avoidance import Settings


class VisualStallRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(19).uniform(.1, .9, (51, 199, 3)).astype('float32')

    def test_repeated_stationary_images_change_heading_without_contact_input(self):
        for phase in (-1, 1):
            policy = VisualStallRecovery(phase=phase)
            for _ in range(2):
                policy.moved(self.image, self.image.copy(), 20., .02)
            heading, stride, record = policy.choose(20.)
            self.assertEqual(heading, 20.+phase*90.)
            self.assertEqual(stride, .005)
            self.assertFalse(record['visually_clear'])
            self.assertEqual(policy.choose(-40.)[0], heading)
            # A second unsuccessful direction continues the same turning sense.
            for _ in range(2): policy.moved(self.image, self.image, heading, stride)
            self.assertNotEqual(policy.choose(20.)[0], heading)

    def test_stationary_images_do_not_translate_or_expire_visual_memory(self):
        policy = VisualStallRecovery()
        policy.frame_heading = 0.
        policy.points = np.array([[.1, 0.]])
        policy.ages = np.array([.1])
        policy.moved(self.image, self.image, 90., .1)
        np.testing.assert_allclose(policy.points, [[0., -.1]], atol=1e-12)
        np.testing.assert_allclose(policy.ages, [.1])

    def test_untextured_or_changing_views_do_not_claim_stationary_motion(self):
        horizon = np.broadcast_to(np.linspace(.1, .9, 51)[:, None, None], self.image.shape).copy()
        for before, after in [(np.zeros_like(self.image), np.zeros_like(self.image)),
                              (horizon, horizon),
                              (self.image, self.image+1e-4)]:
            policy = VisualStallRecovery()
            for _ in range(5):
                self.assertFalse(policy.moved(before, after, 0., .02)['visual_stagnation'])
            self.assertIsNone(policy.recovery_heading)

    def test_recovery_requires_sustained_image_change_and_preserves_policy(self):
        policy = VisualStallRecovery()
        for _ in range(2): policy.moved(self.image, self.image, 0., .02)
        for _ in range(11):
            policy.moved(self.image, np.roll(self.image, 1, axis=1), 90., .005)
            self.assertIsNotNone(policy.recovery_heading)
        policy.moved(self.image, np.roll(self.image, 1, axis=1), 90., .005)
        self.assertIsNone(policy.recovery_heading)

    def test_navigation_adapter_keeps_motor_feedback_and_continuous_steering(self):
        policy = stall_navigation.RecoveryFeedback(Settings(), 1)
        for _ in range(2): policy.moved(self.image, self.image, 0., .02)
        heading, stride, record = policy.choose(0.)
        self.assertEqual(heading, 90.)
        self.assertEqual(record['behaviour'], 'visual_steering')
        self.assertEqual(record['recovery_state'], 'visual_stall_recovery')
        self.assertEqual(policy.entries, 1)
        self.assertFalse(policy.active)  # Navigator is never suspended.
        self.assertIsNone(policy.resume)
        self.assertEqual(policy.commands, [dict(heading=0., distance_m=.02)]*2)
        policy.moved(self.image, self.image, heading, stride)
        self.assertEqual(policy.commands[-1], dict(heading=90., distance_m=.005))

    def test_scoped_navigation_adapter_restores_bindings_on_failure(self):
        original = motor_feedback.FeedbackReflex
        def fail(*args, **kwargs):
            self.assertIs(motor_feedback.FeedbackReflex, stall_navigation.RecoveryFeedback)
            raise RuntimeError('deliberate evaluator failure')
        with patch.object(motor_feedback.engine, 'evaluate', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'deliberate'):
                stall_navigation.evaluate()
        self.assertIs(motor_feedback.FeedbackReflex, original)


if __name__ == '__main__':
    unittest.main()
