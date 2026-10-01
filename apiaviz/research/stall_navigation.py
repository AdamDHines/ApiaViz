"""Opt-in continuous-navigation adapter for the visual-stall recovery control.

Use a fresh protocol/output and this evaluate entry point in a separate process.
Existing UV validation protocols deliberately keep their original controller.
"""
from unittest.mock import patch

from . import motor_feedback
from .visual_stall_recovery import VisualStallRecovery


class _RecoveryBehaviour(VisualStallRecovery):
    def choose(self, desired_heading):
        heading, stride, record = super().choose(desired_heading)
        if record['state'] == 'visual_stall_recovery':
            record['recovery_state'] = record['state']
            record['state'] = 'avoid'
        return heading, stride, record


class RecoveryFeedback(motor_feedback.FeedbackReflex, _RecoveryBehaviour):
    """Keep all commanded-motion feedback while replacing only the reflex."""


def evaluate(*args, **kwargs):
    with patch.object(motor_feedback, 'FeedbackReflex', RecoveryFeedback):
        return motor_feedback.evaluate(*args, **kwargs)
