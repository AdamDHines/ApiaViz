"""Separate controller control: scheduled exploration asks before departing."""
from .coherent_navigation import DirectionConfirmedController


class ConfirmedExplorationController(DirectionConfirmedController):
    def _start_cast(self, step, reason):
        if reason == 'periodic_exploration':
            # Consume the scheduled check, including when it finds a direction.
            # Otherwise every later move would retrigger the overdue check.
            self.last_explore = step
            self._state('reorient', step, 'scheduled_directional_check')
        else:
            super()._start_cast(step, reason)
