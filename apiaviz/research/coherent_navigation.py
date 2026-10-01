"""Versioned development control: confirm decline and keep an evasive course.

No geometry/contact interface is added. The navigator and learned memory remain
live throughout steering. This is not enabled in historical study entry points.
"""
from unittest.mock import patch
import numpy as np

from . import motor_feedback
from .familiarity_controller import FamiliarityController
from .visual_avoidance import wrap
from .visual_stall_recovery import VisualStallRecovery


class DirectionConfirmedController(FamiliarityController):
    """A falling matched-gaze score warrants a direction check, not a blind cast.

    Spatial texture and changing route curvature can lower familiarity even on
    the route. Use the existing charged scan and support criterion; cast if that
    scan finds no supported direction. All thresholds and budgets are unchanged.
    """
    def _start_cast(self, step, reason):
        if reason == 'sustained_decline':
            self._state('reorient', step, 'decline_requires_directional_check')
        else:
            super()._start_cast(step, reason)


class CoherentAvoidance(VisualStallRecovery):
    """An entry-referenced passing side, continuous route input, visual release.

    During an encounter route requests influence the preferred course within
    45 degrees of the entry direction. Safe headings on the committed passing
    side are preferred, with a cost for reversing the last motor command. A
    clear route direction must persist over 6 cm of changing camera imagery
    before the commitment is released. No route location or contact is used.
    """
    schema = 'coherent-image-steering-v2'
    motor_turn_limit_deg = 30.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_heading = None
        self.passing_side = 0
        self.clear_distance = 0.
        self.last_route_clear = False
        # A bearing rejected by image stagnation remains suspect until there
        # has been lateral visual progress. These are commanded-motion vectors,
        # never positions supplied by the evaluator or contact measurements.
        self.stalled_bearings = []

    def _risk(self, headings):
        angle = np.deg2rad(np.asarray(headings)-self.frame_heading)
        directions = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        along = self.points @ directions.T
        across = abs(self.points[:, 0, None]*directions[None, :, 1] -
                     self.points[:, 1, None]*directions[None, :, 0])
        radius = self.settings.clearance_m + .1*np.linalg.norm(self.points, axis=1, keepdims=True)
        blocked = (along > -.01) & (along < self.settings.lookahead_m) & (across < radius)
        risk = np.sum(blocked*np.exp(-np.maximum(along, 0)/self.settings.lookahead_m), axis=0)
        for memory in self.stalled_bearings:
            risk += (abs(wrap(np.asarray(headings)-memory['heading'])) < 60.)
        return risk

    def choose(self, desired_heading):
        self.last_route_clear = False
        if self.recovery_heading is not None:
            if self.entry_heading is None:
                self.entry_heading = float(desired_heading)
                self.passing_side = self.phase
                self.clear_distance = 0.
            heading, stride, record = super().choose(desired_heading)
            heading = float(wrap(self.frame_heading + np.clip(
                wrap(heading-self.frame_heading), -self.motor_turn_limit_deg, self.motor_turn_limit_deg)))
            record.update(heading=heading, offset_deg=float(wrap(heading-desired_heading)))
            record['recovery_state'] = record['state']
            record['state'] = 'avoid'
            return heading, stride, record
        if self.entry_heading is None:
            heading, stride, record = super().choose(desired_heading)
            if record['state'] == 'avoid':
                self.entry_heading = float(desired_heading)
                self.passing_side = int(np.sign(wrap(heading-desired_heading))) or self.phase
                self.clear_distance = 0.
                heading = float(wrap(self.frame_heading + np.clip(wrap(heading-self.frame_heading),
                    -self.motor_turn_limit_deg, self.motor_turn_limit_deg)))
                risk = float(self._risk([heading])[0])
                stride = self.settings.stride_m if risk < .5 else self.settings.probe_m
                record.update(heading=heading, offset_deg=float(wrap(heading-desired_heading)),
                              chosen_risk=risk, visually_clear=risk < .5, stride_m=stride)
            return heading, stride, record
        s = self.settings
        goal = self.entry_heading + np.clip(wrap(desired_heading-self.entry_heading), -45., 45.)
        # Only consider nearby motor courses. The old global candidate set
        # permitted an instantaneous retreat/return reversal after recovery.
        offsets = np.arange(-self.motor_turn_limit_deg, self.motor_turn_limit_deg+.1, s.turn_step_deg)
        near_goal = self.frame_heading + np.clip(wrap(goal-self.frame_heading),
            -self.motor_turn_limit_deg, self.motor_turn_limit_deg)
        candidates = np.r_[self.frame_heading+offsets, near_goal]
        risk = self._risk(candidates)
        clear = risk < .5
        self.last_route_clear = bool(clear[-1] and abs(wrap(near_goal-goal)) < 1e-8
                                     and not self.stalled_bearings)
        if clear[-1]:
            target = len(candidates)-1
        else:
            # Commitment is tied to the entry course, not a changing cast axis.
            side = wrap(candidates-self.entry_heading)*self.passing_side
            on_side = (side >= -1e-8) & (side <= 150.)
            allowed = clear & on_side
            cost = abs(wrap(candidates-goal)) + .75*abs(wrap(candidates-self.frame_heading))
            cost += .001*(wrap(candidates-self.entry_heading)*self.phase < 0)
            available = np.flatnonzero(allowed)
            if len(available):
                target = int(available[np.argmin(cost[available])])
            else:
                # A blocked narrow motor sector calls for a small probe on
                # the committed side, not a passing-side reversal.
                probes = np.flatnonzero(on_side)
                if not len(probes): probes = np.arange(len(candidates))
                target = int(probes[np.argmin((risk*100+cost)[probes])])
        heading = float(wrap(candidates[target]))
        stride = s.stride_m if clear[target] else s.probe_m
        record = dict(state='avoid', desired_heading=float(desired_heading), heading=heading,
            offset_deg=float(wrap(heading-desired_heading)), side=self.passing_side,
            points=len(self.points), frontal_risk=float(risk[-1]), chosen_risk=float(risk[target]),
            visually_clear=bool(clear[target]), stride_m=stride,
            entry_heading=self.entry_heading, route_goal=float(wrap(goal)),
            clear_distance_m=self.clear_distance, steering_schema=self.schema)
        self.decisions.append(record)
        return heading, stride, record

    def moved(self, before, after, heading, distance):
        diagnostic = super().moved(before, after, heading, distance)
        if diagnostic['visual_stagnation'] and self.recovery_heading is not None:
            if not any(abs(wrap(heading-m['heading'])) < 10. for m in self.stalled_bearings):
                self.stalled_bearings.append(dict(heading=float(wrap(heading)), displacement=np.zeros(2)))
        elif not diagnostic['visual_stagnation']:
            displacement = distance*np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
            retained = []
            for memory in self.stalled_bearings:
                memory['displacement'] += displacement
                perpendicular = np.array([-np.sin(np.deg2rad(memory['heading'])),
                                           np.cos(np.deg2rad(memory['heading']))])
                if abs(memory['displacement'] @ perpendicular) < self.settings.lookahead_m:
                    retained.append(memory)
            self.stalled_bearings = retained
        diagnostic['remembered_stalled_bearings'] = [m['heading'] for m in self.stalled_bearings]
        if self.entry_heading is not None:
            self.clear_distance = (self.clear_distance+distance if self.last_route_clear and
                not diagnostic['visual_stagnation'] and self.recovery_heading is None else 0.)
            if self.clear_distance >= .06-1e-10:
                self.entry_heading = None
                self.passing_side = self.side = 0
                self.clear_distance = 0.
        return diagnostic


class CoherentFeedback(motor_feedback.FeedbackReflex, CoherentAvoidance):
    pass


def evaluate(*args, **kwargs):
    with patch.object(motor_feedback, 'FeedbackReflex', CoherentFeedback):
        return motor_feedback.evaluate(*args, **kwargs)
