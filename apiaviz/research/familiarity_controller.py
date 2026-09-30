"""Deterministic cast-and-surge policy using causal, matched-gaze observations.

The policy receives a small sensor interface, never world/route coordinates.
Familiarity changes are evidence from sequential observations, not estimates of
ground-truth route distance. The environment and visual memory stay unchanged.
"""
from dataclasses import asdict, dataclass
import math
from typing import Protocol

import numpy as np


def wrap(angle):
    return (angle + 180.) % 360. - 180.


class Sensor(Protocol):
    heading: float
    time: float
    distance: float

    def observe(self, heading: float): ...
    def rotate(self, heading: float) -> bool: ...
    def start_scan(self): ...
    def stop(self, reason: str): ...


@dataclass(frozen=True)
class Settings:
    check_every: int = 3
    explore_every: int = 12
    check_angle: float = 10.
    scan_step: float = 10.
    scan_extents: tuple = (20., 60., 180.)
    contrast_threshold: float = .15
    ambiguity_tolerance: float = .03
    improvement: float = .025
    deterioration: float = -.025
    trend_alpha: float = .5
    poor_moves: int = 2
    confirmation_moves: int = 2
    cast_angle: float = 20.
    max_cast_angle: float = 60.
    max_cast_moves: int = 8
    surge_moves: int = 2
    max_uninformative_scans: int = 2
    use_temporal: bool = True
    use_contrast: bool = True
    periodic_exploration: bool = True

    def __post_init__(self):
        if any(not math.isfinite(float(v)) for v in asdict(self).values()
               if isinstance(v, (float, int))):
            raise ValueError('Settings must be finite')
        for name in ('check_every', 'explore_every', 'poor_moves', 'confirmation_moves',
                     'max_cast_moves', 'surge_moves', 'max_uninformative_scans'):
            value = getattr(self, name)
            if value < 1 or int(value) != value:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < self.trend_alpha <= 1 or not self.deterioration < 0 < self.improvement:
            raise ValueError('Invalid trend parameters')
        if not 0 < self.cast_angle <= self.max_cast_angle <= 90:
            raise ValueError('Invalid casting angles')
        if not 0 < self.check_angle <= 60 or not 0 < self.scan_step <= 60:
            raise ValueError('Invalid sampling angles')
        if (not self.scan_extents or tuple(sorted(set(self.scan_extents))) != tuple(self.scan_extents)
                or not 0 < self.scan_extents[0] <= self.scan_extents[-1] <= 180):
            raise ValueError('Invalid scan extents')
        if self.contrast_threshold < 0 or self.ambiguity_tolerance < 0:
            raise ValueError('Invalid contrast thresholds')


def acquisition_calibration(codes):
    """Fixed scale from stored codes only; no test views or route labels.

    This is not a probability calibration. The scale is the robust spread of
    pairwise similarities among teaching codes. Memory size can affect it, so
    every condition records the values and uses the same procedure.
    """
    x = np.asarray(codes) > 0
    if x.ndim != 2 or len(x) < 2:
        raise ValueError('At least two teaching codes required')
    x = x.astype(np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    matrix = x @ x.T
    values = matrix[np.triu_indices(len(x), 1)]
    low, high = np.quantile(values, [.1, .9])
    np.fill_diagonal(matrix, -np.inf)
    return dict(scale=float(max(.02, high - low)), low=float(low), high=float(high),
                familiar=float(np.median(matrix.max(axis=1))),
                degenerate=bool(high - low < .02), views=len(x),
                additional_observations=0, procedure='teaching_pairwise_q90_minus_q10_floor_0.02')


class FamiliarityController:
    def __init__(self, calibration, settings=None, phase=1):
        if phase not in (-1, 1):
            raise ValueError('phase must be -1 or 1')
        self.settings = settings or Settings()
        self.scale = float(calibration['scale'])
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Calibration scale must be finite and positive')
        self.phase = phase
        self.state = 'reorient'
        self.anchor = None
        self.travel = None
        self.reference = None
        self.pending = None
        self.trend = 0.
        self.poor = self.good = self.cast_moves = self.cast_attempts = 0
        self.uninformative = self.surge_left = 0
        self.last_check = self.last_explore = 0
        self.decisions = []
        self.transitions = []

    def _state(self, state, step, reason):
        if state != self.state:
            self.transitions.append(dict(step=step, before=self.state, after=state, reason=reason))
        self.state = state

    def _observe(self, sensor, heading):
        value = sensor.observe(heading)
        if value is not None and not math.isfinite(value):
            sensor.stop('invalid_familiarity')
            return None
        return value

    def _peak(self, samples, base):
        # Stable ties prefer the smallest turn; phase resolves left/right ties.
        target = min(samples, key=lambda h: (-samples[h], abs(wrap(h-base)),
                                             -self.phase * wrap(h-base)))
        best = samples[target]
        contrast = (best - float(np.median(list(samples.values())))) / self.scale
        competing = any(abs(wrap(h-target)) > 40 and
                        best-v <= self.settings.ambiguity_tolerance*self.scale
                        for h, v in samples.items())
        supported = contrast >= self.settings.contrast_threshold and not competing
        if not self.settings.use_contrast:
            supported = True
        return target, contrast, supported

    def _scan(self, sensor, base, wide):
        sensor.start_scan()
        samples = {}
        extents = self.settings.scan_extents if wide else (self.settings.check_angle,)
        for extent in extents:
            offsets = list(np.arange(-extent, extent + 1e-6, self.settings.scan_step)) + [0.]
            # Visit nearest unseen heading first; no hypothetical scores.
            targets = {wrap(base + d) for d in offsets}
            while targets - samples.keys():
                target = min(targets - samples.keys(),
                             key=lambda h: (abs(wrap(h-sensor.heading)), -self.phase*wrap(h-base)))
                value = self._observe(sensor, target)
                if value is None:
                    return None
                samples[target] = value
            target, contrast, supported = self._peak(samples, base)
            interior = abs(wrap(target-base)) < extent - 1e-6 or extent == 180
            if supported and interior:
                break
        return dict(target=target, contrast=contrast, supported=bool(supported and interior),
                    samples=[dict(heading=h, familiarity=v) for h, v in samples.items()])

    def _start_cast(self, step, reason):
        self._state('cast', step, reason)
        self.cast_moves = self.cast_attempts = self.good = 0
        self.last_explore = step
        self._next_cast()

    def _next_cast(self):
        index = self.cast_attempts
        angle = min(self.settings.max_cast_angle, self.settings.cast_angle * (1 + index // 2))
        self.travel = wrap(self.anchor + self.phase * (-1 if index % 2 else 1) * angle)
        self.cast_attempts += 1
        self.good = 0

    def step(self, sensor, step):
        s = self.settings
        record = dict(step=step, state_before=self.state, time_s=sensor.time,
                      self_motion_m=sensor.distance, phase=self.phase)
        current = None
        delta = None
        if self.pending is not None:
            before = self.pending
            current = self._observe(sensor, before['reference'])
            if current is None:
                return False
            distance = sensor.distance - before['distance']
            if distance <= 0:
                raise ValueError('A pending comparison requires a completed movement')
            delta = (current - before['value']) / self.scale
            # This is change over an executed move, not an oracle spatial
            # derivative. An unknown displacement or lighting change can enter it.
            self.trend = s.trend_alpha * delta + (1-s.trend_alpha)*self.trend
            record['comparison'] = dict(before=before['value'], after=current,
                reference_heading=before['reference'], movement_heading=before['movement'],
                distance_m=distance, normalized_change=delta,
                change_per_m=(current-before['value'])/distance, trend=self.trend)
            self.pending = None
            if s.use_temporal:
                self.poor = self.poor + 1 if self.trend < s.deterioration else 0
                self.good = self.good + 1 if delta > s.improvement else 0
            if self.state == 'cast':
                self.cast_moves += 1
                if s.use_temporal and self.good >= s.confirmation_moves:
                    self._state('follow', step, 'repeated_translation_improvement')
                    self.surge_left = s.surge_moves
                    self.poor = 0
                    self.last_check = step
                elif self.cast_moves >= s.max_cast_moves:
                    self._state('reorient', step, 'cast_budget')
                elif delta is not None and (not s.use_temporal or delta <= s.improvement):
                    self._next_cast()

        if self.state == 'follow':
            if s.use_temporal and self.poor >= s.poor_moves:
                self._start_cast(step, 'sustained_decline')
            elif s.periodic_exploration and step - self.last_explore >= s.explore_every:
                self._start_cast(step, 'periodic_exploration')
            elif self.surge_left > 0:
                self.surge_left -= 1
            elif step - self.last_check >= s.check_every:
                scan = self._scan(sensor, self.travel, wide=False)
                if scan is None:
                    return False
                record['scan'] = scan
                self.last_check = step
                if scan['supported']:
                    self.anchor = self.travel = scan['target']
                    self.reference = self.anchor
                    current = None  # Reset comparison after a gaze change.
                    self.trend = 0.
                else:
                    self._state('reorient', step, 'ambiguous_local_check')

        if self.state == 'reorient':
            base = sensor.heading if self.anchor is None else self.anchor
            scan = self._scan(sensor, base, wide=True)
            if scan is None:
                return False
            record['scan'] = scan
            if scan['supported']:
                self.uninformative = 0
                self.anchor = self.travel = self.reference = scan['target']
                self.trend = 0.
                self.poor = 0
                self.last_check = step
                self._state('follow', step, 'supported_direction')
            else:
                self.uninformative += 1
                if self.uninformative >= s.max_uninformative_scans:
                    record['termination'] = 'uninformative_views'
                    self.decisions.append(record)
                    sensor.stop('uninformative_views')
                    return False
                if self.anchor is None:
                    self.anchor = self.reference = base
                self._start_cast(step, 'no_directional_support')
            current = None

        # Always acquire the reference view before moving; reuse only a value
        # from this same position and this exact reference heading.
        if current is None:
            current = self._observe(sensor, self.reference)
            if current is None:
                return False
        if not sensor.rotate(self.travel):
            return False
        self.pending = dict(reference=self.reference, value=current,
                            movement=self.travel, distance=sensor.distance)
        record.update(state=self.state, reference_heading=self.reference,
                      movement_heading=self.travel, familiarity=current,
                      trend=self.trend, time_end_s=sensor.time)
        self.decisions.append(record)
        return True


class SensorView:
    """Capability adapter: public interface contains no geometry or renderer.

    This is an API boundary, not a Python security sandbox. The policy never
    accesses the private callbacks or the evaluator's underlying sensor object.
    """
    __slots__ = ('_observe', '_rotate', '_scan', '_stop', 'heading', 'time', 'distance')

    def __init__(self, heading, time, distance, observe, rotate, scan, stop):
        self.heading, self.time, self.distance = heading, time, distance
        self._observe, self._rotate, self._scan, self._stop = observe, rotate, scan, stop

    def observe(self, heading):
        value, self.heading, self.time = self._observe(heading)
        return value

    def rotate(self, heading):
        result, self.heading, self.time = self._rotate(heading)
        return result

    def start_scan(self):
        self._scan()

    def stop(self, reason):
        self._stop(reason)


class ControllerAdapter:
    """Attach a restricted policy to the unchanged fixed-step evaluator."""
    def __init__(self, policy):
        self.policy = policy

    def step(self, sensor, step):
        def observe(heading):
            value = sensor.observe(heading)
            return value, sensor.heading, sensor.time

        def rotate(heading):
            result = sensor.rotate(heading)
            return result, sensor.heading, sensor.time

        def scan():
            sensor.scan_bouts += 1

        def stop(reason):
            sensor.termination = reason

        view = SensorView(sensor.heading, sensor.time, (step-1)*.1, observe, rotate, scan, stop)
        return self.policy.step(view, step)
