"""Causal, time-accounted visual navigation; no route information enters policies.

The evaluator has ground truth for scoring, endpoint stopping and imposed
displacements. Controllers receive only observed familiarity and their own yaw.
Turning is finite-rate and stationary, followed by straight translation. These
are controlled kinematic experiments, not a complete animal locomotor model.
"""
import math

import numpy as np
import torch

from .mechanisms import polyline_distance
from .navigation import choose_heading
from .study import encode


def wrap(angle):
    return (angle + 180.) % 360. - 180.


def segment_collision(a, b, obstacles):
    """Conservative 2D rock discs from scene metadata, including segment interior."""
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    for rock in obstacles:
        centre = np.array([rock['x'], rock['y']])
        t = np.clip(np.dot(centre - a, delta) / max(float(delta @ delta), 1e-15), 0., 1.)
        if np.linalg.norm(centre - (a + t * delta)) <= rock['conservative_radius_m']:
            return True
    return False


def confidence(value, calibration):
    return float(np.clip((value - calibration['low']) / calibration['range'], 0., 1.))


def calibrate(model, memory, codes, world, positions, headings):
    normalized = (codes > 0).float()
    normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-6)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-float('inf'))
    positive = similarity.max(dim=1).values.numpy()
    indices = np.unique(np.linspace(0, len(headings) - 1, min(12, len(headings))).round().astype(int))
    images = world.render(np.repeat(np.asarray(positions)[indices], 2, axis=0),
                          (np.asarray(headings)[indices, None] + [-60., 60.]).ravel())
    negative = -memory(encode(model, images)).detach().numpy()
    low, high = float(np.median(negative)), float(np.median(positive))
    return dict(low=low, high=high, range=max(.05, high - low),
                positive=positive.tolist(), negative=negative.tolist(),
                calibration_stations=indices.tolist(), degenerate=bool(high - low < .05))


class Observations:
    """Own-state and sensor interface; no endpoint or learned-route geometry."""
    def __init__(self, scorer, position, heading, settings, timed=True):
        self.scorer = scorer
        self.position = np.asarray(position, dtype=float).copy()
        self.heading = float(heading)
        self.settings = settings
        self.time = 0.
        self.count = 0
        self.scan_bouts = 0
        self.rotation_deg = 0.
        self.events = []
        self.termination = None
        self.timed = timed

    def rotate(self, target):
        turn = wrap(target - self.heading)
        duration = abs(turn) / self.settings['yaw_speed_deg_s']
        if self.timed and self.time + duration > self.settings['time_budget_s'] + 1e-12:
            self.termination = 'time_budget'
            return False
        before = self.heading
        self.heading += turn
        self.time += duration
        self.rotation_deg += abs(turn)
        self.events.append(dict(kind='turn', time_s=self.time, from_heading=before,
                                heading=self.heading, angle_deg=turn, duration_s=duration))
        return True

    def observe(self, target):
        if self.count >= self.settings['observation_budget']:
            self.termination = 'observation_budget'
            return None
        duration = abs(wrap(target - self.heading)) / self.settings['yaw_speed_deg_s'] + self.settings['observation_s']
        if self.timed and self.time + duration > self.settings['time_budget_s'] + 1e-12:
            self.termination = 'time_budget'
            return None
        if not self.rotate(target): return None
        score = float(self.scorer(self.position, np.array([self.heading]))[0])
        self.count += 1
        self.time += self.settings['observation_s']
        value = -score if np.isfinite(score) else 0.
        self.events.append(dict(kind='observation', time_s=self.time,
                                position=self.position.tolist(), heading=self.heading, familiarity=value))
        return value


class Exhaustive:
    def step(self, sensor, step):
        base = sensor.heading
        offsets = np.arange(60., -61., -10.)
        values = []
        sensor.scan_bouts += 1
        for offset in offsets:
            value = sensor.observe(base + offset)
            if value is None: return False
            values.append(value)
        winner, _ = choose_heading(-np.array(values), offsets)
        return sensor.rotate(base + offsets[winner])


class Active:
    def __init__(self, settings, calibration):
        self.settings, self.calibration = settings, calibration
        self.poor = 0
        self.last_scan = -settings['scan_cooldown_steps']
        self.sign = 1.

    def step(self, sensor, step):
        base = sensor.heading
        value = sensor.observe(base)
        if value is None: return False
        c = confidence(value, self.calibration)
        self.poor = self.poor + 1 if c < self.settings['scan_threshold'] else 0
        scan = self.poor >= self.settings['poor_views'] and step - self.last_scan >= self.settings['scan_cooldown_steps']
        if scan:
            sensor.scan_bouts += 1
            offsets = np.array([0.] + self.settings['scan_offsets'])
            values = [value]
            for offset in offsets[1:]:
                value = sensor.observe(base + offset)
                if value is None: return False
                values.append(value)
            winner, _ = choose_heading(-np.array(values), offsets)
            target = base + offsets[winner]
            self.last_scan, self.poor = step, 0
        else:
            amplitude = self.settings['min_turn_deg'] + (self.settings['max_turn_deg'] - self.settings['min_turn_deg']) * (1. - c) ** 2
            target = base + self.sign * amplitude
        self.sign *= -1.
        return sensor.rotate(target)


class Straight:
    def step(self, sensor, step):
        return True


def evaluate(route, headings, scorer, controller, scenario, settings, stage2,
             bounds, obstacles=(), timed=True):
    route, headings = np.asarray(route), np.asarray(headings)
    initial_normal = np.array([-np.sin(np.radians(headings[0])), np.cos(np.radians(headings[0]))])
    position = route[0] + scenario['lateral'] * initial_normal
    sensor = Observations(scorer, position, headings[0] + scenario['heading'], settings, timed)
    trace = []
    xmin, xmax, ymin, ymax = bounds
    termination = None
    if segment_collision(position, position, obstacles): termination = 'invalid_release_rock'
    def inside(p): return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
    if not inside(position): termination = 'invalid_release_bounds'
    recovery_origin = (0, 0.) if scenario['lateral'] else None
    recovered = None
    recovery_streak = 0
    disturbance_applied = False
    for step in range(1, stage2['max_steps'] + 1):
        if termination: break
        if np.linalg.norm(sensor.position - route[-1]) <= .2:
            termination = 'arrival'
            break
        if scenario['kick'] and step == stage2['kick_before_step']:
            # Externally imposed perturbation; neither its vector nor route
            # information is given to the visual policy.
            h = np.radians(headings[len(headings) // 2])
            displacement = scenario['kick'] * np.array([-np.sin(h), np.cos(h)])
            sensor.position += displacement
            disturbance_applied = True
            recovery_origin = (len(trace), sensor.time)
            recovery_streak = 0
            sensor.events.append(dict(kind='displacement', time_s=sensor.time,
                                      position=sensor.position.tolist(), displacement=displacement.tolist()))
            if not inside(sensor.position):
                termination = 'displacement_outside_field'
                break
            if segment_collision(sensor.position, sensor.position, obstacles):
                termination = 'displacement_into_rock'
                break
        if not controller.step(sensor, step):
            termination = sensor.termination
            break
        duration = .1 / settings['speed_m_s']
        if timed and sensor.time + duration > settings['time_budget_s'] + 1e-12:
            termination = 'time_budget'
            break
        rad = math.radians(sensor.heading)
        proposed = sensor.position + .1 * np.array([math.cos(rad), math.sin(rad)])
        if not inside(proposed):
            termination = 'field_boundary'
            break
        if segment_collision(sensor.position, proposed, obstacles):
            termination = 'rock_collision'
            break
        sensor.position = proposed
        sensor.time += duration
        distance = float(polyline_distance([proposed], route)[0])
        entry = dict(step=step, position=proposed.tolist(), heading=sensor.heading,
                     time_s=sensor.time, polyline_m=distance,
                     deviation_m=float(np.linalg.norm(route - proposed, axis=1).min()),
                     observations=sensor.count, scan_bouts=sensor.scan_bouts)
        trace.append(entry)
        if recovery_origin is not None and recovered is None:
            recovery_streak = recovery_streak + 1 if distance <= stage2['recovery_radius_m'] else 0
            if recovery_streak >= stage2['recovery_consecutive_steps']:
                # Completion of a sustained three-position return; includes all
                # motion since release/kick, including confirmation steps.
                recovered = dict(path_m=(len(trace) - recovery_origin[0]) * .1,
                                 time_s=sensor.time - recovery_origin[1])
    if termination is None:
        termination = 'arrival' if np.linalg.norm(sensor.position - route[-1]) <= .2 else 'step_budget'
    points = np.array([t['position'] for t in trace])
    return dict(reached_nest=termination == 'arrival', termination=termination,
                final_nest_distance_m=float(np.linalg.norm(sensor.position - route[-1])),
                steps=len(trace), path_length_m=len(trace) * .1, time_s=sensor.time,
                observations=sensor.count, scan_bouts=sensor.scan_bouts,
                rotation_deg=sensor.rotation_deg,
                polyline_mean_m=float(np.mean([t['polyline_m'] for t in trace])) if trace else None,
                route_deviation_mean_m=float(np.mean([t['deviation_m'] for t in trace])) if trace else None,
                recovered=recovered is not None, recovery_applicable=recovery_origin is not None,
                recovery_path_m=recovered['path_m'] if recovered else None,
                recovery_time_s=recovered['time_s'] if recovered else None,
                disturbance_applied=disturbance_applied, corrective_resets=0,
                trace=trace, events=sensor.events)
