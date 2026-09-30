"""Camera-only avoidance integration; original fixed-step evaluator is intact."""
import math

import numpy as np

from .active_navigation import Observations, segment_collision
from .mechanisms import polyline_distance
from .visual_avoidance import VisualAvoidance, Settings


class VisualLocomotion:
    """Evaluator owns physics; the reflex receives images and own motion only."""
    def __init__(self, sensor, camera, bounds, obstacles, settings=Settings(), phase=1):
        self.sensor, self.camera = sensor, camera
        self.bounds, self.obstacles = bounds, obstacles
        self.policy = VisualAvoidance(settings, phase)
        self.path = 0.
        self.trace = []
        self.camera_views = 0
        self.cache = None

    def image(self):
        s = self.sensor
        key = (*s.position, s.heading)
        if self.cache is not None and self.cache[0] == key:
            return self.cache[1]
        if s.count >= s.settings['observation_budget']:
            s.termination = 'observation_budget'
            return None
        if s.timed and s.time+s.settings['observation_s'] > s.settings['time_budget_s']+1e-12:
            s.termination = 'time_budget'
            return None
        image = self.camera(s.position.copy(), s.heading)
        s.time += s.settings['observation_s']
        s.count += 1
        self.camera_views += 1
        s.events.append(dict(kind='avoidance_observation', position=s.position.tolist(),
                             heading=s.heading, time_s=s.time))
        self.cache = (key, image)
        return image

    def advance(self, desired_heading, length=.1):
        moved, diverted = 0., False
        s = self.sensor
        while moved < length-1e-10:
            heading, stride, decision = self.policy.choose(desired_heading)
            stride = min(stride, length-moved)
            diverted = diverted or bool(abs((heading-desired_heading+180)%360-180) > 1e-8)
            if not s.rotate(heading):
                return False, diverted
            before = self.image()
            if before is None:
                return False, diverted
            duration = stride/s.settings['speed_m_s']
            # Reserve the post-movement visual observation before translating.
            if s.count >= s.settings['observation_budget']:
                s.termination = 'observation_budget'
                return False, diverted
            if s.timed and s.time+duration+s.settings['observation_s'] > s.settings['time_budget_s']+1e-12:
                s.termination = 'time_budget'
                return False, diverted
            delta = stride*np.array([math.cos(math.radians(heading)), math.sin(math.radians(heading))])
            proposed = s.position+delta
            xmin, xmax, ymin, ymax = self.bounds
            if not (xmin <= proposed[0] <= xmax and ymin <= proposed[1] <= ymax):
                s.termination = 'field_boundary'
                return False, diverted
            # This test ONLY prevents penetration and records failure. It never
            # chooses a turn, supplies a range, or reveals which side is clear.
            if segment_collision(s.position, proposed, self.obstacles):
                s.termination = 'rock_collision'
                return False, diverted
            s.position = proposed
            s.time += duration
            self.path += stride
            moved += stride
            after = self.image()
            if after is None:
                raise AssertionError('Reserved camera observation unavailable')
            diagnostic = self.policy.moved(before, after, heading, stride)
            self.trace.append({**decision, 'position':s.position.tolist(), 'heading':s.heading,
                'time_s':s.time, 'path_m':self.path, 'stride_m':stride, 'flow':diagnostic})
        return True, diverted


def evaluate(route, headings, scorer, controller_factory, scenario, settings,
             stage2, bounds, obstacles=(), avoidance_settings=Settings(), phase=1):
    """Same teaching/readout, with shared visually guided substeps for locomotion.

    A detour invalidates the navigator's pending straight-movement comparison.
    Restart only its transient motor state after that macro move; never retrain
    or reset the visual memory. Executed path and every camera view are charged.
    """
    route, headings = np.asarray(route), np.asarray(headings)
    h = np.deg2rad(headings[0])
    position = route[0]+scenario['lateral']*np.array([-np.sin(h), np.cos(h)])
    sensor = Observations(scorer, position, headings[0]+scenario['heading'], settings)
    def camera(pos, heading):
        return scorer.world.scan(pos, [heading])[0].permute(1, 2, 0).numpy()
    motion = VisualLocomotion(sensor, camera, bounds, obstacles, avoidance_settings, phase)
    controller = controller_factory()
    episode_step, resets = 0, 0
    episodes = [controller]
    trace = []
    termination = None
    xmin, xmax, ymin, ymax = bounds
    inside = lambda p: xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
    if segment_collision(position, position, obstacles):
        termination = 'invalid_release_rock'
    if not inside(position):
        termination = 'invalid_release_bounds'
    recovery_origin = (0., 0.) if scenario['lateral'] else None
    recovery_streak, recovered, disturbance_applied = 0, None, False
    for step in range(1, stage2['max_steps']+1):
        if termination:
            break
        if np.linalg.norm(sensor.position-route[-1]) <= .2:
            termination = 'arrival'
            break
        if scenario['kick'] and step == stage2['kick_before_step']:
            h = np.deg2rad(headings[len(headings)//2])
            displacement = scenario['kick']*np.array([-np.sin(h), np.cos(h)])
            sensor.position += displacement
            disturbance_applied = True
            recovery_origin = (motion.path, sensor.time)
            recovery_streak = 0
            sensor.events.append(dict(kind='displacement', position=sensor.position.tolist(),
                                      displacement=displacement.tolist(), time_s=sensor.time))
            if not inside(sensor.position):
                termination = 'displacement_outside_field'
                break
            if segment_collision(sensor.position, sensor.position, obstacles):
                termination = 'displacement_into_rock'
                break
        episode_step += 1
        if not controller.step(sensor, episode_step):
            termination = sensor.termination
            break
        ok, diverted = motion.advance(sensor.heading)
        if diverted:
            resets += 1
            controller = controller_factory()
            episodes.append(controller)
            episode_step = 0
            sensor.events.append(dict(kind='motor_state_reset', reason='visual_detour', time_s=sensor.time))
        distance = float(polyline_distance([sensor.position], route)[0])
        if not ok:
            termination = sensor.termination
            break
        trace.append(dict(step=step, position=sensor.position.tolist(), heading=sensor.heading,
            time_s=sensor.time, path_m=motion.path, polyline_m=distance,
            deviation_m=float(np.linalg.norm(route-sensor.position, axis=1).min()),
            observations=sensor.count, scan_bouts=sensor.scan_bouts, diverted=diverted))
        if recovery_origin is not None and recovered is None:
            recovery_streak = recovery_streak+1 if distance <= stage2['recovery_radius_m'] else 0
            if recovery_streak >= stage2['recovery_consecutive_steps']:
                recovered = dict(path_m=motion.path-recovery_origin[0], time_s=sensor.time-recovery_origin[1])
    if termination is None:
        termination = 'arrival' if np.linalg.norm(sensor.position-route[-1]) <= .2 else 'step_budget'
    points = np.asarray([r['position'] for r in motion.trace])
    distances = polyline_distance(points, route) if len(points) else []
    weights = np.asarray([r['stride_m'] for r in motion.trace])
    return dict(reached_nest=termination == 'arrival', termination=termination,
        final_nest_distance_m=float(np.linalg.norm(sensor.position-route[-1])),
        steps=len(trace), path_length_m=motion.path, time_s=sensor.time,
        observations=sensor.count, avoidance_observations=motion.camera_views,
        scan_bouts=sensor.scan_bouts, rotation_deg=sensor.rotation_deg,
        motor_state_resets=resets, corrective_resets=0,
        polyline_mean_m=float(np.average(distances, weights=weights)) if len(points) else None,
        recovered=recovered is not None, recovery_applicable=recovery_origin is not None,
        recovery_path_m=recovered['path_m'] if recovered else None,
        recovery_time_s=recovered['time_s'] if recovered else None,
        disturbance_applied=disturbance_applied, trace=trace, microtrace=motion.trace,
        events=sensor.events,
        decisions=[dict(episode=i, **d) for i,c in enumerate(episodes)
                   for d in getattr(getattr(c, 'policy', None), 'decisions', [])])
