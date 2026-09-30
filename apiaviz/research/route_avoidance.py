"""A camera-only detour with a remembered visual direction and bounded handoff.

No route/world coordinates enter the reflex or navigator. The evaluator keeps
the original physics, teaching/readout and budgets. Original experiments remain
reproducible through their unchanged modules.
"""
import numpy as np

from .active_navigation import Observations, segment_collision
from .avoidance_navigation import VisualLocomotion
from .familiarity_controller import SensorView
from .mechanisms import polyline_distance
from .visual_avoidance import Settings, VisualAvoidance, wrap
from .evaluation_safety import safe_displacement


class DetourReflex(VisualAvoidance):
    """Hold the motor intention through an obstacle; release after visual clearance.

    All distances are commanded self-motion. Neither contact nor successful
    movement, a map, depth buffer or route location is used as a steering cue.
    """
    def __init__(self, settings=Settings(), phase=1, clear_m=.06, max_detour_m=2.):
        super().__init__(settings, phase)
        if not 0 < clear_m < max_detour_m:
            raise ValueError('Require 0 < clear distance < detour budget')
        self.clear_m, self.max_detour_m = clear_m, max_detour_m
        self.active = False
        self.goal = None
        self.clear_path = self.detour_path = self.path = 0.
        self.entries = self.exits = 0
        self.resume = None
        self.last_clear = False
        self.transitions = []

    def choose(self, desired_heading):
        goal = self.goal if self.active else desired_heading
        heading, stride, record = super().choose(goal)
        if record['state'] == 'avoid' and not self.active:
            self.active, self.goal = True, float(goal)
            self.clear_path = self.detour_path = 0.
            self.entries += 1
            self.transitions.append(dict(kind='detour_start', path_m=self.path, heading=self.goal))
        self.last_clear = record['state'] == 'clear'
        record.update(behaviour='detour' if self.active else 'route', detour=self.entries,
                      held_heading=float(goal))
        return heading, stride, record

    def moved(self, before, after, heading, distance):
        diagnostic = super().moved(before, after, heading, distance)
        self.path += distance
        if self.active:
            self.detour_path += distance
            self.clear_path = self.clear_path+distance if self.last_clear else 0.
            reason = ('visual_clearance' if self.clear_path >= self.clear_m-1e-10 else
                      'detour_budget' if self.detour_path >= self.max_detour_m-1e-10 else None)
            if reason:
                self.active = False
                self.exits += 1
                self.resume = reason
                self.transitions.append(dict(kind='detour_end', path_m=self.path,
                    reason=reason, detour_path_m=self.detour_path, heading=self.goal))
        return diagnostic


class SamplingControl(VisualAvoidance):
    """Acquire/process the same camera pairs, with evasive steering disabled."""
    active = False
    resume = None
    entries = exits = 0

    def choose(self, desired_heading):
        stride = self.settings.probe_m if self.frame_heading is None else self.settings.stride_m
        return desired_heading, stride, dict(state='sampling_control', behaviour='route',
            desired_heading=float(desired_heading), heading=float(desired_heading), points=len(self.points))


class RouteNavigator:
    """Persistent navigator; the scheduler supplies commanded self-motion."""
    def __init__(self, policy):
        self.policy = policy
        self.steps = 0
        self.reacquisitions = 0
        self.handoffs = []

    def reacquire(self, reason, path):
        p = self.policy
        # Keep the last visually supported anchor and the passing/search phase.
        # Leave an interrupted cast explicitly, instead of clearing its temporal
        # sample forever and accidentally disabling the cast termination rule.
        p.pending = None
        p.trend = 0.
        p.poor = p.good = 0
        p.cast_moves = p.cast_attempts = 0
        p.surge_left = 0
        if p.anchor is not None:
            p.travel = p.reference = p.anchor
            p._state('follow', self.steps, 'detour_visual_reacquisition')
        else:
            p._state('reorient', self.steps, 'detour_without_visual_anchor')
        p.last_check = self.steps-p.settings.check_every
        p.last_explore = self.steps
        self.reacquisitions += 1
        self.handoffs.append(dict(reason=reason, path_m=path, anchor=p.anchor))

    def step(self, sensor, distance, locomotion_step):
        def observe(h):
            value = sensor.observe(h)
            return value, sensor.heading, sensor.time
        def rotate(h):
            ok = sensor.rotate(h)
            return ok, sensor.heading, sensor.time
        def scan(): sensor.scan_bouts += 1
        def stop(reason): sensor.termination = reason
        view = SensorView(sensor.heading, sensor.time, distance, observe, rotate, scan, stop)
        self.steps += 1
        result = self.policy.step(view, self.steps)
        if self.policy.decisions:
            self.policy.decisions[-1]['locomotion_step'] = locomotion_step
        return result


def evaluate(route, headings, scorer, controller_factory, scenario, settings,
             stage2, bounds, obstacles=(), avoidance_settings=Settings(), phase=1,
             mode='detour', clear_m=.06, max_detour_m=2., safety=None, progress=None):
    if mode not in ('detour','sampling_control'):
        raise ValueError(mode)
    route, headings = np.asarray(route), np.asarray(headings)
    h = np.deg2rad(headings[0])
    release_vector = scenario['lateral']*np.array([-np.sin(h),np.cos(h)])
    position = route[0]+release_vector
    release=None
    if safety is not None:
        position,release=safe_displacement(route[0],release_vector,bounds,obstacles,safety)
    sensor = Observations(scorer,position,headings[0]+scenario['heading'],settings)
    if release is not None:
        sensor.events.append(dict(kind='release',time_s=0.,**release))
    camera = lambda pos,yaw: scorer.world.scan(pos,[yaw])[0].permute(1,2,0).numpy()
    motion = VisualLocomotion(sensor,camera,bounds,obstacles,avoidance_settings,phase,safety=safety,
        arrival_test=(lambda pos:np.linalg.norm(pos-route[-1])<=.2) if safety else None)
    reflex = (DetourReflex(avoidance_settings,phase,clear_m,max_detour_m) if mode=='detour'
              else SamplingControl(avoidance_settings,phase))
    motion.policy = reflex
    navigator = RouteNavigator(controller_factory().policy)
    trace=[]
    termination=None
    xmin,xmax,ymin,ymax=bounds
    inside=lambda p: xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
    if segment_collision(position,position,obstacles): termination='invalid_release_rock'
    if not inside(position): termination='invalid_release_bounds'
    initial_position=position.copy()
    recovery_origin=(0.,0.) if np.linalg.norm(position-route[0])>1e-12 else None
    recovered=None
    recovery_streak=0
    disturbance_applied=False
    disturbance_attempted=False
    kick_record=None
    for step in range(1,stage2['max_steps']+1):
        if progress is not None:
            progress(step-1,sensor.time,sensor.count)
        if termination: break
        if np.linalg.norm(sensor.position-route[-1]) <= .2:
            termination='arrival'; break
        if scenario['kick'] and step == stage2['kick_before_step']:
            h=np.deg2rad(headings[len(headings)//2])
            displacement=scenario['kick']*np.array([-np.sin(h),np.cos(h)])
            disturbance_attempted=True
            if safety is not None:
                sensor.position,kick_record=safe_displacement(sensor.position,displacement,bounds,obstacles,safety)
                disturbance_applied=kick_record['applied_distance_m']>1e-12
                sensor.events.append(dict(kind='displacement',time_s=sensor.time,**kick_record))
            else:
                sensor.position += displacement
                disturbance_applied=True
                sensor.events.append(dict(kind='displacement',position=sensor.position.tolist(),
                    displacement=displacement.tolist(),time_s=sensor.time))
            if disturbance_applied:
                recovery_origin=(motion.path,sensor.time)
                recovery_streak=0
                recovered=None
            if not inside(sensor.position): termination='displacement_outside_field'; break
            if segment_collision(sensor.position,sensor.position,obstacles):
                termination='displacement_into_rock'; break
            if safety is not None and np.linalg.norm(sensor.position-route[-1]) <= .2:
                termination='arrival'; break
        # React only to internal visual/motor state, never evaluator geometry.
        if not reflex.active:
            if reflex.resume:
                navigator.reacquire(reflex.resume,motion.path)
                sensor.events.append(dict(kind='route_reacquisition',reason=reflex.resume,
                    time_s=sensor.time,path_m=motion.path,anchor=navigator.policy.anchor))
                reflex.resume=None
            if not navigator.step(sensor,motion.commanded_path,step):
                termination=sensor.termination; break
            desired=sensor.heading
        else:
            desired=reflex.goal
        before_path=motion.path
        before_command=motion.commanded_path
        ok,diverted=motion.advance(desired)
        if not ok and (safety is None or motion.commanded_path<=before_command):
            termination=sensor.termination; break
        distance=float(polyline_distance([sensor.position],route)[0])
        trace.append(dict(step=step,position=sensor.position.tolist(),heading=sensor.heading,
            time_s=sensor.time,path_m=motion.path,polyline_m=distance,
            translated_m=motion.path-before_path,
            partial=not ok,
            deviation_m=float(np.linalg.norm(route-sensor.position,axis=1).min()),
            observations=sensor.count,scan_bouts=sensor.scan_bouts,diverted=diverted,
            behaviour='detour' if reflex.active else 'route'))
        if recovery_origin is not None and recovered is None:
            recovery_streak=recovery_streak+1 if distance <= stage2['recovery_radius_m'] and motion.path > before_path+1e-12 else 0
            if recovery_streak >= stage2['recovery_consecutive_steps']:
                recovered=dict(path_m=motion.path-recovery_origin[0],time_s=sensor.time-recovery_origin[1])
        if not ok:
            termination=sensor.termination; break
    if termination is None:
        termination='arrival' if np.linalg.norm(sensor.position-route[-1]) <= .2 else 'step_budget'
    points=np.asarray([r['position'] for r in motion.trace])
    distances=polyline_distance(points,route) if len(points) else []
    weights=[r['stride_m'] for r in motion.trace]
    return dict(reached_nest=termination=='arrival',termination=termination,
        final_nest_distance_m=float(np.linalg.norm(sensor.position-route[-1])),steps=len(trace),
        path_length_m=motion.path,time_s=sensor.time,observations=sensor.count,
        commanded_path_m=motion.commanded_path,blocked_proposals=motion.blocked_proposals,
        avoidance_observations=motion.camera_views,scan_bouts=sensor.scan_bouts,
        rotation_deg=sensor.rotation_deg,motor_state_resets=0,corrective_resets=0,
        route_reacquisitions=navigator.reacquisitions,detour_episodes=reflex.entries,
        detour_completions=reflex.exits,
        detour_transitions=getattr(reflex,'transitions',[]),
        polyline_mean_m=float(np.average(distances,weights=weights)) if sum(weights) > 0 else None,
        recovered=recovered is not None,recovery_applicable=recovery_origin is not None,
        recovery_path_m=recovered['path_m'] if recovered else None,
        recovery_time_s=recovered['time_s'] if recovered else None,
        disturbance_applied=disturbance_applied,trace=trace,microtrace=motion.trace,events=sensor.events,
        disturbance_attempted=disturbance_attempted,initial_position=initial_position.tolist(),
        evaluation_safety=safety.configuration() if safety else None,release=release,
        displacement=kick_record,perturbation_adjusted=bool((release and release['adjusted']) or (kick_record and kick_record['adjusted'])),
        decisions=[dict(episode=0,**d) for d in navigator.policy.decisions])
