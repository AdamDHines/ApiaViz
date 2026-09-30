import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from apiaviz.research.active_navigation import Observations
from apiaviz.research.avoidance_navigation import VisualLocomotion
from apiaviz.research.collision_geometry import RockGeometry
from apiaviz.research.evaluation_safety import EvaluationSafety, safe_displacement, inside_body
from apiaviz.research.familiarity_controller import FamiliarityController, ControllerAdapter
from apiaviz.research.motor_feedback import evaluate, FeedbackNavigator
from apiaviz.research.route_full_audit import check_motion
from apiaviz.research.route_full import perturbation_analysis
from apiaviz.research.visual_avoidance import Settings, VisualAvoidance


def rectangle(x0,x1,y0,y1):
    return [[[x0,y0],[x1,y0],[x1,y1]],[[x0,y0],[x1,y1],[x0,y1]]]


SENSOR=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,time_budget_s=2.,observation_budget=100)
STAGE=dict(max_steps=10,kick_before_step=1,recovery_radius_m=.1,recovery_consecutive_steps=3)
BOUNDS=[-3,3,-3,3]


class Scorer:
    def __init__(self):
        self.positions=[]
        self.world=SimpleNamespace(scan=self.scan)
    def scan(self,pos,hs):
        self.positions.append(np.asarray(pos).copy())
        return torch.full((len(hs),3,51,199),.5)
    def __call__(self,pos,hs):
        self.positions.append(np.asarray(pos).copy())
        return np.full(len(hs),-.5)


def controller():
    p=FamiliarityController(dict(scale=.3))
    p.state='follow'; p.anchor=p.travel=p.reference=0.
    return ControllerAdapter(p)


class EvaluationSafetyTests(unittest.TestCase):
    def test_kick_is_clipped_before_rock_and_never_tunnels(self):
        mesh=RockGeometry([('wall',rectangle(-.1,.1,.2,.3))],.005)
        for distance in (.25,.5,4.):
            pos,record=safe_displacement([0,0],[0,distance],BOUNDS,mesh)
            self.assertAlmostEqual(pos[1],.195-1e-5,places=8)
            self.assertTrue(record['adjusted'])
            self.assertFalse(mesh.intersects([0,0],pos))
            self.assertEqual(record['requested_distance_m'],distance)
            self.assertEqual(pos[0],0.)
            self.assertEqual(record['constraint'],'rock_contact')

    def test_unobstructed_direction_preserved_and_body_stays_inside_field(self):
        mesh=RockGeometry([],.005)
        pos,record=safe_displacement([0,0],[.3,.4],[-1,1,-1,1],mesh)
        np.testing.assert_array_equal(pos,[.3,.4]); self.assertFalse(record['adjusted'])
        pos,record=safe_displacement([0,0],[2,1],[-1,1,-1,1],mesh)
        self.assertTrue(inside_body(pos,[-1,1,-1,1],mesh))
        self.assertAlmostEqual(pos[0]/pos[1],2.)
        self.assertEqual(record['constraint'],'field_boundary')

    def test_no_room_skips_kick_without_moving_and_invalid_anchor_is_setup_error(self):
        mesh=RockGeometry([('wall',rectangle(-.1,.1,.005001,.2))],.005)
        pos,record=safe_displacement([0,0],[0,.5],BOUNDS,mesh)
        np.testing.assert_array_equal(pos,[0,0])
        self.assertEqual(record['applied_distance_m'],0.)
        with self.assertRaisesRegex(ValueError,'anchor'):
            safe_displacement([0,.1],[0,.5],BOUNDS,mesh)

    def test_release_and_kick_never_acquire_camera_inside_a_rock(self):
        mesh=RockGeometry([('wall',rectangle(-.1,.1,.2,.3))],.005)
        for scenario in (dict(lateral=.25,heading=0.,kick=0.),dict(lateral=0.,heading=0.,kick=.25)):
            scorer=Scorer()
            row=evaluate([[0,0],[2,0]],[0,0],scorer,controller,scenario,SENSOR,STAGE,BOUNDS,mesh,safety=EvaluationSafety())
            self.assertNotIn(row['termination'],('displacement_into_rock','invalid_release_rock'))
            self.assertTrue(row['perturbation_adjusted'])
            self.assertTrue(all(not mesh.intersects(p,p) for p in scorer.positions))
            self.assertEqual(row['corrective_resets'],0)
            self.assertEqual(row['motor_state_resets'],0)
            detail={k:row[k] for k in ('events','microtrace')}
            check_motion([0,scenario['lateral']],detail,row,
                dict(evaluation=STAGE,avoidance_settings=dict(stride_m=.02)),
                dict(route=[[0,0],[2,0]],headings=[0,0],world_bounds_m=BOUNDS),mesh,scenario)
            json.dumps(row,allow_nan=False)

    def test_skipped_kick_counts_attempt_not_applied_displacement(self):
        mesh=RockGeometry([('wall',rectangle(-.1,.1,.005001,.2))],.005)
        scenario=dict(lateral=0.,heading=0.,kick=.5)
        row=evaluate([[0,0],[2,0]],[0,0],Scorer(),controller,scenario,SENSOR,STAGE,BOUNDS,mesh,safety=EvaluationSafety())
        self.assertTrue(row['disturbance_attempted'])
        self.assertFalse(row['disturbance_applied'])
        self.assertFalse(row['recovery_applicable'])
        check_motion([0,0],row,row,dict(evaluation=STAGE,avoidance_settings=dict(stride_m=.02)),
            dict(route=[[0,0],[2,0]],headings=[0,0],world_bounds_m=BOUNDS),mesh,scenario)

    def test_arrival_during_partial_macro_move_wins_over_next_time_budget_check(self):
        scenario=dict(lateral=0.,heading=0.,kick=0.)
        args=([[0,0],[.202,0]],[0,0],Scorer(),controller,scenario,
              dict(SENSOR,time_budget_s=.25),STAGE,BOUNDS,RockGeometry([],.005))
        historical=evaluate(*args)
        self.assertEqual(historical['termination'],'time_budget')
        self.assertLess(historical['final_nest_distance_m'],.2)
        fixed=evaluate(*args,safety=EvaluationSafety())
        self.assertEqual(fixed['termination'],'arrival')
        self.assertAlmostEqual(fixed['path_length_m'],.005)
        self.assertTrue(fixed['trace'][-1]['partial'])
        self.assertLessEqual(fixed['time_s'],.25)

    def test_field_contact_blocks_and_is_not_a_hidden_steering_signal(self):
        motions=[]
        for bounds in ([-1,1,-1,1],[-1,.01,-1,1]):
            sensor=Observations(None,[0,0],0.,SENSOR)
            motion=VisualLocomotion(sensor,lambda p,h:np.zeros((51,199,3)),bounds,RockGeometry([],.005),safety=EvaluationSafety())
            self.assertTrue(motion.advance(0)[0]); motions.append(motion)
        clear,blocked=motions
        self.assertGreater(blocked.blocked_proposals,0)
        self.assertEqual(clear.policy.decisions,blocked.policy.decisions)
        self.assertEqual(clear.sensor.count,blocked.sensor.count)
        self.assertAlmostEqual(clear.sensor.time,blocked.sensor.time)
        self.assertEqual(next(e for e in blocked.sensor.events if e['kind']=='blocked_proposal')['reason'],'field_boundary')

    def test_safe_kick_into_goal_is_scored_before_zero_sensor_budget(self):
        row=evaluate([[0,0],[0,.65]],[0,0],Scorer(),controller,
            dict(lateral=0.,heading=0.,kick=.5),dict(SENSOR,observation_budget=0),
            STAGE,BOUNDS,RockGeometry([],.005),safety=EvaluationSafety())
        self.assertEqual(row['termination'],'arrival')
        self.assertEqual(row['observations'],0)
        self.assertEqual(row['path_length_m'],0.)
        self.assertTrue(row['disturbance_applied'])

    def test_dose_report_keeps_failures_and_excludes_entire_paired_blocks(self):
        rows=[]
        for seed in (1,2):
            for method in ('linear_colour','sobel_colour','ardin_input'):
                for phase in (-1,1):
                    shortened=(seed==1 and method=='sobel_colour' and phase==1)
                    rows.append(dict(world='fixture',seed=seed,method=method,phase=phase,
                        controller='familiarity',scenario='kick_left50',reached_nest=not shortened,
                        release=dict(applied_distance_m=0.,adjusted=False),
                        displacement=dict(applied_distance_m=.1 if shortened else .5,adjusted=shortened)))
        report=perturbation_analysis(rows,dict(scenarios=[dict(name='kick_left50',lateral=0.,kick=.5)]))
        self.assertEqual(sum(g['n'] for g in report['strata']),12)
        self.assertEqual(sum(g['arrivals'] for g in report['strata']),11)
        shortened=next(g for g in report['strata'] if g['kick_status']=='shortened')
        self.assertEqual(shortened['applied_dose']['displacement'],dict(min_m=.1,max_m=.1))
        sensitivity=report['full_dose_sensitivity']
        self.assertEqual(sensitivity['excluded_blocks'],[('fixture',1,'kick_left50')])
        self.assertEqual(sensitivity['remaining_trials'],6)
        self.assertEqual(sensitivity['paired_arrival_effects'][0]['paired_cases'],1)

    def test_bad_numeric_inputs_raise_instead_of_becoming_navigation_failures(self):
        for name,value in [('speed_m_s',0),('yaw_speed_deg_s',float('nan')),('observation_budget',1.5)]:
            with self.assertRaises(ValueError): Observations(None,[0,0],0,dict(SENSOR,**{name:value}))
        for value in (float('nan'),-float('inf')):
            sensor=Observations(lambda p,h:[value],[0,0],0,SENSOR)
            with self.assertRaisesRegex(ValueError,'familiarity'): sensor.observe(0)
        sensor=Observations(lambda p,h:[float('inf')],[0,0],0,SENSOR)
        self.assertEqual(sensor.observe(0),0.)  # Existing silent-code sentinel.
        with self.assertRaises(ValueError): Settings(probe_m=0)
        with self.assertRaises(ValueError): Settings(max_turn_deg=95,turn_step_deg=10)

    def test_obstructed_least_risky_forward_choice_is_not_labelled_clear(self):
        reflex=VisualAvoidance(); reflex.frame_heading=0.
        angles=np.deg2rad(np.arange(-180,180,5))
        reflex.points=.04*np.column_stack([np.cos(angles),np.sin(angles)])
        reflex.ages=np.zeros(len(angles))
        _,_,decision=reflex.choose(0.)
        self.assertGreaterEqual(decision['chosen_risk'],.5)
        self.assertFalse(decision['visually_clear'])
        self.assertEqual(decision['state'],'avoid')

    def test_cancelling_commands_do_not_fabricate_a_heading_or_freeze_casts(self):
        policy=controller().policy; policy.state='cast'; policy.travel=20.; policy.cast_moves=2
        policy.pending=dict(reference=0.,value=.5,movement=20.,distance=0.)
        reflex=SimpleNamespace(commands=[dict(heading=0.,distance_m=.05),dict(heading=180.,distance_m=.05)])
        nav=FeedbackNavigator(policy,reflex)
        sensor=Observations(lambda p,h:[-.5],[0,0],0.,dict(SENSOR,time_budget_s=100))
        self.assertTrue(nav.step(sensor,.1,1))
        self.assertEqual(policy.cast_moves,3)
        feedback=policy.decisions[-1]['motor_feedback']
        self.assertFalse(feedback['direction_defined'])
        self.assertIsNone(feedback['net_heading'])
        self.assertIsNone(policy.decisions[-1]['comparison']['movement_heading'])
        json.dumps(policy.decisions,allow_nan=False)


if __name__=='__main__': unittest.main()
