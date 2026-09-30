import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from apiaviz.research.active_navigation import evaluate as old_evaluate
from apiaviz.research.familiarity_controller import FamiliarityController, ControllerAdapter
from apiaviz.research.route_avoidance import DetourReflex, RouteNavigator, evaluate
from apiaviz.research.motor_feedback import FeedbackNavigator


class RouteAvoidanceTests(unittest.TestCase):
    def test_failed_observation_does_not_overwrite_previous_feedback_record(self):
        from copy import deepcopy
        from apiaviz.research.active_navigation import Observations
        p=FamiliarityController(dict(scale=.3))
        p.pending=dict(reference=0.,value=.5,movement=20.,distance=0.)
        p.decisions=[dict(step=1,locomotion_step=1,motor_feedback={'previous':True})]
        previous=deepcopy(p.decisions)
        nav=FeedbackNavigator(p,SimpleNamespace(commands=[dict(heading=40.,distance_m=.1)]))
        settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                      time_budget_s=360.,observation_budget=0)
        sensor=Observations(None,[0,0],40.,settings)
        self.assertFalse(nav.step(sensor,.1,2))
        self.assertEqual(p.decisions,previous)

    def test_continuous_evasion_cannot_freeze_a_cast_counter(self):
        from apiaviz.research.active_navigation import Observations
        p=FamiliarityController(dict(scale=.3))
        p.state='cast'; p.anchor=p.reference=0.; p.travel=20.
        p.pending=dict(reference=0.,value=.5,movement=20.,distance=0.)
        reflex=SimpleNamespace(commands=[])
        nav=FeedbackNavigator(p,reflex)
        settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                      time_budget_s=360.,observation_budget=2600)
        sensor=Observations(lambda pos,hs:np.full(len(hs),-.5),[0,0],40.,settings)
        for step in range(1,30):
            reflex.commands=[dict(heading=40.,distance_m=.1)]
            if not nav.step(sensor,step*.1,step): break
        self.assertEqual(sensor.termination,'uninformative_views')
        self.assertLess(step,30)
        self.assertTrue(any(t['reason']=='cast_budget' for t in p.transitions))

    def test_actual_motor_path_retains_familiarity_pair_and_advances_cast(self):
        from apiaviz.research.active_navigation import Observations
        p=FamiliarityController(dict(scale=.3))
        p.state='cast'; p.anchor=p.reference=0.; p.travel=20.; p.cast_moves=3
        p.pending=dict(reference=0.,value=.5,movement=20.,distance=0.)
        reflex=SimpleNamespace(commands=[dict(heading=60.,distance_m=.04),dict(heading=40.,distance_m=.06)])
        nav=FeedbackNavigator(p,reflex)
        settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                      time_budget_s=360.,observation_budget=2600)
        sensor=Observations(lambda pos,hs:np.full(len(hs),-.52),[0,0],40.,settings)
        self.assertTrue(nav.step(sensor,.1,2))
        self.assertEqual(p.cast_moves,4)
        decision=p.decisions[-1]
        self.assertAlmostEqual(decision['comparison']['normalized_change'],.02/.3)
        self.assertTrue(decision['motor_feedback']['diverted'])
        self.assertAlmostEqual(decision['motor_feedback']['walked_m'],.1)
        self.assertGreater(decision['comparison']['movement_heading'],40.)
        self.assertLess(decision['comparison']['movement_heading'],60.)

    def test_navigator_receives_images_scores_and_own_motion_not_geometry(self):
        seen=[]
        test=self
        class Policy:
            decisions=[]
            def step(self,view,step):
                for attr in ('position','route','scorer','obstacles','endpoint','events'):
                    test.assertFalse(hasattr(view,attr))
                seen.append((view.distance,view.heading,step))
                return True
        nav=RouteNavigator(Policy())
        sensor=SimpleNamespace(position=[123,456],heading=37.,time=8.,scorer='private')
        nav.step(sensor,1.73,20)
        self.assertEqual(seen,[(1.73,37.,1)])

    def test_detour_keeps_intended_direction_and_requires_sustained_clearance(self):
        reflex=DetourReflex()
        reflex.frame_heading=0
        reflex.points=np.array([[.08,0.]])
        reflex.ages=np.zeros(1)
        reflex.choose(0)
        self.assertTrue(reflex.active)
        _,_,d=reflex.choose(90)
        self.assertEqual(d['held_heading'],0)
        reflex.points=np.empty((0,2)); reflex.ages=np.empty(0)
        image=np.zeros((51,199,3))
        for i in range(3):
            heading,_,_=reflex.choose(90)
            self.assertEqual(heading,0)
            reflex.moved(image,image,heading,.02)
            self.assertEqual(reflex.active,i<2)
        self.assertEqual(reflex.resume,'visual_clearance')
        self.assertEqual((reflex.entries,reflex.exits),(1,1))

    def test_detour_has_finite_motor_budget_even_with_persistent_visual_obstacle(self):
        reflex=DetourReflex(clear_m=.06,max_detour_m=.1)
        image=np.zeros((51,199,3))
        for _ in range(5):
            reflex.frame_heading=0
            reflex.points=np.array([[.08,0.]])
            reflex.ages=np.zeros(1)
            heading,_,_=reflex.choose(0)
            reflex.moved(image,image,heading,.02)
        self.assertFalse(reflex.active)
        self.assertEqual(reflex.resume,'detour_budget')

    def test_handoff_exits_interrupted_cast_without_losing_visual_anchor(self):
        policy=FamiliarityController(dict(scale=.3),phase=-1)
        policy.anchor=25.; policy.travel=65.; policy.reference=25.
        policy.state='cast'; policy.cast_moves=7; policy.pending=dict(value=.8)
        nav=RouteNavigator(policy); nav.steps=11
        nav.reacquire('visual_clearance',1.7)
        self.assertEqual(policy.anchor,25.)
        self.assertEqual(policy.travel,25.)
        self.assertEqual(policy.phase,-1)
        self.assertEqual(policy.state,'follow')
        self.assertIsNone(policy.pending)
        self.assertGreaterEqual(nav.steps-policy.last_check,policy.settings.check_every)
        self.assertEqual(policy.last_explore,11)

    def test_sampling_control_retains_navigation_and_accounts_extra_views(self):
        route=np.column_stack([np.arange(0,2.01,.1),np.zeros(21)])
        sensor=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                    time_budget_s=360.,observation_budget=2600)
        stage=dict(max_steps=60,kick_before_step=16,recovery_radius_m=.1,recovery_consecutive_steps=3)
        scenario=dict(lateral=0.,heading=0.,kick=0.)
        class Scorer:
            world=SimpleNamespace(scan=lambda pos, hs:torch.ones((len(hs),3,51,199))*.5)
            def __call__(self,p,hs):
                return -(.6*np.exp(-p[1]**2/1.2)+.3*np.cos(np.radians(hs)))
        factory=lambda:ControllerAdapter(FamiliarityController(dict(scale=.3)))
        original=old_evaluate(route,np.zeros(21),Scorer(),factory(),scenario,sensor,stage,[-5,5,-5,5])
        result=evaluate(route,np.zeros(21),Scorer(),factory,scenario,sensor,stage,[-5,5,-5,5],mode='sampling_control')
        self.assertTrue(result['reached_nest'])
        self.assertEqual(original['termination'],result['termination'])
        np.testing.assert_allclose([r['position'] for r in result['trace']],
                                   [r['position'] for r in original['trace']],atol=1e-12)
        self.assertEqual(result['observations'],original['observations']+result['avoidance_observations'])
        self.assertAlmostEqual(result['time_s'],original['time_s']+.05*result['avoidance_observations'])
        json.dumps(result,allow_nan=False)


if __name__ == '__main__': unittest.main()
