import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apiaviz.research.active_navigation import Observations, segment_collision
from apiaviz.research.avoidance_navigation import VisualLocomotion
from apiaviz.research.collision_geometry import RockGeometry, SCHEMA
from apiaviz.research.motor_feedback import evaluate
from apiaviz.research.familiarity_controller import FamiliarityController, ControllerAdapter


def rectangle(x0, x1, y0, y1):
    return np.array([[[x0,y0],[x1,y0],[x1,y1]], [[x0,y0],[x1,y1],[x0,y1]]])


class CollisionTests(unittest.TestCase):
    def test_disc_false_positive_is_clear_of_actual_projection(self):
        old = [dict(x=0, y=0, conservative_radius_m=.413)]
        mesh = RockGeometry([('rock', rectangle(-.2,.2,-.15,.15))], .005)
        a, b = [.367,0], [.347,0]
        self.assertTrue(segment_collision(a,b,old))
        self.assertFalse(segment_collision(a,b,mesh))

    def test_swept_body_edges_tangency_inside_and_zero_motion(self):
        mesh = RockGeometry([('rock', rectangle(0,1,0,1))], .01)
        for a,b in [([-1,.5],[2,.5]), ([-1,-.01],[2,-.01]), ([.5,.5],[.5,.5]),
                    ([1.01,.5],[1.01,.5]), ([-.005,-.005],[-.005,-.005])]:
            self.assertTrue(mesh.intersects(a,b), (a,b))
        self.assertFalse(mesh.intersects([-1,-.011],[2,-.011]))
        self.assertFalse(mesh.intersects([-1,-1],[-.01,-.01]))

    def test_concavity_and_thin_triangle_are_preserved(self):
        triangles = np.concatenate([rectangle(0,1,0,.1), rectangle(0,.1,0,1)])
        mesh = RockGeometry([('L',triangles)], .01)
        self.assertFalse(mesh.intersects([.3,.3],[.7,.7]))
        edge = RockGeometry([('edge', [[[0,0],[0,1],[0,1]]])], 0)
        self.assertTrue(edge.intersects([-1,.5],[1,.5]))
        self.assertFalse(edge.intersects([.1,.5],[.1,.7]))

    def test_scene_hash_and_asset_hash_are_required(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'collision.json'
            path.write_text(json.dumps(dict(schema=SCHEMA,scene_sha256='a',rocks=[
                dict(name='r', triangles_xy_m=rectangle(0,1,0,1).tolist())])))
            with self.assertRaises(ValueError): RockGeometry.load(path,'b',.005)
            with self.assertRaises(ValueError): RockGeometry.load(path,'a',.005,expected_sha256='wrong')

    def test_blocked_contact_is_nonterminal_charged_and_not_a_policy_signal(self):
        settings = dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                        time_budget_s=100.,observation_budget=500)
        image = np.zeros((51,199,3))
        mesh = RockGeometry([('rock',rectangle(.012,.1,-.1,.1))], .005)
        motions=[]
        for geometry in ([],mesh):
            sensor=Observations(None,[0,0],0,settings)
            motion=VisualLocomotion(sensor,lambda p,h:image,[-1,1,-1,1],geometry)
            self.assertTrue(motion.advance(0)[0])
            motions.append(motion)
        clear,blocked=motions
        self.assertIsNone(blocked.sensor.termination)
        self.assertGreater(blocked.blocked_proposals,0)
        self.assertAlmostEqual(blocked.commanded_path,.1)
        self.assertLess(blocked.path,.012)
        self.assertEqual(clear.sensor.count,blocked.sensor.count)
        self.assertAlmostEqual(clear.sensor.time,blocked.sensor.time)
        self.assertEqual(clear.policy.decisions,blocked.policy.decisions)
        np.testing.assert_allclose(clear.policy.points,blocked.policy.points)
        event=next(e for e in blocked.sensor.events if e['kind']=='blocked_proposal')
        self.assertIn('proposed_position',event)
        self.assertIn('decision',event)
        self.assertIn('visual_points',event)
        json.dumps(blocked.sensor.events,allow_nan=False)
        from apiaviz.research.route_full_audit import check_motion
        row=dict(path_length_m=blocked.path,commanded_path_m=blocked.commanded_path,
                 blocked_proposals=blocked.blocked_proposals,disturbance_applied=False,
                 final_nest_distance_m=float(np.linalg.norm(blocked.sensor.position-[1,0])))
        self.assertEqual(check_motion([0,0],dict(events=blocked.sensor.events,microtrace=blocked.trace),
            row,dict(evaluation=dict(kick_before_step=5),avoidance_settings=dict(stride_m=.02)),
            dict(world_bounds_m=[-1,1,-1,1],route=[[0,0],[1,0]],headings=[0,0]),mesh,dict(kick=0.)),(blocked.path,0))

    def test_blocked_attempt_still_reserves_camera_and_observation_budget(self):
        settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
                      time_budget_s=100.,observation_budget=2)
        sensor=Observations(None,[0,0],0,settings)
        mesh=RockGeometry([('wall',rectangle(.004,.1,-1,1))],.001)
        motion=VisualLocomotion(sensor,lambda p,h:np.zeros((51,199,3)),[-2,2,-2,2],mesh)
        self.assertFalse(motion.advance(0)[0])
        self.assertEqual(sensor.termination,'observation_budget')
        self.assertEqual(sensor.count,2)
        self.assertAlmostEqual(sensor.time,.15)
        self.assertEqual(motion.path,0.)
        self.assertEqual(motion.blocked_proposals,1)
        self.assertAlmostEqual(motion.commanded_path,.005)

    def test_external_displacement_into_mesh_remains_distinct(self):
        from types import SimpleNamespace
        class Scorer:
            world=SimpleNamespace(scan=lambda pos,hs:torch.full((len(hs),3,51,199),.5))
            def __call__(self,pos,hs): return np.full(len(hs),-.5)
        mesh=RockGeometry([('wall',rectangle(-.1,.1,.45,.55))],.005)
        result=evaluate([[0,0],[2,0]],[0,0],Scorer(),
            lambda:ControllerAdapter(FamiliarityController(dict(scale=.3))),
            dict(lateral=0.,heading=0.,kick=.5),
            dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,time_budget_s=2.,observation_budget=100),
            dict(max_steps=10,kick_before_step=1,recovery_radius_m=.1,recovery_consecutive_steps=3),[-3,3,-3,3],mesh)
        self.assertEqual(result['termination'],'displacement_into_rock')
        self.assertEqual(result['path_length_m'],0.)
        self.assertEqual(result['blocked_proposals'],0)

    def test_continuous_navigator_keeps_memory_and_budgets_when_all_moves_block(self):
        from types import SimpleNamespace
        class Scorer:
            world=SimpleNamespace(scan=lambda pos,hs:torch.full((len(hs),3,51,199),.5))
            def __call__(self,pos,hs): return np.full(len(hs),-.5)
        policy=FamiliarityController(dict(scale=.3))
        policy.state='follow'
        policy.anchor=policy.travel=policy.reference=0.
        mesh=RockGeometry([('wall',rectangle(.005,.1,-1,1))], .001)
        result=evaluate([[0,0],[2,0]],[0,0],Scorer(),lambda:ControllerAdapter(policy),
            dict(lateral=0.,heading=0.,kick=0.),
            dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,time_budget_s=2.,observation_budget=100),
            dict(max_steps=10,kick_before_step=5,recovery_radius_m=.1,recovery_consecutive_steps=3),
            [-3,3,-3,3],mesh)
        self.assertEqual(result['termination'],'time_budget')
        self.assertGreater(result['blocked_proposals'],0)
        self.assertEqual(result['motor_state_resets'],0)
        self.assertLessEqual(result['time_s'],2.+1e-12)
        self.assertGreater(len(policy.decisions),0)
        json.dumps(result,allow_nan=False)


if __name__ == '__main__': unittest.main()
