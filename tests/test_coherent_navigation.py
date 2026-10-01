import unittest
from unittest.mock import patch
import numpy as np

from apiaviz.research.coherent_navigation import (
    CoherentAvoidance, CoherentFeedback, DirectionConfirmedController)
from apiaviz.research.visual_avoidance import Settings
from apiaviz.research.familiarity_controller import FamiliarityController, SensorView


class CoherentNavigationTests(unittest.TestCase):
    def test_stall_exit_cannot_immediately_return_to_rejected_bearing(self):
        image = np.random.default_rng(81).random((51,199,3)).astype('float32')
        for phase in (-1, 1):
            p = CoherentAvoidance(phase=phase)
            for _ in range(2):
                p.moved(image, image, 0., .02)
            headings = []
            previous = 0.
            with patch('apiaviz.research.visual_avoidance.flow_points', return_value=(np.empty((0,2)), {})):
                # Six centimetres used to discard recovery and command 0 again.
                for _ in range(12):
                    h, d, _ = p.choose(0.)
                    self.assertLessEqual(abs((h-previous+180)%360-180), 30.+1e-8)
                    p.moved(image, np.roll(image,1,axis=1), h, d)
                    previous = h
                    headings.append(h)
                self.assertIsNone(p.recovery_heading)
                self.assertTrue(p.stalled_bearings)
                self.assertIsNotNone(p.entry_heading)
                h, _, record = p.choose(0.)
                self.assertGreaterEqual(abs(h), 60.)
                self.assertEqual(record['state'], 'avoid')
                # Continued lateral progress permits a gradual route handoff.
                for _ in range(60):
                    h, d, _ = p.choose(0.)
                    self.assertLessEqual(abs((h-previous+180)%360-180), 30.+1e-8)
                    p.moved(image, np.roll(image,1,axis=1), h, d)
                    previous = h
                self.assertFalse(p.stalled_bearings)
                self.assertIsNone(p.entry_heading)
                self.assertAlmostEqual(h, 0.)

    def test_decline_checks_direction_before_casting_and_accounts_scan(self):
        def execute(cls):
            p = cls(dict(scale=1.))
            p.state='follow'; p.anchor=p.travel=p.reference=0.; p.trend=-.1; p.poor=1
            p.pending=dict(reference=0.,value=.95,movement=0.,distance=0.)
            calls=[]; scans=[]; rotations=[]
            def observe(h):
                calls.append(h)
                return .8-.01*abs(h),h,len(calls)*.05
            def rotate(h): rotations.append(h); return True,h,len(calls)*.05
            sensor=SensorView(0.,0.,.1,observe,rotate,lambda:scans.append(1),lambda r:None)
            self.assertTrue(p.step(sensor,2))
            return p,calls,scans
        old,_,_=execute(FamiliarityController)
        new,calls,scans=execute(DirectionConfirmedController)
        self.assertEqual(old.state,'cast');self.assertEqual(new.state,'follow')
        self.assertEqual(new.travel,0.);self.assertEqual(len(scans),1)
        self.assertGreater(len(calls),1)
        self.assertEqual(new.pending['value'],.8)  # New gaze is reacquired.

    def test_unsupported_direction_still_casts(self):
        p=DirectionConfirmedController(dict(scale=1.))
        p._start_cast(4,'sustained_decline');self.assertEqual(p.state,'reorient')
        p.anchor=0.;p._start_cast(4,'no_directional_support')
        self.assertEqual(p.state,'cast')

    def test_passing_side_uses_entry_axis_when_route_request_reverses(self):
        for side in [-1,1]:
            p=CoherentAvoidance(phase=side)
            p.frame_heading=0.;p.points=np.array([[.07,0.],[.1,0.]])
            p.ages=np.zeros(2)
            h,_,_=p.choose(0.)
            self.assertEqual(p.passing_side,side)
            for request in [60.,-60.,120.,-120.]:
                # Same sensed obstacle: changing route requests cannot redefine side.
                h,_,r=p.choose(request)
                self.assertEqual(r['side'],side)
                self.assertEqual(r['entry_heading'],0.)
                self.assertLessEqual(abs(r['route_goal']),45.)
                if r['frontal_risk'] >= .5: self.assertGreaterEqual(h*side,0.)

    def test_stationary_commands_do_not_release_commitment_or_erase_points(self):
        image=np.random.default_rng(3).random((51,199,3)).astype('float32')
        p=CoherentAvoidance();p.entry_heading=0.;p.frame_heading=0.
        p.points=np.array([[.1,0.]]);p.ages=np.array([.1]);p.last_route_clear=True
        for _ in range(3):p.moved(image,image,0.,.02)
        self.assertIsNotNone(p.entry_heading)
        self.assertEqual(p.clear_distance,0.)
        np.testing.assert_allclose(p.points,[[.1,0.]])

    def test_clear_changing_images_release_without_suspending_navigator(self):
        p=CoherentFeedback(Settings(),1);p.entry_heading=0.;p.frame_heading=0.
        image=np.random.default_rng(3).random((51,199,3)).astype('float32')
        with patch('apiaviz.research.visual_avoidance.flow_points',return_value=(np.empty((0,2)),{})):
            for _ in range(3):
                h,d,r=p.choose(0.);p.moved(image,np.roll(image,1,axis=1),h,d)
        self.assertIsNone(p.entry_heading)
        self.assertEqual(len(p.commands),3)
        self.assertFalse(p.active);self.assertIsNone(p.resume)


if __name__=='__main__':unittest.main()
