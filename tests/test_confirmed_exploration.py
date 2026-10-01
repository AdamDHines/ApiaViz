import unittest
from apiaviz.research.confirmed_exploration import ConfirmedExplorationController
from apiaviz.research.familiarity_controller import SensorView


class ConfirmedExplorationTests(unittest.TestCase):
    def test_scheduled_check_keeps_supported_course_and_advances_schedule(self):
        p=ConfirmedExplorationController(dict(scale=1.));p.state='follow'
        p.anchor=p.travel=p.reference=0.;p.last_check=11
        calls=[]
        def observe(h):calls.append(h);return .9-.01*abs(h),h,len(calls)*.05
        sensor=SensorView(0.,0.,1.1,observe,lambda h:(True,h,len(calls)*.05),lambda:None,lambda r:None)
        self.assertTrue(p.step(sensor,12));self.assertEqual(p.state,'follow')
        self.assertEqual(p.travel,0.);self.assertEqual(p.last_explore,12)
        self.assertEqual(p.cast_attempts,0);self.assertGreater(len(calls),1)
        self.assertEqual(p.pending['value'],.9)

    def test_failed_direction_check_retains_bounded_cast_fallback(self):
        p=ConfirmedExplorationController(dict(scale=1.));p.anchor=p.reference=0.
        p._start_cast(12,'periodic_exploration');self.assertEqual(p.state,'reorient')
        p._start_cast(12,'no_directional_support');self.assertEqual(p.state,'cast')
        self.assertEqual(p.cast_attempts,1);self.assertEqual(p.last_explore,12)


if __name__=='__main__':unittest.main()
