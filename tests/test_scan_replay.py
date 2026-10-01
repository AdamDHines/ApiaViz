import unittest
import numpy as np
from scripts.audit_scanning_regression import yaw_at
from scripts.replay_scan_timing import position_at


class ScanReplayTests(unittest.TestCase):
    def test_yaw_crosses_seam_without_reversing(self):
        events=[dict(kind='turn',from_heading=350.,heading=370.,angle_deg=20.,time_s=2.,duration_s=1.)]
        np.testing.assert_allclose(yaw_at(events,[0,1,1.5,2,3],350),[350,350,360,370,370])

    def test_logged_negative_half_turn_keeps_its_direction(self):
        events=[dict(kind='turn',from_heading=10.,heading=-170.,angle_deg=-180.,time_s=1.,duration_s=1.),
                dict(kind='turn',from_heading=-170.,heading=-170.,angle_deg=0.,time_s=2.,duration_s=0.)]
        np.testing.assert_allclose(yaw_at(events,[0,.5,1,2,3],10),[10,-80,-170,-170,-170])

    def test_position_holds_during_scanning_and_blocked_commands(self):
        moves=[dict(position=[.1,0],time_s=6.05,commanded_stride_m=.1),
               dict(position=[.1,0],time_s=8.05,commanded_stride_m=.1)]
        for t,expected in [(0,0),(5,0),(5.5,.05),(6,.1),(6.8,.1),(7.5,.1),(9,.1)]:
            np.testing.assert_allclose(position_at(moves,[0,0],t,.1,.05),[expected,0],atol=1e-12)


if __name__=='__main__':unittest.main()
