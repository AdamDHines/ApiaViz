import cv2
import json
import numpy as np
import unittest

from apiaviz.research.active_navigation import Observations
from apiaviz.research.avoidance_navigation import VisualLocomotion
from apiaviz.research.visual_avoidance import VisualAvoidance, flow_points
from apiaviz.research.avoidance_continuity import ContinuousFactory


def textured_plane():
    rng = np.random.default_rng(72)
    before = cv2.GaussianBlur(rng.uniform(.1, .9, (51, 199, 3)).astype(np.float32), (3, 3), .6)
    angles = np.deg2rad(np.linspace(-148, 148, 199))
    previous = np.arctan2(.14*np.sin(angles), .15*np.cos(angles))
    x = ((np.rad2deg(previous)+148)/296*198).astype(np.float32)
    after = cv2.remap(before, np.tile(x, (51, 1)),
                      np.tile(np.arange(51, dtype=np.float32)[:, None], (1, 199)), cv2.INTER_LINEAR)
    return before, after


def test_flow_recovers_approximate_visual_distance():
    before, after = textured_plane()
    points, stats = flow_points(before, after, .01)
    frontal = points[(points[:, 0] > 0) & (abs(points[:, 1]) < .15)]
    assert stats['supported_pixels'] > 100
    assert len(frontal) > 10
    assert abs(np.median(frontal[:, 0])-.14) < .035


def test_no_motion_does_not_invent_obstacle():
    before, _ = textured_plane()
    points, _ = flow_points(before, before, .01)
    assert len(points) == 0
    with unittest.TestCase().assertRaises(ValueError):
        flow_points(before, before, 0)


def test_passing_side_and_mirror_symmetry():
    before, after = textured_plane()
    # A textured wall on one side of the frontal field.
    before[:, :95] = .5
    after[:, :95] = .5
    a, b = VisualAvoidance(phase=1), VisualAvoidance(phase=-1)
    a.moved(before, after, 0, .01)
    b.moved(before[:, ::-1], after[:, ::-1], 0, .01)
    ha, _, _ = a.choose(0)
    hb, _, _ = b.choose(0)
    assert ha < 0 and hb > 0
    assert abs(ha+hb) <= 10


def sensor_settings():
    return dict(speed_m_s=.1, yaw_speed_deg_s=180., observation_s=.05,
                time_budget_s=100., observation_budget=500)


def test_collision_geometry_cannot_select_a_turn():
    image = np.ones((51, 199, 3), dtype=np.float32)*.5
    a = Observations(None, [0, 0], 0, sensor_settings())
    b = Observations(None, [0, 0], 0, sensor_settings())
    rock = dict(x=.04, y=0., conservative_radius_m=.01)
    clear = VisualLocomotion(a, lambda p,h:image, [-1,1,-1,1], [])
    blocked = VisualLocomotion(b, lambda p,h:image, [-1,1,-1,1], [rock])
    assert clear.advance(0)[0]
    assert not blocked.advance(0)[0]
    assert b.termination == 'rock_collision'
    # Invisible obstacle changes physics only: it produces no evasive turn.
    assert all(r['heading'] == 0 for r in blocked.trace)
    assert abs(a.position[0]-.1) < 1e-10
    assert blocked.path < .04


def test_all_microsteps_and_camera_views_charged():
    image = np.ones((51, 199, 3), dtype=np.float32)*.5
    s = Observations(None, [0,0], 0, sensor_settings())
    m = VisualLocomotion(s, lambda p,h:image, [-1,1,-1,1], [])
    ok, diverted = m.advance(np.float64(0))
    assert ok
    json.dumps(dict(diverted=diverted, trace=m.trace), allow_nan=False)
    assert abs(sum(t['stride_m'] for t in m.trace)-.1) < 1e-10
    assert s.count == len(m.trace)+1 == m.camera_views
    assert abs(s.time-(1.+s.count*.05)) < 1e-10


def test_reserve_postmovement_view_and_respect_time_budget():
    image = np.zeros((51,199,3))
    settings = sensor_settings()
    settings['observation_budget'] = 1
    s = Observations(None,[0,0],0,settings)
    m = VisualLocomotion(s, lambda p,h:image, [-1,1,-1,1], [])
    assert not m.advance(0)[0]
    assert m.path == 0 and s.termination == 'observation_budget'
    settings['observation_budget'] = 100
    settings['time_budget_s'] = .12
    s = Observations(None,[0,0],0,settings)
    m = VisualLocomotion(s, lambda p,h:image, [-1,1,-1,1], [])
    assert not m.advance(0)[0]
    assert s.time <= .12 and m.path == 0


def test_old_visual_points_expire_by_commanded_motion():
    a = VisualAvoidance()
    image = np.zeros((51,199,3))
    a.points = np.array([[.1,.1]])
    a.ages = np.zeros(1)
    a.frame_heading = 0
    a.moved(image,image,90,.02)
    assert np.allclose(a.points, [[.08,-.1]])
    a.moved(image,image,90,.11)
    assert len(a.points) == 0


def test_detour_preserves_direction_and_invalidates_temporal_pair():
    from types import SimpleNamespace
    policy = SimpleNamespace(decisions=[], pending={'value':.5}, trend=.2,
                             poor=2, good=1, travel=35., anchor=20., phase=-1)
    class Controller:
        def __init__(self): self.policy = policy
        def step(self, sensor, step):
            policy.decisions.append(dict(step=step))
            return True
    factory = ContinuousFactory(Controller())
    first = factory()
    first.step(None, 1)
    second = factory()
    assert policy.pending is None and policy.trend == 0
    assert policy.poor == policy.good == 0
    assert (policy.travel, policy.anchor, policy.phase) == (35.,20.,-1)
    second.step(None, 1)
    assert first.policy.decisions == [dict(step=1)]
    assert second.policy.decisions == [dict(step=2)]


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items() if name.startswith("test_"))
