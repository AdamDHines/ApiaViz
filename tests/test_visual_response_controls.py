import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('response_controls', Path(__file__).resolve().parents[1]/'scripts/visual_response_controls.py')
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


class ResponseControlsTests(unittest.TestCase):
    def test_selection_is_balanced_and_independent_of_results(self):
        w = dict(route=[[float(i)/10, 0.] for i in range(31)], headings=[0.]*31)
        poses = controls.select_poses(w)
        self.assertEqual(len(poses), 28)
        self.assertEqual(len({p['station'] for p in poses}), 7)
        for i in range(0, len(poses), 4):
            group = poses[i:i+4]
            self.assertEqual([p['kind'] for p in group], ['taught', 'midpoint', 'left20', 'right20'])
            np.testing.assert_allclose(np.mean([group[2]['position'], group[3]['position']], axis=0), group[0]['position'])

    def test_interleaved_grids_expose_missed_peak(self):
        angles = np.unique(np.r_[np.arange(-180., 180., 5.), np.arange(25., 66.)])
        matrix = torch.zeros(len(angles), 31)
        matrix[:, 0] = .2
        matrix[np.where(angles == 45.)[0][0], 15] = 1.
        matrix[np.where(angles == 0.)[0][0], 1] = .5
        pose = dict(position=[1.5, 0.], station=15, tangent=45.)
        r = controls.metrics(matrix, angles, pose, np.array([[0., 0.], [3., 0.]]), 0.)
        self.assertEqual(r['grid10']['heading'], 0.)
        self.assertEqual(r['grid5']['heading'], 45.)
        self.assertEqual(r['grid10_shift5']['heading'], 45.)
        self.assertAlmostEqual(r['place_margin'], .8)
        self.assertEqual(r['five_degree_loss'], 1.)

    def test_recovery_is_geometric_not_tangent_agreement(self):
        angles = np.unique(np.r_[np.arange(-180., 180., 5.), np.arange(-20., 21.)])
        matrix = torch.zeros(len(angles), 31)
        matrix[np.where(angles == 0.)[0][0], 15] = 1.
        pose = dict(position=[1.5, .2], station=15, tangent=0.)
        r = controls.metrics(matrix, angles, pose, np.array([[0., 0.], [3., 0.]]), 0.)
        self.assertEqual(r['grid5']['error'], 0.)
        self.assertFalse(r['grid5']['inward'])


if __name__ == '__main__':
    unittest.main()
