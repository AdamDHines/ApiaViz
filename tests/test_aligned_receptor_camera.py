import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from apiaviz.research.aligned_receptor_camera import AlignedReceptorCamera
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_input import sample_receptors


class AlignedReceptorCameraTests(unittest.TestCase):
    def test_both_paths_sample_linear_radiance_before_common_response(self):
        raw=np.ones((144,480,6),dtype='float32')*.1
        raw[30:100,230:250]=1000.
        config=dict(elevation_deg=[90.,-20.],retina_shape=[51,199],visible_white=1.)
        with tempfile.TemporaryDirectory() as directory:
            camera=AlignedReceptorCamera(directory,config)
            with patch.object(camera,'frame',return_value=raw):
                uv=camera.scan([0,0],[0],uv=True);rgb=camera.scan([0,0],[0])
                torch.testing.assert_close(rgb,uv/(uv+1),rtol=0,atol=0)
            legacy=DualCamera(directory,config)
            with patch.object(legacy,'frame',return_value=raw):old=legacy.scan([0,0],[0])
            self.assertGreater(float((old-rgb).abs().max()),.1)

    def test_visible_input_is_independent_of_all_uv_receptor_values(self):
        raw=np.random.default_rng(5).uniform(0,10,(144,480,6)).astype('float32')
        config=dict(elevation_deg=[90.,-20.],retina_shape=[51,199],visible_white=1.)
        with tempfile.TemporaryDirectory() as directory:
            camera=AlignedReceptorCamera(directory,config)
            with patch.object(camera,'frame',return_value=raw):
                before=camera.scan([0,0],[35.])
                raw[:,:,:3]=0.
                after=camera.scan([0,0],[35.])
                torch.testing.assert_close(before,after,rtol=0,atol=0)
                self.assertEqual(int(torch.count_nonzero(camera.scan([0,0],[35.],uv=True))),0)


if __name__=='__main__':unittest.main()
