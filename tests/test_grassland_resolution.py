import unittest

import numpy as np
import torch
from torch.nn import functional as F

from apiaviz.research.angular_frontend import AngularEncoder, scaled_kernel, spatial_filter
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.grassland_resolution import SHAPES, conditions
from apiaviz.research.grassland_smoke import sample_panorama
from apiaviz.research.study import fingerprint


class ResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sampling_angular_grid_at_all_resolutions(self):
        az=np.radians(180-(np.arange(720)+.5)*.5)
        el=60.5-(np.arange(152)+.5)*.5
        panorama=np.empty((152,720,3))
        panorama[:,:,0]=(1+np.cos(az))[None]*127.5
        panorama[:,:,1]=(1+np.sin(az))[None]*127.5
        panorama[:,:,2]=(el[:,None]+15.5)/76*255
        headings=np.array([0,90,180,270,359.9])
        for h,w in SHAPES:
            x=sample_panorama(panorama,headings,(h,w)).numpy()
            target=np.radians(headings[:,None]+np.linspace(-148,148,w))
            np.testing.assert_allclose(x[:,0,0],(1+np.cos(target))/2,atol=5e-6)
            np.testing.assert_allclose(x[:,1,0],(1+np.sin(target))/2,atol=5e-6)
            np.testing.assert_allclose(x[0,2,:,0],(np.linspace(60,-15,h)+15.5)/76,atol=1e-7)
            self.assertTrue(torch.equal(sample_panorama(panorama,[0],(h,w)),sample_panorama(panorama,[360],(h,w))))

    def test_scaled_kernel_matches_direct_fractional_tap_sampling(self):
        # Independent grid_sample reference checks reflection and interpolation
        # across the full image, including corners, with an asymmetric kernel.
        generator=torch.Generator().manual_seed(3)
        x=torch.rand((1,1,31,37),generator=generator,dtype=torch.float64)
        original=torch.randn((1,1,3,5),generator=generator,dtype=torch.float64)
        sy,sx=2.37,1.84
        kernel=scaled_kernel(original,sy,sx)
        actual=spatial_filter(x,kernel)
        expected=torch.zeros_like(x)
        yy,xx=torch.meshgrid(torch.arange(31,dtype=x.dtype),torch.arange(37,dtype=x.dtype),indexing='ij')
        for iy in range(3):
            for ix in range(5):
                y=yy+(iy-1)*sy;xcoord=xx+(ix-2)*sx
                grid=torch.stack([xcoord/36*2-1,y/30*2-1],-1)[None]
                expected+=original[0,0,iy,ix]*F.grid_sample(x,grid,padding_mode='reflection',align_corners=True)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(kernel.sum(),original.sum(),atol=1e-12,rtol=1e-12)

    def test_exact_reference_and_unchanged_wiring(self):
        images=torch.rand((3,3,18,74),generator=torch.Generator().manual_seed(5))
        for method in ('linear_colour','sobel_colour'):
            original=RefinementEncoder(method=method,seed=19,code_dim=128)
            angular=AngularEncoder(method=method,seed=19,code_dim=128)
            self.assertEqual(fingerprint(original),fingerprint(angular))
            self.assertTrue(torch.equal(original(images),angular(images)))
            for shape in SHAPES[1:]:
                larger=F.interpolate(images,size=shape,mode='bilinear',align_corners=True)
                pooled=angular.pooled_features(larger)
                self.assertEqual([list(p.shape[1:]) for p in pooled],[[3,8,64],[2,8,64]])
                self.assertTrue(torch.isfinite(angular(larger)).all())
                self.assertEqual(fingerprint(original),fingerprint(angular))

    def test_hex_phase_and_offsets_match_original_at_reference(self):
        encoder=AngularEncoder(code_dim=128)
        images=torch.rand((2,3,18,74),generator=torch.Generator().manual_seed(9))
        x=images[:,-2:]*2-1
        kernels=encoder.kernels(images)
        sampler=encoder.features.backbone.spatial_sampler
        results=[spatial_filter(x,k,sampler.bias,groups=2) for k in kernels['hex']]
        actual=torch.where(kernels['even'],*results)
        torch.testing.assert_close(actual,sampler(x),atol=2e-7,rtol=1e-6)

    def test_prespecified_trial_count(self):
        self.assertEqual(sum(3*len(c['methods']) for c in conditions()),39)


if __name__=='__main__': unittest.main()
