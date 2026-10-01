import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from apiaviz.research.retinal_camera import integrate_receptors,IntegratedReceptorCamera


class RetinalIntegrationTests(unittest.TestCase):
    def test_flat_fields_wrap_and_nonnegative_hdr(self):
        raw=np.ones((144,480,3),dtype=np.float32)*1234
        x=integrate_receptors(raw,[-721,0,359,720])
        torch.testing.assert_close(x,torch.full_like(x,1234),rtol=0,atol=0)
        raw[:]=0;raw[65,0]=10000
        x=integrate_receptors(raw,[180,-180,540])
        self.assertGreater(float(x.max()),0)
        self.assertGreaterEqual(float(x.min()),0)
        torch.testing.assert_close(x[0],x[1],rtol=0,atol=0)
        torch.testing.assert_close(x[0],x[2],rtol=0,atol=0)

    def test_box_integration_preserves_small_bright_source_energy_during_yaw(self):
        raw=np.zeros((144,480,3),dtype=np.float32);raw[67,230]=1000
        x=integrate_receptors(raw,np.linspace(-20,20,81))
        # Sun remains well inside the retinal field. Nonoverlapping azimuth cells
        # partition its energy even when it straddles receptor boundaries.
        energy=x.sum((1,2,3))
        self.assertGreater(float(energy.min()),0)
        self.assertLess(float((energy.max()-energy.min())/energy.mean()),1e-6)

    def test_equal_bands_equal_response_and_no_uv_leakage(self):
        rng=np.random.default_rng(7);band=rng.uniform(0,5,(144,480,3)).astype(np.float32)
        raw=np.concatenate([band,band],axis=2)
        config=dict(elevation_deg=[90,-20],retina_shape=[51,199],visible_white=1)
        with tempfile.TemporaryDirectory() as directory:
            c=IntegratedReceptorCamera(directory,config)
            with patch.object(c,'frame',return_value=raw):
                uv=c.scan([0,0],[37],uv=True);visible=c.scan([0,0],[37])
                torch.testing.assert_close(visible,uv/(uv+1),rtol=0,atol=0)
                raw[:,:,:3]=99999
                torch.testing.assert_close(c.scan([0,0],[37]),visible,rtol=0,atol=0)

    def test_constant_sky_vertical_average_uses_solid_angle(self):
        raw=np.zeros((144,480,3),dtype=np.float32)
        raw[:70]=1
        x=integrate_receptors(raw,[0])[0,0,:,0].numpy()
        # Independent high-resolution quadrature over each receptor cell.
        for i,centre in enumerate(np.linspace(60,-15,51)):
            el=np.linspace(centre-.75,centre+.75,10001)
            expected=np.trapz((el>=90-70*110/144)*np.cos(np.deg2rad(el)),el)/np.trapz(np.cos(np.deg2rad(el)),el)
            self.assertAlmostEqual(float(x[i]),expected,places=4)


if __name__=='__main__':unittest.main()
