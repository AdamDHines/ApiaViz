import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.spectral_input import SCHEMA,CHANNELS,file_sha
from apiaviz.research.uv_input import sample_receptors,load_views,AZIMUTH
from apiaviz.research.uv_encoder import ApiaVizUVEncoder,UVEncoderConfig
from apiaviz.research.spike_overlap import SpikeOverlapMemory


class UVEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x=torch.rand(3,3,18,74,generator=torch.Generator().manual_seed(9))
        self.config=UVEncoderConfig(visible_code_dim=128,uv_code_dim=64,seed=19)
        self.model=ApiaVizUVEncoder(self.config)

    def test_visible_streams_are_exact_and_uv_never_enters_them(self):
        old=RefinementEncoder('linear_colour',seed=19,code_dim=128)
        expected=old(self.x[:,[2,1]])
        actual=self.model(self.x)
        torch.testing.assert_close(actual[:,:128],expected,atol=0,rtol=0)
        changed=self.x.clone(); changed[:,0]=changed[:,0].flip(-1)*4
        torch.testing.assert_close(self.model(changed)[:,:128],expected,atol=0,rtol=0)
        self.assertFalse(torch.equal(self.model(changed)[:,128:],actual[:,128:]))

    def test_uv_pathway_ablation_and_dark_uv_are_different(self):
        off=self.model(self.x,uv_enabled=False)
        self.assertEqual(torch.count_nonzero(off[:,128:]),0)
        dark=self.x.clone(); dark[:,0]=0
        self.assertGreater(torch.count_nonzero(self.model(dark)[:,128:]),0)
        torch.testing.assert_close(off[:,:128],self.model(self.x)[:,:128],atol=0,rtol=0)

    def test_linear_opponent_polarities_and_radiance_above_one(self):
        x=torch.ones(1,3,18,74); x[:,0]=3; x[:,1]=2
        uv=self.model.backbone(x)['uv']
        torch.testing.assert_close(uv[:,3],torch.ones_like(uv[:,3]),atol=1e-6,rtol=0)
        torch.testing.assert_close(uv[:,5],torch.full_like(uv[:,5],2),atol=1e-6,rtol=0)
        self.assertEqual(torch.count_nonzero(uv[:,[4,6]]),0)
        self.assertTrue(torch.isfinite(self.model(x)).all())
        for value in (0.,1.,5.):
            self.assertTrue(torch.isfinite(self.model(torch.full_like(x,value))).all())

    def test_frozen_reproducible_checkpoint_and_view_memory(self):
        self.assertFalse(any(p.requires_grad for p in self.model.parameters()))
        clone=ApiaVizUVEncoder(self.config)
        clone.load_state_dict(self.model.state_dict())
        codes=self.model(self.x)
        torch.testing.assert_close(clone(self.x),codes,atol=0,rtol=0)
        torch.testing.assert_close(torch.cat([self.model(x[None]) for x in self.x]),codes,atol=1e-6,rtol=1e-5)
        memory=SpikeOverlapMemory(codes)
        torch.testing.assert_close(memory(codes),torch.full((3,),-1.),atol=1e-6,rtol=0)

    def test_invalid_inputs_and_configuration_fail(self):
        for x in (self.x[:,:2],self.x.to(torch.uint8),-self.x,self.x*float('nan')):
            with self.assertRaises(ValueError): self.model(x)
        for config in (dict(radiance_scale=0),dict(uv_code_dim=0),dict(visible_code_dim=127)):
            with self.assertRaises(ValueError): UVEncoderConfig(**config)

    def test_sampler_preserves_channels_angles_seam_and_linear_units(self):
        h,w=110,360
        az=np.deg2rad(180-(np.arange(w)+.5))
        elevation=90-(np.arange(h)+.5)
        panorama=np.stack(np.broadcast_arrays(2+np.cos(az)[None,:],
            (elevation[:,None]+90)/90,np.ones((h,w))*4),axis=-1)
        views=sample_receptors(panorama,[0,360,180])
        torch.testing.assert_close(views[0],views[1],atol=0,rtol=0)
        self.assertAlmostEqual(float(views[0,0,25,99]),3,places=4)
        self.assertAlmostEqual(float(views[2,0,25,99]),1,places=4)
        self.assertAlmostEqual(float(views[0,1,0,99]),150/90,places=6)
        self.assertEqual(float(views[0,2].min()),4.)
        shifted=sample_receptors(panorama,[25],panorama_heading=25)
        torch.testing.assert_close(shifted[0],views[0],atol=0,rtol=0)
        with self.assertRaises(ValueError): sample_receptors(panorama,[0],elevation=(30,-20))

    def test_verified_adapter_rejects_wrong_calibration_and_display_labels(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); data=np.ones((110,360,3),dtype=np.float32)*2
            np.save(d/'intensity.npy',data)
            record=dict(schema=SCHEMA,channels=CHANNELS,linear=True,shape=list(data.shape),
                arrays=dict(intensity=dict(file='intensity.npy',sha256=file_sha(d/'intensity.npy'))),
                calibration_sha256='pinned',azimuth=AZIMUTH,pose=dict(heading=0.),elevation_deg=[90,-20])
            sidecar=d/'frame.json'; sidecar.write_text(json.dumps(record))
            views,_=load_views(sidecar,[0],calibration_sha256='pinned')
            self.assertEqual(tuple(views.shape),(1,3,51,199)); self.assertEqual(float(views.min()),2.)
            with self.assertRaisesRegex(ValueError,'calibration'):
                load_views(sidecar,[0],calibration_sha256='other')
            record['channels']=['red','green','blue']; sidecar.write_text(json.dumps(record))
            with self.assertRaises(ValueError): load_views(sidecar,[0],calibration_sha256='pinned')


if __name__=='__main__': unittest.main()
