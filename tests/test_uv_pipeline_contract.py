import unittest
import torch
from apiaviz.src.modules import LocalLuminanceAdapter
from apiaviz.research.uv_encoder import ApiaVizUVEncoder, UVEncoderConfig
from apiaviz.research.uv_precision import UVPrecisionEncoder, StableLuminanceAdapter
from apiaviz.research.frontend_refinements import refinement_maps


class UVPipelineContractTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_uv_visible_maps_exactly_match_preferred_linear_colour_frontend(self):
        model=ApiaVizUVEncoder(UVEncoderConfig(schema='apiaviz-uv-v2',response_half=1.))
        x=torch.rand(2,3,51,199,generator=torch.Generator().manual_seed(19))
        gb=(x/(x+1))[:,[2,1]]
        form,colour=refinement_maps(gb,'linear_colour',model.backbone.visible)
        maps=model.backbone(x)
        torch.testing.assert_close(form,maps['form'],rtol=0,atol=0)
        torch.testing.assert_close(colour,maps['colour'],rtol=0,atol=0)
        altered=x.clone();altered[:,0]=torch.roll(altered[:,0],15,-1)
        changed=model.backbone(altered)
        for name in ['form','colour']:torch.testing.assert_close(maps[name],changed[name],rtol=0,atol=0)
        self.assertGreater(float((maps['uv']-changed['uv']).abs().max()),.01)
        torch.testing.assert_close(model(x)[:,:8000],model(altered)[:,:8000],rtol=0,atol=0)

    def test_constant_neutral_fields_have_no_fictitious_spikes(self):
        model=UVPrecisionEncoder()
        for value in [0.,.01,.1,.5,1.,10.,1000.]:
            x=torch.full((1,3,51,199),value)
            for name,m in model.backbone(x).items():self.assertEqual(int(torch.count_nonzero(m)),0,name)
            self.assertEqual(int(torch.count_nonzero(model(x))),0)

    def test_precision_repair_matches_double_precision_reference(self):
        x=(torch.rand(2,1,51,199,generator=torch.Generator().manual_seed(3))*.01-.9)
        old=LocalLuminanceAdapter();new=StableLuminanceAdapter()
        reference=old(x.double()).float()
        self.assertLess(float((new(x)-reference).abs().max()),float((old(x)-reference).abs().max()))
        torch.testing.assert_close(new(x),reference,atol=2e-6,rtol=1e-4)

    def test_uv_only_landmark_survives_into_spikes_and_visible_stays_silent(self):
        model=UVPrecisionEncoder()
        image=torch.full((1,3,51,199),.1)
        image[:,0,15:35,70:100]=.3
        codes=model(image)
        self.assertEqual(int(torch.count_nonzero(codes[:,:8000])),0)
        self.assertGreater(int(torch.count_nonzero(codes[:,8000:])),0)
        self.assertEqual(int(torch.count_nonzero(model(image,uv_enabled=False))),0)


if __name__=='__main__':unittest.main()
