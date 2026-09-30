import unittest
import json
import numpy as np
import torch
from apiaviz.research.mechanisms import MechanismEncoder, mechanism_maps, polyline_distance, probe_grid, probe_summary
from apiaviz.research.spike_overlap import SpikeOverlapMemory
from apiaviz.research.mechanism_control import MatchedCountEncoder
from apiaviz.research.paper_baselines import FrontendEncoder, METHODS
from apiaviz.research.fast_render import SparseWorldRenderer
from apiaviz.nav.torch_route import TorchWorldRenderer

class MechanismControls(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.images=torch.rand(3,3,18,74,generator=torch.Generator().manual_seed(19))

    def test_baselines_are_unchanged(self):
        for method in METHODS:
            a=FrontendEncoder(method,code_dim=128,seed=19)
            b=MechanismEncoder(method,code_dim=128,seed=19)
            torch.testing.assert_close(a(self.images),b(self.images),atol=0,rtol=0)

    def test_swaps_change_only_the_named_stream(self):
        model=MechanismEncoder(code_dim=128)
        get=lambda m:mechanism_maps(self.images,m,model.features.backbone)
        apia,sobel,simple=get("apiaviz"),get("sobel_colour"),get("opponent")
        torch.testing.assert_close(get("apia_simple_colour")[0],apia[0],atol=0,rtol=0)
        torch.testing.assert_close(get("apia_simple_colour")[1],simple[1],atol=0,rtol=0)
        torch.testing.assert_close(get("sobel_apia_colour")[0],sobel[0],atol=0,rtol=0)
        torch.testing.assert_close(get("sobel_apia_colour")[1],apia[1],atol=0,rtol=0)

    def test_count_diagnostic_has_exact_shared_budget(self):
        for method in ("apiaviz","gray","sobel_colour","no_hex","no_adapt","no_dog"):
            model=MatchedCountEncoder(method,code_dim=128)
            for stream in model(self.images).chunk(2,dim=1):
                torch.testing.assert_close(stream.sum(1),torch.full((3,),3.))

    def test_polyline_error_has_no_sampling_floor(self):
        np.testing.assert_allclose(polyline_distance([[.05,0],[.05,.2],[1.1,0]],[[0,0],[1,0]]),[0,.2,.1],atol=1e-12)

    def test_probe_centres_are_not_teaching_samples(self):
        route=np.column_stack([np.arange(80)*.1,np.zeros(80)])
        pp,hh,records,offsets=probe_grid(route,np.zeros(80))
        self.assertEqual(len(records),40);self.assertEqual(len(pp),520)
        self.assertTrue(np.all(np.min(abs(pp[:,0,None]-route[:,0]),axis=1)>.049))

    def test_probe_results_are_finite_serializable_records(self):
        model=MechanismEncoder(code_dim=128)
        memory=SpikeOverlapMemory(model(self.images))
        images=self.images[:1].repeat(26,1,1,1)
        result=probe_summary(model,memory,images,[dict(lateral_m=0.),dict(lateral_m=.1)],np.arange(60.,-61.,-10.))
        json.dumps(result,allow_nan=False)

    def test_sparse_renderer_preserves_wrap_and_painter_order(self):
        rng=np.random.default_rng(3)
        world={k:rng.uniform(-3,3,(80,3)).astype(np.float32) for k in ("X","Y","Z")}
        world["colp"]=rng.uniform(0,1,(80,3)).astype(np.float32)
        for colour in (False,True):
            for mode in ("normal","binary_objects"):
                opts=dict(device=torch.device("cpu"),resolution=4,hfov=296,color=colour,render_mode=mode)
                a,b=TorchWorldRenderer(world,**opts),SparseWorldRenderer(world,**opts)
                for heading in (-180,-90,0,90,180,359):
                    torch.testing.assert_close(a.render_single(.2,.3,.01,heading),b.render_single(.2,.3,.01,heading),atol=0,rtol=0)

if __name__=="__main__":unittest.main()
