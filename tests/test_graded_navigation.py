import unittest
import numpy as np
import torch
from apiaviz.research.graded_navigation import GradedNavigationEncoder, memory_and_calibration
from apiaviz.research import uv_trials as base
from apiaviz.research.graded_smoke import protocol


class GradedNavigationTests(unittest.TestCase):
    def test_navigation_matches_image_control(self):
        torch.set_num_threads(2)
        model=GradedNavigationEncoder()
        x=torch.rand(2,3,51,199,generator=torch.Generator().manual_seed(25))*4
        expected=model.readouts(model.pooled(x/(x+1),'combined'))['graded']
        torch.testing.assert_close(model(x),expected,rtol=0,atol=0)

    def test_amplitudes_survive_memory_and_calibration(self):
        model=type('Graded',(),{'response_format':'graded'})()
        codes=torch.tensor([[1.,.01],[.01,1.],[1.,1.]])
        memory,calibration=memory_and_calibration(model,codes)
        query=torch.tensor([[.1,1.]])
        expected=-(torch.nn.functional.normalize(query,dim=1) @ torch.nn.functional.normalize(codes,dim=1).T).max(1).values
        torch.testing.assert_close(memory(query),expected)
        self.assertLess(float(memory(query)), -.99)
        self.assertGreater(calibration['scale'],.1)
        binary,old=memory_and_calibration(object(),codes)
        torch.testing.assert_close(binary(query),torch.tensor([-1.]))
        self.assertEqual(old,base.acquisition_calibration(codes.numpy()))
        self.assertEqual(old['scale'],.02)

    def test_trial_scorer_preserves_amplitude_and_labels_cost(self):
        class Camera:
            def scan(self,position,headings,uv=False):
                self.uv=uv
                return torch.tensor([[.1,1.]])
        class Encoder:
            response_format='graded'
            def __call__(self,x):return x
        model=Encoder();codes=torch.tensor([[1.,.01],[.01,1.]])
        memory,_=memory_and_calibration(model,codes);camera=Camera()
        scorer=base.TrialScorer(camera,model,memory,'apiaviz_uv')
        np.testing.assert_allclose(scorer([0,0],[0]),memory(torch.tensor([[.1,1.]])).numpy())
        self.assertTrue(camera.uv);self.assertIsNone(scorer.active_spikes)
        self.assertEqual(scorer.active_response_components,2)

    def test_protocol_has_nine_matched_cases_and_unchanged_budgets(self):
        import json
        source=base.ROOT/'apiaviz/output/navigation-fidelity-v1'
        if not (source/'protocol.json').exists():self.skipTest('Local frozen source absent')
        p=protocol(source);old=json.loads((source/'protocol.json').read_text())
        self.assertEqual(len(list(base.cases(p))),9)
        for name in ('sensor_settings','evaluation','evaluation_safety','avoidance_settings','render'):
            self.assertEqual(p[name],old[name])
        self.assertEqual(p['controller_settings']['scan_extents'],[20.,60.])
        self.assertEqual(p['protocol_revision'],'graded-navigation-smoke-v3-bounded')
        self.assertEqual(p['controller_settings']['scan_step'],5.)
        self.assertFalse(p['encoder']['graded_config']['adaptation'])


if __name__=='__main__':unittest.main()
