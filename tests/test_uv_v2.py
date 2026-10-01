import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout
import io
from dataclasses import replace

import numpy as np
import torch

from apiaviz.research.dual_camera import canonical_position,position_key,render_seed
from apiaviz.research.uv_encoder import ApiaVizUVEncoder,UVEncoderConfig
from apiaviz.research.uv_trials import make_model
from apiaviz.research.uv_trials import configuration,teaching_views,atomic_json
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.uv_validation import protocol,assess


class UVV2Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_roundoff_reuses_pose_without_snapping_physics_to_route(self):
        for x in [.2,.3,-.2,3.1415926535]:
            self.assertEqual(position_key([x,0.],8),position_key([x+1e-16,-1e-16],8))
            np.testing.assert_allclose(canonical_position([x,0.],8),[x,0.],atol=5.1e-9,rtol=0)
        self.assertNotEqual(position_key([.2,0.],8),position_key([.20001,0.],8))
        # Historical exact-key semantics stay available.
        self.assertNotEqual(position_key([.2,0.]),position_key([.2+1e-16,0.]))
        with self.assertRaises(ValueError): canonical_position([1,2],2)

    def test_common_samples_are_pose_independent_and_replicate_specific(self):
        a=position_key([.2,0],8);b=position_key([.3,0],8)
        p=dict(seed=17,seed_policy='common-v2')
        self.assertEqual(render_seed(a,p),render_seed(b,p))
        self.assertNotEqual(render_seed(a,p),render_seed(a,dict(p,seed=18)))
        self.assertNotEqual(render_seed(a,dict(seed=17)),render_seed(b,dict(seed=17)))

    def test_response_is_before_all_opponency_and_preserves_raw_input(self):
        c=UVEncoderConfig(schema='apiaviz-uv-v2',response_half=1.,visible_code_dim=128,uv_code_dim=64)
        new=ApiaVizUVEncoder(c); old=ApiaVizUVEncoder(replace(c,schema='apiaviz-uv-v1',response_half=None))
        x=torch.rand(3,3,18,74,generator=torch.Generator().manual_seed(11))*.2
        x[:,:,4,35]=torch.tensor([1000.,3000.,5000.]);original=x.clone()
        a=new.backbone(x);b=old.backbone(x/(x+1))
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(x,original,atol=0,rtol=0)
        self.assertLessEqual(float(a['colour'].max()),1.)
        self.assertLessEqual(float(a['uv'][:,3:].max()),1.)
        self.assertTrue(torch.isfinite(new(x)).all())

    def test_protocol_factory_keeps_baselines_visible_and_uv_ablation_explicit(self):
        p=dict(encoder=dict(baseline_code_dim=128,uv_config=dict(schema='apiaviz-uv-v2',response_half=1.,visible_code_dim=128,uv_code_dim=64,uv_enabled=False)))
        m=make_model(p,'apiaviz_uv',19);x=torch.rand(1,3,18,74)
        self.assertEqual(torch.count_nonzero(m(x)[:,128:]),0)
        self.assertGreater(torch.count_nonzero(m(x,uv_enabled=True)[:,128:]),0)
        for method in ['sobel_colour','ardin_input']:
            baseline=make_model(p,method,19)
            self.assertFalse(hasattr(baseline,'backbone'))
            self.assertEqual(baseline(x).shape[1],128)

    def test_validation_uses_complete_budgets_and_independent_acquisitions(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory); atomic_json(source/'protocol.json',configuration(False))
            p=protocol(source);off=protocol(source,True)
            atomic_json(source/'protocol.json',configuration(True));small=protocol(source)
        self.assertEqual(p['trials'],36);self.assertEqual(off['trials'],12)
        self.assertEqual(p['evaluation']['max_steps'],200)
        self.assertEqual(p['sensor_settings']['time_budget_s'],360.)
        self.assertNotEqual(p['teaching_seed'],p['render']['seed'])
        self.assertEqual(p['controller_settings'],off['controller_settings'])
        self.assertFalse(off['encoder']['uv_config']['uv_enabled'])
        self.assertEqual(p['worlds'],off['worlds'])
        self.assertEqual(small['evaluation']['max_steps'],200)
        self.assertEqual(small['sensor_settings']['time_budget_s'],360.)

    def test_independent_teaching_never_uses_recall_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            env=Path(directory);a=torch.ones(3,3,9,12);b=a*.5
            torch.save(dict(uv=a,rgb=b),env/'teaching.pt')
            atomic_json(env/'complete.json',dict(assets={'teaching.pt':file_sha(env/'teaching.pt')}))
            uv,rgb=teaching_views(env,{},None,dict(acquisition='independent-teaching-recall-v2'))
            torch.testing.assert_close(uv,a);torch.testing.assert_close(rgb,b)
            torch.save(dict(uv=a*2,rgb=b),env/'teaching.pt')
            with self.assertRaisesRegex(ValueError,'teaching images changed'):
                teaching_views(env,{},None,dict(acquisition='independent-teaching-recall-v2'))

    def test_readiness_does_not_count_arrival_before_a_kick_as_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);rows={};hashes={}
            p=dict(trials=12,methods=['apiaviz_uv'],readiness_rule=dict(
                aligned_arrivals_per_model=4,aligned_trials_per_model=6,
                kick_arrivals_per_model=2,kick_trials_per_model=6))
            atomic_json(out/'protocol.json',p)
            for i in range(12):
                key=str(i);atomic_json(out/f'{key}.json',{'id':key});hashes[key]=file_sha(out/f'{key}.json')
                rows[key]=dict(id=key,trace=f'{key}.json',method='apiaviz_uv',scenario='aligned' if i<6 else 'kick_right50',
                    reached_nest=True,disturbance_applied=False,displacement=None)
            atomic_json(out/'audit.json',dict(passed=True,trials=12,protocol_sha256=file_sha(out/'protocol.json'),traces=hashes))
            with patch('apiaviz.research.uv_trials.verify_sources'),patch('apiaviz.research.uv_trials.load_completed',return_value=rows),redirect_stdout(io.StringIO()):
                self.assertFalse(assess(out)['ready_for_larger_study'])
                for i in (6,7):
                    rows[str(i)].update(disturbance_applied=True,displacement=dict(adjusted=False,applied_distance_m=.5))
                self.assertTrue(assess(out)['ready_for_larger_study'])
                rows['6']['displacement']['adjusted']=True
                self.assertFalse(assess(out)['ready_for_larger_study'])


if __name__=='__main__': unittest.main()
