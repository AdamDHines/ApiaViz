from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from apiaviz.research.dual_camera import BANDS,digest,load_cached,position_key,visible_weights,visible_image
from apiaviz.research.route_full import cases
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.uv_trials import configuration,TrialScorer,atomic_json,load_completed,verify_sources
from apiaviz.research.uv_trial_report import movie_selection,path_with_kicks,statistics


class UVTrialsTests(unittest.TestCase):
    def test_full_and_smoke_matrices_are_explicit_and_separate(self):
        full=configuration(False); smoke=configuration(True)
        self.assertEqual(len(list(cases(full))),378)
        self.assertEqual(len(list(cases(smoke))),6)
        self.assertEqual(full['methods'],['apiaviz_uv','sobel_colour','ardin_input'])
        self.assertEqual(full['encoder']['visible_code_dim']+full['encoder']['uv_code_dim'],full['encoder']['baseline_code_dim'])
        self.assertEqual(full['evaluation_safety']['schema'],'navigation-safety-v1')
        self.assertEqual(len({w['geometry_seed'] for w in full['worlds']}),3)

    def test_visible_response_cannot_receive_uv(self):
        wl=np.arange(320.,701.,5.); weights=visible_weights(wl)
        self.assertTrue(np.all(weights[:,wl<400]==0))
        self.assertTrue(all(np.interp(399.9,wl,band)==0 for band in weights))
        np.testing.assert_allclose(np.trapz(weights,wl,axis=1),1.)
        self.assertTrue(np.all(visible_image(np.array([0.,1.,100.]))<1))
        calls=[]
        class Camera:
            def scan(self,p,h,*,uv=False):
                calls.append(uv)
                return torch.ones(len(h),3,8,8)
        model=lambda x:torch.ones(len(x),10)
        memory=lambda c:-c.mean(1)
        for method,expected in [('apiaviz_uv',True),('sobel_colour',False),('ardin_input',False)]:
            scorer=TrialScorer(Camera(),model,memory,method)
            scorer([0,0],[0]); self.assertEqual(calls[-1],expected)
            scorer.world.scan([0,0],[0]); self.assertFalse(calls[-1]) # Reflex always visible.

    def test_raw_cache_is_hashed_and_pose_specific(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); position=[0.,.1]; key=position_key(position)
            self.assertEqual(position_key([0.,.1]),position_key([-0.,.1]))
            self.assertNotEqual(key,position_key([0.,.1000000000001]))
            data=np.ones((8,16,6),dtype=np.float32)*3
            np.savez_compressed(d/f'{key}.npz',responses=data)
            meta=dict(schema='apiaviz-dual-frame-v1',linear=True,position=position,render_hash='frozen',channels=BANDS,shape=list(data.shape),array_sha256=file_sha(d/f'{key}.npz'))
            atomic_json(d/f'{key}.json',meta)
            np.testing.assert_array_equal(load_cached(d,key,'frozen')[0],data)
            with self.assertRaises(ValueError): load_cached(d,key,'other')
            np.savez_compressed(d/f'{key}.npz',responses=data*.5)
            with self.assertRaisesRegex(ValueError,'checksum'): load_cached(d,key,'frozen')

    def test_source_drift_and_unexpected_resume_records_fail(self):
        p=configuration(True); verify_sources(p)
        changed=deepcopy(p); changed['source_sha256']['pixi.toml']='wrong'
        with self.assertRaisesRegex(ValueError,'Source changed'): verify_sources(changed)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); (out/'trials').mkdir(); atomic_json(out/'protocol.json',p)
            atomic_json(out/'trials/bad.json',dict(result=dict(id='not-a-planned-trial')))
            with self.assertRaisesRegex(ValueError,'Unexpected'): load_completed(out,p)

    def test_movie_selection_is_prespecified_and_includes_first_failure(self):
        p=configuration(False); rows=[]
        for key,world,seed,method,scenario,phase in cases(p):
            rows.append(dict(id=key,world=world['name'],seed=seed,method=method,scenario=scenario['name'],phase=phase,
                reached_nest=not (scenario['name']=='left50' and phase==-1)))
        selected=movie_selection(list(reversed(rows)),p)
        self.assertEqual(len(selected),27) # 18 prespecified + 9 first failures.
        self.assertEqual(len({r['id'] for r in selected}),27)
        self.assertEqual(sum(not r['reached_nest'] for r in selected),9)

    def test_movie_path_does_not_draw_external_kick_as_walking(self):
        detail=dict(microtrace=[dict(time_s=1.,position=[.1,0.]),dict(time_s=2.,position=[.2,.5])],
            events=[dict(kind='displacement',time_s=1.,position=[.1,.5])])
        path,times=path_with_kicks(detail,[0,0])
        self.assertTrue(np.isnan(path[2]).all())
        np.testing.assert_array_equal(path[3],[.1,.5])
        np.testing.assert_array_equal(times,[0,1,1,1,2])


if __name__=='__main__': unittest.main()
