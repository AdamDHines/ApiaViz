from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from apiaviz.research.uv_parallel import SharedCamera,batches,save_teaching
from apiaviz.research.uv_trials import configuration,cases


class ParallelTrialsTests(unittest.TestCase):
    def test_assignments_cover_pending_trials_with_exclusive_checkpoints(self):
        p=configuration(False); before=deepcopy(p)
        all_cases={c[0]:c for c in cases(p)}
        completed=set(list(all_cases)[:5])
        groups=batches(p,completed); assigned=[key for group in groups for key in group]
        self.assertEqual(set(assigned),set(all_cases)-completed)
        self.assertEqual(len(assigned),len(set(assigned)))
        checkpoints=[]
        for group in groups:
            keys={(all_cases[k][1]['name'],all_cases[k][2],all_cases[k][3]) for k in group}
            self.assertEqual(len(keys),1); checkpoints.extend(keys)
        self.assertEqual(len(checkpoints),len(set(checkpoints)))
        self.assertEqual(p,before)
        self.assertEqual(batches(p,set(all_cases)),[])

    def test_simultaneous_cache_misses_have_one_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            cameras=[SharedCamera(directory,{}) for _ in range(4)]
            writes=[]
            def fake_render(camera,position):
                path=camera.cache/'simulated-frame'
                if not path.exists():
                    time.sleep(.02); writes.append(1); path.write_text('complete')
                return path.read_text()
            with patch('apiaviz.research.dual_camera.DualCamera.frame',fake_render):
                with ThreadPoolExecutor(4) as pool:
                    results=list(pool.map(lambda c:c.frame([0.,0.]),cameras))
            self.assertEqual(results,['complete']*4)
            self.assertEqual(len(writes),1)

    def test_teaching_is_preserved_and_conflicting_images_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env=Path(directory); uv=torch.ones(2,3,4,4); rgb=uv*.5
            save_teaching(env,uv,rgb); original=(env/'teaching.pt').read_bytes()
            save_teaching(env,uv,rgb)
            self.assertEqual((env/'teaching.pt').read_bytes(),original)
            with self.assertRaisesRegex(ValueError,'teaching images changed'):
                save_teaching(env,uv*2,rgb)
            self.assertEqual((env/'teaching.pt').read_bytes(),original)


if __name__=='__main__': unittest.main()
