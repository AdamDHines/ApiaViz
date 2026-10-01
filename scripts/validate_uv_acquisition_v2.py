"""Bounded, independent-seed acquisition checks; never runs navigation."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tqdm import tqdm
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model,atomic_json
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.familiarity_controller import acquisition_calibration
from apiaviz.research.spike_overlap import SpikeOverlapMemory


def run(source,out):
    out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(2)
    p=json.loads((source/'protocol.json').read_text()); p['encoder']['uv_config']=dict(schema='apiaviz-uv-v2',response_half=1.)
    report=dict(schema='apiaviz-acquisition-validation-v2',source_protocol_sha256=file_sha(source/'protocol.json'),
        response_half=1.,pose_decimals=8,seed_policy='common-v2',conditions=[],roundoff_equal=True,
        interpretation='Independent seeds measure repeatability; a cache hit alone is not a convergence test.')
    for world in tqdm(p['worlds'],desc='Acquisition worlds'):
        env=source/'worlds'/world['name']; wp=world
        indices=[2,3,len(wp['route'])//2]
        for spp in [64,256]:
            arrays=[]
            for seed in [20261001,20261002]:
                target=out/f'{world["name"]}-{spp}-{seed}'; target.mkdir()
                shutil.copytree(env/'geometry',target/'geometry'); shutil.copyfile(env/'calibration.json',target/'calibration.json')
                config=json.loads((env/'render.json').read_text()); config.update(spp=spp,seed=seed,pose_decimals=8,seed_policy='common-v2')
                atomic_json(target/'render.json',config)
                uv=[];rgb=[]; shifted=[];headed=[];started=time.perf_counter()
                with DualCamera(target,config) as camera:
                    for index in indices:
                        pos=np.array(wp['route'][index]); h=wp['headings'][index]
                        a=camera.scan(pos,[h],uv=True)
                        before=camera.renders
                        b=camera.scan(pos+[1e-16,-1e-16],[h],uv=True)
                        assert torch.equal(a,b) and camera.renders==before
                        uv.append(a); rgb.append(camera.scan(pos,[h]));shifted.append(camera.scan(pos+[1e-5,0],[h],uv=True))
                        headed.append(camera.scan(pos,[h+30,h-30],uv=True))
                arrays.append(dict(uv=torch.cat(uv),rgb=torch.cat(rgb),shifted=torch.cat(shifted),headed=torch.cat(headed)))
                torch.save(arrays[-1],target/'retina.pt')
                print(world['name'],spp,seed,'seconds',round(time.perf_counter()-started,2),flush=True)
            for method in p['methods']:
                model=make_model(p,method,19); key='uv' if method=='apiaviz_uv' else 'rgb'
                a,b=[encode(model,r[key]) for r in arrays]
                similarity=torch.nn.functional.cosine_similarity((a>0).float(),(b>0).float()).numpy()
                item=dict(world=world['name'],spp=spp,method=method,stations=indices,same_pose_independent_seed_similarity=similarity.tolist())
                if method=='apiaviz_uv':
                    shifted=encode(model,arrays[0]['shifted']);headed=encode(model,arrays[0]['headed'])
                    item['ten_micron_translation_similarity']=torch.nn.functional.cosine_similarity((a>0).float(),(shifted>0).float()).tolist()
                    item['thirty_degree_heading_similarity']=torch.nn.functional.cosine_similarity((a.repeat_interleave(2,0)>0).float(),(headed>0).float()).tolist()
                report['conditions'].append(item)
            atomic_json(out/'validation.json',report)
    paths=['apiaviz/src/uv.py','apiaviz/research/uv_encoder.py','apiaviz/research/dual_camera.py','scripts/uv_mitsuba/trial_camera.py','scripts/validate_uv_acquisition_v2.py']
    report['source_sha256']={name:file_sha(ROOT/name) for name in paths};report['complete']=True
    atomic_json(out/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'apiaviz/output/uv-trials-v1');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
