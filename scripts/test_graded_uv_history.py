"""Separate fast integration from slow gain history at unchanged illumination."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tqdm import tqdm
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.graded_uv import GradedUVEncoder,initialize,advance
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.study import fingerprint
from test_graded_uv_controls import pooled,summarize,tensor_sha,DT,SEEDS
from visual_response_controls import select_poses


def responses(images,initial,dt,variant):
    state=initialize(initial);result=[]
    for x in images:
        _,state=advance(x,state,dt)
        a=x.mean((2,3),keepdim=True).clamp_min(1e-4) if variant=='instant_gain' else 1.
        result.append(state.irradiance/(state.irradiance+a))
    return torch.stack(result)


def job(source_string,out_string,w):
    torch.set_num_threads(2);source,out=Path(source_string),Path(out_string)
    parent=json.loads((source/'protocol.json').read_text());study=Path(parent['study']);env=study/'worlds'/w['name']
    camera=DualCamera(source/w['name'],json.loads((env/'render.json').read_text()))
    bank=torch.load(env/'teaching.pt',weights_only=True)
    models={s:GradedUVEncoder(s) for s in SEEDS};frozen={str(s):fingerprint(m) for s,m in models.items()}
    memory={};hashes={};rows=[]
    for variant in ['instant_gain','integration_only']:
        train=responses(bank['uv'][:,None],bank['uv'][:1],1.,variant)[:,0]
        for mode in ['pairwise','combined']:
            features=pooled(models[19],train,mode)
            for seed,m in models.items():
                for readout,c in m.readouts(features).items():
                    key=variant,mode,seed,readout;memory[key]=c;hashes[str(key)]=tensor_sha(c)
    anchor=float(w['headings'][0]);route=np.asarray(w['route'])
    for pose in tqdm(select_poses(w),desc=w['name']+' history control'):
        initial=camera.scan(pose['position'],[anchor],uv=True)
        for direction in [-1,1]:
            angles=anchor+direction*np.arange(0.,360.,5.);raw=camera.scan(pose['position'],angles,uv=True)
            for variant in ['instant_gain','integration_only']:
                response=responses(raw[:,None],initial,DT,variant)[:,0]
                for mode in ['pairwise','combined']:
                    features=pooled(models[19],response,mode)
                    for seed,m in models.items():
                        for readout,c in m.readouts(features).items():
                            rows.append(dict(world=w['name'],**pose,seed=seed,variant=variant,mode=mode,readout=readout,
                                direction=direction,gain=1.,**summarize(c@memory[variant,mode,seed,readout].T,angles,pose,route,anchor)))
    assert frozen=={str(s):fingerprint(m) for s,m in models.items()}
    for key,c in memory.items():assert hashes[str(key)]==tensor_sha(c)
    base.atomic_json(out/f'{w["name"]}.json',dict(records=rows,raw_records=camera.references,
        teaching_sha256=file_sha(env/'teaching.pt'),memory_sha256=hashes,encoder_fingerprints=frozen))
    return w['name']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'protocol.json').read_text());study=Path(parent['study']);worlds=json.loads((study/'protocol.json').read_text())['worlds']
    base.atomic_json(out/'protocol.json',dict(schema='graded-uv-history-v1',source=str(a.source.resolve()),
        source_protocol_sha256=file_sha(a.source/'protocol.json'),source_sha256=base.sources(),
        variants=dict(instant_gain='Retain fast 20ms irradiance integration; replace slow background with current retinal mean.',
                      integration_only='Retain fast 20ms irradiance integration; fixed denominator one, no gain adaptation.'),
        selection='All original 84 poses, three seeds, two scan directions, both opponent axes and all readouts; normal illumination only.',
        limitation='Follow-up mechanism control after observing scan-order sensitivity. Instant normalization is a counterfactual, not fitted biological adaptation or a proposed trial default.'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(job,str(a.source.resolve()),str(out),w) for w in worlds]
        for f in jobs:print('Completed',f.result(),flush=True)
    base.atomic_json(out/'complete.json',dict(protocol_sha256=file_sha(out/'protocol.json'),
        results={w['name']:file_sha(out/f'{w["name"]}.json') for w in worlds}))


if __name__=='__main__':main()
