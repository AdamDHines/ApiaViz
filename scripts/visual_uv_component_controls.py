"""Fixed UV component removals to test the measured low-pass dominance.

Each control rebuilds teaching codes, retaining all cells, connections, response
scales and timing. Component masks are analytical removals, never fitted weights.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tqdm import tqdm
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.study import encode, fingerprint
from apiaviz.research.spectral_input import file_sha
from visual_response_controls import binary, metrics, select_poses

MASKS=dict(uv_spatial_only=[1,1,1,0,0,0,0], uv_opponent_only=[0,0,0,1,1,1,1],
           uv_no_lowpass=[1,1,0,1,1,1,1])


def job(source_string,out_string,w):
    torch.set_num_threads(2)
    source,out=Path(source_string),Path(out_string)
    parent=json.loads((source/'protocol.json').read_text());study=Path(parent['study'])
    p=json.loads((study/'protocol.json').read_text());env=study/'worlds'/w['name']
    camera=DualCamera(source/w['name'],json.loads((env/'render.json').read_text()))
    bank=torch.load(env/'teaching.pt',weights_only=True);poses=select_poses(w)
    anchor=float(w['headings'][0])
    angles_list=[np.unique(np.round(np.r_[np.arange(-180.,180.,5.)+anchor,np.arange(-20.,21.)+pose['tangent']],8)) for pose in poses]
    images=[camera.scan(pose['position'],a,uv=True) for pose,a in zip(poses,angles_list)]
    rows=[]
    for seed in [19,31,43]:
        model=base.make_model(p,'apiaviz_uv',seed);frozen=fingerprint(model)
        reference=encode(model,bank['uv'])[:,:8000]
        for name,mask in MASKS.items():
            mask_tensor=torch.tensor(mask,dtype=torch.float32)[None,:,None,None]
            def masking(module,inputs,output):return dict(output,uv=output['uv']*mask_tensor)
            handle=model.backbone.register_forward_hook(masking)
            train=encode(model,bank['uv'])
            assert torch.equal(reference>0,train[:,:8000]>0)
            memory=binary(train)
            for pose,angles,image in tqdm(list(zip(poses,angles_list,images)),desc=f'{w["name"]} {seed} {name}'):
                record=metrics(binary(encode(model,image))@memory.T,angles,pose,np.asarray(w['route']),anchor)
                rows.append(dict(world=w['name'],seed=seed,condition=name,**pose,angles=angles.tolist(),**record))
            handle.remove()
        assert fingerprint(model)==frozen
    base.atomic_json(out/f'{w["name"]}.json',dict(records=rows,raw_records=camera.references,
        teaching_sha256=file_sha(env/'teaching.pt')))
    return w['name']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'protocol.json').read_text());p=json.loads((Path(parent['study'])/'protocol.json').read_text())
    source_sha=base.sources()
    for f in ('visual_response_controls.py','visual_uv_component_controls.py'):source_sha['scripts/'+f]=file_sha(ROOT/'scripts'/f)
    base.atomic_json(out/'protocol.json',dict(schema='uv-component-controls-v1',source_protocol_sha256=file_sha(a.source/'protocol.json'),
        source_sha256=source_sha,masks=MASKS,selection='All parent poses, seeds, images, metrics.',
        rationale='Follow-up to measured 85% low-pass versus 6% opponent feature energy; three fixed removals, no tuning or selection.',
        limitation='Offline removal controls, not full-route variants. Plane zeros are imposed before stream standardization, so removing a plane also changes normalization and random-drive competition.'))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(job,str(a.source.resolve()),str(out),w) for w in p['worlds']]
        for future in futures:print('Completed',future.result(),flush=True)
    base.atomic_json(out/'complete.json',dict(protocol_sha256=file_sha(out/'protocol.json'),
        results={w['name']:file_sha(out/f'{w["name"]}.json') for w in p['worlds']}))


if __name__=='__main__':main()
