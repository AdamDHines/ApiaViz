"""Counterfactual angular retrieval at recorded aligned-route positions.

Camera caches only, no rendering/navigation. Ground-truth route is used only to
label retrieval error, never to select a movement or train the encoder.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model,atomic_json,sources
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha


def run(study,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text())
    result=dict(source_sha256=sources(),inputs={},records=[],
        selection='First ten and every tenth subsequent macro endpoint in all six aligned ApiaViz v2 trials.',
        limitation='Retrospective counterfactual scans are not free policy observations or navigation outcomes.')
    model=make_model(p,'apiaviz_uv',19)
    for w in p['worlds']:
        env=study/'worlds'/w['name'];bank=env/'teaching.pt'
        result['inputs'][str(bank)]=file_sha(bank)
        codes=encode(model,torch.load(bank,weights_only=True)['uv'])
        camera=DualCamera(env,json.loads((env/'render.json').read_text()))
        route=np.array(w['route']);headings=np.array(w['headings'])
        for phase in [1,-1]:
            path=study/'trials'/f'{w["name"]}-19-apiaviz_uv-aligned-familiarity-{phase}.json'
            result['inputs'][str(path)]=file_sha(path);d=json.loads(path.read_text())
            for r in d['trace']:
                if r['step']>10 and r['step']%10:continue
                pos=np.asarray(r['position']);index=int(np.linalg.norm(route-pos,axis=1).argmin())
                offsets=np.arange(-180.,180.,10.)
                angles=headings[index]+offsets
                query=encode(model,camera.scan(pos,angles,uv=True))
                streams={}
                for name,sl in [('all',slice(None)),('visible',slice(0,8000)),('uv',slice(8000,None))]:
                    sim=F.normalize((query[:,sl]>0).float(),dim=1)@F.normalize((codes[:,sl]>0).float(),dim=1).T
                    score,match=sim.max(1);i=int(score.argmax())
                    streams[name]=dict(best_offset_deg=float(offsets[i]),best_score=float(score[i]),
                        matched_station=int(match[i]),directional_contrast=float(score[i]-score.median()),
                        correct_tangent_score=float(score[18]))
                result['records'].append(dict(world=w['name'],phase=phase,step=r['step'],
                    polyline_m=r['polyline_m'],nearest_station=index,streams=streams))
        for key,value in camera.references.items():result['inputs'][str(env/'camera'/f'{key}.json')]=value
    atomic_json(out/'audit.json',result)
    for name in ['all','visible','uv']:
        for near in [True,False]:
            rows=[r for r in result['records'] if (r['polyline_m']<=.05)==near]
            print(name,'near' if near else 'off route',len(rows),'within20deg',
                sum(abs(r['streams'][name]['best_offset_deg'])<=20 for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/uv-validation-v2')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
