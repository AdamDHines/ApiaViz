"""Attribute first ApiaViz/Sobel steering disagreement using shared cached poses.

Replay only the two images already used for the temporal comparison. Report
whole-code and separate-stream familiarity changes; no navigation or rendering.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model,atomic_json
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha


def run(study, target):
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text())
    result=dict(protocol_sha256=file_sha(study/'protocol.json'),script_sha256=file_sha(Path(__file__)),records=[],inputs={})
    for w in p['worlds']:
        env=study/'worlds'/w['name']; bp=env/'teaching.pt'
        result['inputs'][str(bp)]=file_sha(bp)
        bank=torch.load(bp,weights_only=True)
        camera=DualCamera(env,json.loads((env/'render.json').read_text()))
        models=dict(apiaviz_uv=make_model(p,'apiaviz_uv',19),sobel_colour=make_model(p,'sobel_colour',19),
                    linear_rgb8k=RefinementEncoder('linear_colour',seed=19,code_dim=8000),
                    sobel_rgb8k=RefinementEncoder('sobel_colour',seed=19,code_dim=8000))
        codes={name:encode(m,bank['uv' if name=='apiaviz_uv' else 'rgb']) for name,m in models.items()}
        for phase in (-1,1):
            trials=[]
            for method in ('apiaviz_uv','sobel_colour'):
                path=study/'trials'/f'{w["name"]}-19-{method}-aligned-familiarity-{phase}.json'
                result['inputs'][str(path)]=file_sha(path);trials.append(json.loads(path.read_text()))
            a,b=trials
            pair=next((x,y) for x,y in zip(a['decisions'],b['decisions']) if (x.get('state'),x.get('movement_heading'))!=(y.get('state'),y.get('movement_heading')))
            step=pair[0]['step'];pos=[a['trace'][step-3]['position'],a['trace'][step-2]['position']]
            shared=all(np.allclose(a['trace'][i]['position'],b['trace'][i]['position'],atol=1e-10,rtol=0) for i in (step-3,step-2))
            heading=pair[0]['comparison']['reference_heading']
            row=dict(world=w['name'],phase=phase,step=step,shared_positions=shared,positions=pos,reference_heading=heading,
                     actual_decisions=[{k:v for k,v in d.items() if k!='motor_feedback'} for d in pair],metrics={})
            for name,model in models.items():
                images=torch.cat([camera.scan(position,[heading],uv=name=='apiaviz_uv') for position in pos])
                query=encode(model,images)
                slices={'all':slice(None)}
                if name=='apiaviz_uv': slices.update(form=slice(0,4000),colour=slice(4000,8000),uv=slice(8000,None),visible=slice(0,8000))
                else:
                    n=query.shape[1]//2;slices.update(form=slice(0,n),colour=slice(n,None))
                row['metrics'][name]={}
                for stream,sl in slices.items():
                    sim=F.normalize((query[:,sl]>0).float(),dim=1)@F.normalize((codes[name][:,sl]>0).float(),dim=1).T
                    score,match=sim.max(1)
                    row['metrics'][name][stream]=dict(scores=score.tolist(),matches=match.tolist(),change=float(score[1]-score[0]))
                if name in ('apiaviz_uv','sobel_colour') and shared:
                    original=pair[0 if name=='apiaviz_uv' else 1]['comparison']
                    # The logged comparisons use these exact positions and gaze.
                    assert abs(row['metrics'][name]['all']['scores'][0]-original['before'])<2e-6
                    assert abs(row['metrics'][name]['all']['scores'][1]-original['after'])<2e-6
            result['records'].append(row)
        for key,value in camera.references.items():result['inputs'][str(env/'camera'/f'{key}.json')]=value
    atomic_json(target,result)
    for r in result['records']:
        print(r['world'],r['phase'],r['step'],'shared',r['shared_positions'],
              {m:{s:round(v['change'],5) for s,v in streams.items()} for m,streams in r['metrics'].items()})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/coherent-validation-v3')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
