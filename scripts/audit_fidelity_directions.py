"""Read-only local-versus-global retrieval audit of one completed fidelity case."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model,atomic_json
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha


def run(study,trial,out):
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2);p=json.loads((study/'protocol.json').read_text())
    path=study/'trials'/f'{trial}.json';d=json.loads(path.read_text());row=d['result']
    w=next(w for w in p['worlds'] if w['name']==row['world']);route=np.array(w['route'])
    env=study/'worlds'/w['name'];camera=DualCamera(env,json.loads((env/'render.json').read_text()))
    model=make_model(p,row['method'],row['seed']);bank=torch.load(env/'teaching.pt',weights_only=True)
    memory=F.normalize((encode(model,bank['uv'])>0).float(),dim=1)
    records=[]
    for dec in d['decisions']:
        step=dec['step']
        if step>60 or 'scan' not in dec:continue
        pos=d['trace'][step-2]['position'] if step>1 else row['initial_position']
        station=int(np.linalg.norm(route-pos,axis=1).argmin())
        angles=np.arange(-180.,180.,10.)+w['headings'][0]
        query=F.normalize((encode(model,camera.scan(pos,angles,uv=True))>0).float(),dim=1)
        score,match=(query@memory.T).max(1);i=int(score.argmax())
        scan=dec['scan'];logged=max(s['familiarity'] for s in scan['samples'])
        target=float(angles[i]);tangent=w['headings'][station]
        records.append(dict(step=step,position=pos,polyline_m=d['trace'][step-2]['polyline_m'] if step>1 else 0,
            nearest_station=station,tangent=tangent,selected=scan['target'],global_target=target,
            global_score=float(score[i]),local_score=logged,global_advantage=float(score[i]-logged),
            selected_error=float(abs((scan['target']-tangent+180)%360-180)),
            global_error=float(abs((target-tangent+180)%360-180)),matched_station=int(match[i]),
            scan_size=len(scan['samples']),supported=scan['supported']))
    atomic_json(out,dict(trial_sha256=file_sha(path),protocol_sha256=file_sha(study/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),records=records))
    for r in records:print(r)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--trial',default='meander-19-apiaviz_uv-aligned-familiarity--1');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.trial,a.output)
