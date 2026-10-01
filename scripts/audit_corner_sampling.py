"""Offline angular-resolution diagnostic at a taught corner, no policy changes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.spectral_input import file_sha


def run(study,out):
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    env=study/'worlds/bend';bank=torch.load(env/'teaching.pt',weights_only=True)
    camera=DualCamera(env,json.loads((env/'render.json').read_text()))
    angles=np.arange(-10.,71.);records=[]
    for method in p['methods']:
        model=base.make_model(p,method,19);uv=method=='apiaviz_uv'
        images=camera.scan([2.,0.],angles,uv=uv)
        for enabled in ((True,False) if uv else (None,)):
            def coding(x):
                code=model(x,uv_enabled=enabled) if uv else base.encode(model,x)
                return F.normalize((code>0).float(),dim=1)
            similarity=coding(images)@coding(bank['uv' if uv else 'rgb']).T
            scores,station=similarity.max(1)
            records.append(dict(method=method,uv_enabled=enabled,headings=angles.tolist(),
                scores=scores.tolist(),matched_stations=station.tolist()))
    base.atomic_json(out,dict(protocol_sha256=file_sha(study/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),position=[2.,0.],teaching_heading=45.,
        teaching_sha256=file_sha(env/'teaching.pt'),camera_records=camera.references,records=records,
        limitation='One selected taught corner; retrospective diagnostic only. Uses existing independent recall seed. Teaching heading is an offline label, never provided to the policy.'))
    for r in records:
        print(r['method'],r['uv_enabled'],{h:round(r['scores'][r['headings'].index(h)],6) for h in (0.,40.,45.,50.)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
