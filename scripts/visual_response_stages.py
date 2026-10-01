"""Localize yaw sensitivity before/after fixed projection and spike readout."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.study import encode
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize
from visual_response_controls import binary, select_poses


@torch.no_grad()
def stages(model, images, uv=False):
    output={}
    for offset in range(0,len(images),16):
        x=images[offset:offset+16]
        if uv:
            maps=model.backbone(x)
            pooled=[adaptive_avg_pool2d_anysize(maps[n],(8,64)) for n in model.stream_names]
        else: pooled=model.pooled_features(x)
        features=[]
        for v in pooled:
            flat=v.flatten(1)
            features.append(F.normalize(flat-flat.mean(1,keepdim=True),dim=1))
        currents=model.currents(x)
        values=dict(features=F.normalize(torch.cat(features,1),dim=1),
                    currents=F.normalize(torch.cat([F.normalize(c,dim=1) for c in currents],1),dim=1),
                    spikes=binary(encode(model,x)))
        for name,part in zip(('form','colour','uv'),features):values[name+'_features']=part
        for name,v in values.items():output.setdefault(name,[]).append(v)
    return {name:torch.cat(parts) for name,parts in output.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    parent=json.loads((a.source/'protocol.json').read_text());study=Path(parent['study'])
    p=json.loads((study/'protocol.json').read_text())
    base.atomic_json(out/'protocol.json',dict(source_protocol_sha256=file_sha(a.source/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),source_sha256=base.sources(),seed=19,
        selection='All 21 preselected taught positions; dense -20..20 degree yaw; same saved independent recall.',
        definition='Feature and current streams normalized separately then concatenated equally; spike readout is actual production normalized binary concatenation. Stage cosines are diagnostics, not interchangeable memory scores.'))
    rows=[]
    for w in tqdm(p['worlds'],desc='Response stages'):
        env=study/'worlds'/w['name'];bank=torch.load(env/'teaching.pt',weights_only=True)
        camera=DualCamera(a.source/w['name'],json.loads((env/'render.json').read_text()))
        models=dict(api_uv=base.make_model(p,'apiaviz_uv',19),
                    api_rgb=RefinementEncoder('linear_colour',seed=19),
                    sobel=RefinementEncoder('sobel_colour',seed=19))
        for name,model in models.items():
            uv=name=='api_uv'
            memory=stages(model,bank['uv' if uv else 'rgb'],uv)
            for pose in select_poses(w):
                if pose['kind']!='taught':continue
                images=camera.scan(pose['position'],np.arange(-20.,21.)+pose['tangent'],uv=uv)
                query=stages(model,images,uv)
                for stage,v in query.items():
                    sim=v@memory[stage].T
                    curve=sim[:,pose['station']].tolist()
                    rows.append(dict(world=w['name'],station=pose['station'],method=name,stage=stage,
                        curve=curve,repeat=curve[20],yaw5_loss=curve[20]-.5*(curve[15]+curve[25]),
                        noise_loss=1-curve[20]))
    base.atomic_json(out/'results.json',dict(records=rows,protocol_sha256=file_sha(out/'protocol.json')))


if __name__=='__main__':main()
