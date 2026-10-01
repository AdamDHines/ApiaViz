"""Numerical UV control on flat fields and every saved acquisition test image."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from apiaviz.research.uv_encoder import ApiaVizUVEncoder, UVEncoderConfig
from apiaviz.research.uv_precision import UVPrecisionEncoder
from apiaviz.research.uv_trials import atomic_json,sources
from apiaviz.research.spectral_input import file_sha


def run(acquisition,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    config=UVEncoderConfig(schema='apiaviz-uv-v2',response_half=1.)
    old,new=ApiaVizUVEncoder(config),UVPrecisionEncoder(config)
    result=dict(source_sha256=sources(),inputs={},flat_fields=[],acquisitions=[],
        limitation='Numerical invariance/repeatability check, not route success evidence.')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in result['source_sha256']:z.write(ROOT/name,name)
    for value in [0.,.01,.1,.5,1.,10.,1000.]:
        x=torch.full((1,3,51,199),value)
        counts=[[int(d['counts'].sum()) for d in m.diagnostics(x)['streams']] for m in [old,new]]
        assert counts[1]==[0,0,0]
        result['flat_fields'].append(dict(radiance=value,v2_spikes=counts[0],v3_spikes=counts[1]))
    for path in sorted(acquisition.glob('*/retina.pt')):
        result['inputs'][str(path)]=file_sha(path)
        images=torch.load(path,weights_only=True)
        for key in ['uv','shifted','headed']:
            a,b=[m(images[key]) for m in [old,new]]
            result['acquisitions'].append(dict(source=path.parent.name,stimulus=key,
                changed_spike_bits=int(((a>0)!=(b>0)).sum()),bits=a.numel(),
                overlap=F.cosine_similarity((a>0).float(),(b>0).float()).tolist()))
    atomic_json(out/'validation.json',result)
    print('Flat fields:',result['flat_fields'])
    print('Changed acquired-image bits:',sum(r['changed_spike_bits'] for r in result['acquisitions']),
        '/',sum(r['bits'] for r in result['acquisitions']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acquisition',type=Path,default=ROOT/'apiaviz/output/uv-acquisition-v2-check1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.acquisition,a.output)
