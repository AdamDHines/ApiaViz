"""Image-only flow comparison on the frozen paired contact-approach renders."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research.dual_camera import sensor_views
from apiaviz.research.coherent_navigation import CoherentAvoidance
from apiaviz.research.visual_avoidance import gray
from apiaviz.research.uv_trial_report import display_response
from apiaviz.research.spectral_input import file_sha


def run(source,out):
    out.mkdir(parents=True,exist_ok=True)
    target=out/'surface-contact.json'
    if target.exists():raise FileExistsError(target)
    d=json.loads((source/'renders.json').read_text());records=[]
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
    for i,name in enumerate(('flat','detailed')):
        frames=[]
        for phase in ('before','after'):
            path=source/f'{name}-{phase}.npy';r=next(r for r in d['records'] if r['file']==path.name)
            assert file_sha(path)==r['sha256']
            frames.append(sensor_views(np.load(path),[d['heading']],d['render_config'])[0].permute(1,2,0).numpy())
        policy=CoherentAvoidance();diagnostic=policy.moved(*frames,d['heading'],d['distance_m'])
        heading,stride,decision=policy.choose(d['heading'])
        a=gray(frames[1]);frontal=a[22:39,85:114]
        records.append(dict(surface=name,diagnostic=diagnostic,decision=decision,
            frontal_gray_std=float(frontal.std()),points=policy.points.tolist()))
        axes[i,0].imshow(display_response(frames[1][:,::-1]),aspect='auto')
        axes[i,0].axis('off');axes[i,0].set_title(f'{name}: visible response · fixed display transfer')
        points=policy.points
        axes[i,1].scatter(points[:,1],points[:,0],s=8)
        axes[i,1].plot([0,0],[0,.18],'r--')
        axes[i,1].set(xlim=(-.3,.3),ylim=(-.05,.4),aspect='equal',xlabel='Lateral estimate (m)',ylabel='Forward estimate (m)',
            title=f"{name}: {decision['state']} · visual risk {decision['frontal_risk']:.2f}")
    fig.suptitle('Same recorded pre-contact translation · same integrated retina · no contact/geometry passed to avoidance')
    fig.savefig(out/'surface-contact.png',dpi=150);plt.close(fig)
    target.write_text(json.dumps(dict(records=records,source_sha256=file_sha(source/'renders.json'),
        script_sha256=file_sha(Path(__file__)),distance_m=d['distance_m'],heading=d['heading'],
        limitation='One prespecified historical contact approach; no navigation or generalized avoidance claim.'),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='points'} for r in records],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'apiaviz/output/surface-contact-probe-v1')
    p.add_argument('--output',type=Path,default=ROOT/'docs/navigation-fidelity-v1');a=p.parse_args();run(a.source,a.output)
