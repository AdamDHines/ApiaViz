"""Retrospective image-only replay of the first rejected ApiaViz hairpin scan.

Uses an already recorded position; no movements, new renders or controller edits.
Teaching indices/headings are reported only as evaluator-side diagnostics.
"""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.familiarity_controller import FamiliarityController,Settings
from apiaviz.research.spectral_input import file_sha


def run(study,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'ambiguity.json').exists():raise FileExistsError('Preserve diagnostic')
    torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    path=study/'trials/hairpin-19-apiaviz_uv-aligned-familiarity-1.json'
    detail=json.loads(path.read_text())
    decision=next(v for v in detail['decisions'] if 'scan' in v and not v['scan']['supported'])
    previous=next(v for v in detail['trace'] if v['step']==decision['step']-1)
    position=previous['position'];headings=[v['heading'] for v in decision['scan']['samples']]
    env=study/'worlds/hairpin';world=next(w for w in p['worlds'] if w['name']=='hairpin')
    camera=DualCamera(env,json.loads((env/'render.json').read_text()))
    records=[]
    for method in p['methods']:
        checkpoint_path=env/f'encoder-{method}-19.pt'
        saved=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
        model=base.make_model(p,method,19)
        codes=base.encode(model,camera.scan(position,headings,uv=method=='apiaviz_uv'))
        if method!='apiaviz_uv':codes=(codes>0).float()
        scores=torch.nn.functional.normalize(codes,dim=1)@saved['memory']['memory'].T
        best,indices=scores.max(1);values=dict(zip(headings,best.tolist()))
        controller=FamiliarityController(saved['calibration'],Settings(**p['controller_settings']),phase=1)
        target,contrast,supported=controller._peak(values,previous['heading'])
        samples=[dict(heading=h,familiarity=float(value),teaching_index=int(index),
            taught_heading=world['headings'][index]) for h,value,index in zip(headings,best,indices)]
        if method=='apiaviz_uv':
            np.testing.assert_allclose(best.numpy(),[v['familiarity'] for v in decision['scan']['samples']],atol=1e-6,rtol=0)
            assert target==decision['scan']['target'] and supported==decision['scan']['supported']
        records.append(dict(method=method,target=target,contrast=contrast,supported=bool(supported),
            calibration=saved['calibration'],ambiguity_score_margin=controller.settings.ambiguity_tolerance*controller.scale,
            checkpoint_sha256=file_sha(checkpoint_path),samples=samples))
    assert camera.renders==0
    base.atomic_json(out/'ambiguity.json',dict(protocol_sha256=file_sha(study/'protocol.json'),
        trace_sha256=file_sha(path),script_sha256=file_sha(Path(__file__)),recorded_step=decision['step'],
        position=position,camera_references=camera.references,records=records,
        limitation='Post-hoc selected failed-trial pose, existing camera only. Counterfactual baseline scans are not new route trials. Teaching labels are analysis only, never controller inputs.'))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-graded-nav-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apiaviz.research.uv_trial_report import NAMES,COLORS
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for record in records:
        samples=sorted(record['samples'],key=lambda v:v['heading']);x=[v['heading'] for v in samples]
        y=np.array([v['familiarity'] for v in samples]);scale=record['calibration']['scale']
        for ax,values in zip(axes,[y,(y-y.max())/scale]):
            ax.plot(x,values,label=NAMES[record['method']],color=COLORS[record['method']])
    axes[0].set(ylabel='Best taught-view cosine similarity',title=f'Hairpin step {decision["step"]}: same camera pose for all models')
    axes[0].legend();axes[1].set(xlabel='Heading (degrees)',ylabel='Difference from best / teaching scale',ylim=(-.8,.03))
    axes[1].axhline(-p['controller_settings']['ambiguity_tolerance'],color='black',ls=':',label='Ambiguity margin')
    axes[1].legend()
    for ax in axes:
        for heading in (80,140):ax.axvline(heading,color='gray',lw=.5,ls='--')
    fig.savefig(out/'ambiguity.png',dpi=160);plt.close(fig)
    for record in records:
        print(record['method'],record['target'],record['supported'],record['contrast'])
        print(sorted(record['samples'],key=lambda x:-x['familiarity'])[:3])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/graded-navigation-smoke-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/graded-navigation-smoke-v1')
    args=parser.parse_args();run(args.study.resolve(),args.output.resolve())
