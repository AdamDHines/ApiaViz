"""Read-only post-run diagnosis using saved trials and camera arrays; no rendering.

Counterfactual response compression and stream ablations are diagnostic probes,
not corrected navigation outcomes. Original protocols and results are untouched.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from apiaviz.research.dual_camera import DualCamera,position_key
from apiaviz.research.uv_encoder import ApiaVizUVEncoder,UVEncoderConfig
from apiaviz.research.uv_trials import verify_sources
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize


def percentiles(values):
    return dict(zip(('p10','median','p90'),map(float,np.quantile(values,[.1,.5,.9])))) if len(values) else None


def similarities(a,b):
    parts=((0,10000,'all'),(0,8000,'uv_off'),(0,4000,'form'),(4000,8000,'colour'),(8000,10000,'uv'))
    return {name:float(F.cosine_similarity((a[:,lo:hi]>0).float(),(b[:,lo:hi]>0).float()).item()) for lo,hi,name in parts}


def run(study,out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    p=json.loads((study/'protocol.json').read_text()); verify_sources(p)
    rows=json.loads((study/'report/trials.json').read_text()); audit=json.loads((study/'audit.json').read_text())
    result=dict(protocol_sha256=file_sha(study/'protocol.json'),audit_sha256=file_sha(study/'audit.json'),
        script_sha256=file_sha(Path(__file__)),completed=len(rows),scope='Post-hoc diagnosis, no new navigation or renderer calls',groups=[],representatives=[],features=[],repeatability=[])
    aggregates={m:dict(block_reasons=Counter(),terminal_rule=Counter(),unsupported_contrast=[],last_deviation=[],first_cast_steps=[]) for m in p['methods']}
    aligned={}
    for r in tqdm(rows,desc='Read frozen traces'):
        path=study/r['trace']; data=path.read_bytes()
        assert hashlib.sha256(data).hexdigest()==audit['traces'][r['id']],r['id']
        d=json.loads(data); assert d['protocol_sha256']==file_sha(study/'protocol.json')
        a=aggregates[r['method']]
        a['block_reasons'].update(e['reason'] for e in d['events'] if e['kind']=='blocked_proposal')
        casts=[dec['step'] for dec in d['decisions'] if dec.get('state')=='cast']
        if casts:a['first_cast_steps'].append(casts[0])
        if r['termination']=='uninformative_views':
            scan=d['decisions'][-1]['scan']; a['unsupported_contrast'].append(scan['contrast'])
            ck=torch.load(study/'worlds'/r['world']/f"encoder-{r['method']}-{r['seed']}.pt",weights_only=True,map_location='cpu')
            samples=scan['samples']; best=max(s['familiarity'] for s in samples)
            competing=any(abs((s['heading']-scan['target']+180)%360-180)>40 and best-s['familiarity']<=p['controller_settings']['ambiguity_tolerance']*ck['calibration']['scale'] for s in samples)
            low=scan['contrast']<p['controller_settings']['contrast_threshold']
            a['terminal_rule'].update([('low_contrast_and_competing' if competing else 'low_contrast') if low else ('competing_peak' if competing else 'other')])
            if d['trace']: a['last_deviation'].append(d['trace'][-1]['polyline_m'])
        if r['scenario']=='aligned' and r['seed']==19 and r['phase']==1:
            aligned[r['world'],r['method']]=d
            block=next((e for e in d['events'] if e['kind']=='blocked_proposal'),None)
            result['representatives'].append(dict(id=r['id'],termination=r['termination'],steps=r['steps'],
                first_block=None if block is None else {k:block[k] for k in ('time_s','reason','position')},
                first_12=[{k:dec[k] for k in ('step','state','familiarity','movement_heading','trend') if k in dec} for dec in d['decisions'][:12]]))
    for method,a in aggregates.items():
        selected=[r for r in rows if r['method']==method]; aligned_rows=[r for r in selected if r['scenario']=='aligned']
        result['groups'].append(dict(method=method,n=len(selected),arrivals=sum(r['reached_nest'] for r in selected),
            aligned_arrivals=sum(r['reached_nest'] for r in aligned_rows),aligned_n=len(aligned_rows),
            terminations=dict(Counter(r['termination'] for r in selected)),block_reasons=dict(a['block_reasons']),
            terminal_rule=dict(a['terminal_rule']),unsupported_contrast=percentiles(a['unsupported_contrast']),
            uninformative_final_route_distance_m=percentiles(a['last_deviation']),first_cast_step=percentiles(a['first_cast_steps'])))
    for world in tqdm(p['worlds'],desc='Cached spectral diagnostics'):
        env=study/'worlds'/world['name']; teaching=torch.load(env/'teaching.pt',weights_only=True,map_location='cpu'); x=teaching['uv']
        model=ApiaVizUVEncoder(UVEncoderConfig(seed=19)); ck=torch.load(env/'encoder-apiaviz_uv-19.pt',weights_only=True,map_location='cpu'); model.load_state_dict(ck['encoder'])
        train=encode(model,x); compressed=encode(model,x/(1+x))
        feature=dict(world=world['name'],raw_receptor_quantiles=np.quantile(x.numpy(),[.5,.9,.999,1],axis=(0,2,3)).tolist(),
            pixels_any_channel_over_one_fraction=float((x.max(1).values>1).float().mean()),streams={},calibration={})
        for label,inputs in [('raw',x),('compressed_diagnostic',x/(1+x))]:
            for stream,maps in model.backbone(inputs).items():
                pooled=adaptive_avg_pool2d_anysize(maps,(8,64)); energy=pooled.square().flatten(1)
                concentration=energy.topk(5,1).values.sum(1)/energy.sum(1).clamp_min(1e-15)
                feature['streams'][label+'_'+stream]=percentiles(concentration.numpy())
        for method in p['methods']:
            c=torch.load(env/f'encoder-{method}-19.pt',weights_only=True,map_location='cpu')
            feature['calibration'][method]=c['calibration']
        result['features'].append(feature)
        # Same physical position and gaze, but a distinct exact-float cache key.
        # No scene changes or new rendering: compare previously acquired arrays.
        camera=DualCamera(env,json.loads((env/'render.json').read_text()))
        route=np.array(world['route'][:-1]); seen=set()
        for event in aligned[world['name'],'apiaviz_uv']['events']:
            if event['kind']!='observation':continue
            delta=np.linalg.norm(route-event['position'],axis=1); index=int(delta.argmin())
            if delta[index]>1e-10 or abs((event['heading']-world['headings'][index]+180)%360-180)>1e-8:continue
            key=position_key(event['position'])
            if key==position_key(route[index]) or key in seen:continue
            seen.add(key)
            y=camera.scan(event['position'],[event['heading']],uv=True)
            raw_codes=encode(model,y); comp_codes=encode(model,y/(1+y))
            result['repeatability'].append(dict(world=world['name'],station=index,position_difference_m=float(delta[index]),
                teaching_key=position_key(route[index]),query_key=key,
                raw_similarity=similarities(train[index:index+1],raw_codes),
                compressed_diagnostic_similarity=similarities(compressed[index:index+1],comp_codes),
                recorded_familiarity=event['familiarity'],query_max=y.amax((0,2,3)).tolist(),teaching_max=x[index].amax((1,2)).tolist()))
            if len(seen)>=6:break
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    groups=result['groups']; names=['ApiaViz + UV','Sobel + colour','Ardin-style']; colors=['#168e88','#cc8735','#7978b9']
    axes[0].bar(names,[g['arrivals']/g['n']*100 for g in groups],color=colors)
    for i,g in enumerate(groups): axes[0].text(i,g['arrivals']/g['n']*100+1,f"{g['arrivals']}/{g['n']}",ha='center')
    axes[0].set(ylabel='Arrivals (%)',ylim=(0,60),title='All 378 scheduled trials')
    for j,stream in enumerate(['form','colour','uv']):
        axes[1].scatter([j]*3,[f['streams']['raw_'+stream]['median']*100 for f in result['features']],s=60,label=stream)
    axes[1].set(xticks=range(3),xticklabels=['Form','Colour','UV'],ylabel='Energy in top 5 pooled values (%)',title='ApiaViz: medians within each world',ylim=(0,105))
    for i,world in enumerate(p['worlds']):
        d=aligned[world['name'],'apiaviz_uv']; dec=d['decisions'][:12]
        axes[2].plot([v['step'] for v in dec],[v.get('familiarity',np.nan) for v in dec],label=world['name'])
    axes[2].set(xlabel='Movement iteration',ylabel='Familiarity',title='Aligned ApiaViz, seed 19, phase +1');axes[2].legend()
    fig.savefig(out/'diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('groups','features','repeatability')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/uv-trials-v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.study,args.output)
