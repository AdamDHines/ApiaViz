"""Verify completed fidelity validation and write a compact research record."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trial_report import frame_display,movie_selection
from apiaviz.research.spectral_input import file_sha


def run(study,prototype,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve existing results')
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    rows=list(base.load_completed(study,p).values());assert len(rows)==18
    audit=json.loads((study/'audit.json').read_text());assert audit['passed'] and audit['trials']==18
    assert audit['protocol_sha256']==file_sha(study/'protocol.json')
    banks=[]
    for w in p['worlds']:
        env=study/'worlds'/w['name'];base.verify_environment(env)
        bank=torch.load(env/'teaching.pt',weights_only=True)
        cfg=json.loads((env/'teaching-camera/render.json').read_text());camera=DualCamera(env/'teaching-camera',cfg)
        for i in (0,len(w['route'])//2):
            for key in ('uv','rgb'):
                view=camera.scan(w['route'][i],[w['headings'][i]],uv=key=='uv')[0]
                torch.testing.assert_close(view,bank[key][i],rtol=0,atol=0)
        banks.append(dict(world=w['name'],teaching_sha256=file_sha(env/'teaching.pt'),
            reconstruction='both spectral and visible arrays exactly reconstructed at first and middle stations',
            geometry=json.loads((env/'geometry-equivalence.json').read_text())))
    contact=[]
    for row in rows:
        path=study/row['trace'];assert audit['traces'][row['id']]==file_sha(path)
        d=json.loads(path.read_text());w=next(w for w in p['worlds'] if w['name']==row['world'])
        env=study/'worlds'/w['name'];meta=json.loads((env/'world.json').read_text())
        geometry=base.RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
        base.audit_trial(row,d,p,w,geometry)
        blocked=[e for e in d['events'] if e['kind']=='blocked_proposal']
        contact.append(dict(id=row['id'],blocked=len(blocked),first_block=blocked[0] if blocked else None,
            blocked_by_reason={reason:sum(e['reason']==reason for e in blocked) for reason in sorted({e['reason'] for e in blocked})},
            trace_sha256=file_sha(path),scans=sum('scan' in d for d in d['decisions']),
            cast_decisions=sum(d.get('state')=='cast' for d in d['decisions']),
            first_cast_step=next((v['step'] for v in d['decisions'] if v.get('state')=='cast'),None),
            first_20cm_error_step=next((v['step'] for v in d['trace'] if v['polyline_m']>.2),None),
            closest_nest_distance_m=min(float(np.linalg.norm(np.asarray(v['position'])-w['route'][-1])) for v in d['microtrace'])))
    movies=[]
    assert {f.stem for f in (study/'report/movies').glob('*.mp4')}=={r['id'] for r in movie_selection(rows,p)}
    for movie in sorted((study/'report/movies').glob('*.mp4')):
        record=json.loads(movie.with_suffix('.json').read_text())
        assert record['video_sha256']==file_sha(movie)
        assert record['trace_sha256']==file_sha(study/'trials'/f'{movie.stem}.json')
        subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True,capture_output=True)
        movies.append(dict(path=str(movie.relative_to(ROOT)),sha256=file_sha(movie),decoded=True))
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=p['render'];fig,axes=plt.subplots(2,2,figsize=(12,5),layout='constrained')
    for i,tag in enumerate(('flat','detailed')):
        uv,rgb=frame_display(np.load(prototype/f'{tag}-0.npy'),0,cfg)
        for ax,image,name in zip(axes[i],(rgb,uv),('Visible','UV / blue / green false colours')):
            ax.imshow(image,aspect='auto');ax.axis('off');ax.set_title(f'{tag}: {name}')
    fig.suptitle('Identical geometry, rays, solar illumination and fixed display transfer · material / normal control')
    fig.savefig(out/'surface-comparison.png',dpi=150);plt.close(fig)
    previous=ROOT/'apiaviz/output/coherent-validation-v3/report/trials.json'
    old=json.loads(previous.read_text())
    old_events={r['id']:json.loads((previous.parents[1]/r['trace']).read_text())['events'] for r in old}
    results=[]
    for method in p['methods']:
        now=[r for r in rows if r['method']==method];before=[r for r in old if r['method']==method]
        results.append(dict(method=method,arrivals=sum(r['reached_nest'] for r in now),n=len(now),
            previous_arrivals=sum(r['reached_nest'] for r in before),
            blocked=sum(r['blocked_proposals'] for r in now),previous_blocked=sum(r['blocked_proposals'] for r in before),
            observations=sum(r['observations'] for r in now),time_s=sum(r['time_s'] for r in now),
            blocked_by_reason={reason:sum(c['blocked_by_reason'].get(reason,0) for c in contact if any(r['id']==c['id'] for r in now))
                               for reason in ('rock_contact','field_boundary')},
            previous_blocked_by_reason={reason:sum(e['kind']=='blocked_proposal' and e.get('reason')==reason for r in before for e in old_events[r['id']])
                                        for reason in ('rock_contact','field_boundary')},
            terminations={k:sum(r['termination']==k for r in now) for k in sorted({r['termination'] for r in now})}))
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');x=np.arange(3)
    ax.bar(x-.18,[r['previous_arrivals'] for r in results],.36,label='Previous v3')
    ax.bar(x+.18,[r['arrivals'] for r in results],.36,label='Combined fidelity changes')
    ax.set(xticks=x,xticklabels=['ApiaViz + UV','Sobel + colour','Ardin-style'],ylim=(0,6.6),ylabel='Arrivals / 6 complete routes',title='Matched aligned development screen · all outcomes retained')
    ax.legend();fig.savefig(out/'arrivals.png',dpi=150);plt.close(fig)
    matched=json.loads((out/'matched-views.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained',sharex=True)
    for method,enabled,label in [('apiaviz_uv',True,'ApiaViz + UV'),('apiaviz_uv',False,'ApiaViz UV stream off'),
            ('sobel_colour',None,'Sobel + colour'),('ardin_input',None,'Ardin-style')]:
        group=[r for r in matched['records'] if r['method']==method and r['uv_enabled']==enabled]
        axes[0].plot([r['step'] for r in group],[r['tangent_error_deg'] for r in group],label=label,
                     linestyle='--' if enabled is False else '-')
        axes[1].plot([r['step'] for r in group],
            [100*(r['distance_after_m']-r['distance_before_m']) for r in group],
            linestyle='--' if enabled is False else '-')
    axes[0].set(ylabel='Tangent error (degrees)',title='Same camera positions on failed ApiaViz meander trajectory')
    axes[0].legend(ncol=2);axes[1].axhline(0,color='black',linewidth=.7)
    axes[1].set(xlabel='Recorded decision step',ylabel='Projected lateral error change (cm)',
                title='10 cm along retrieved heading: negative values point toward route')
    fig.savefig(out/'matched-directions.png',dpi=150);plt.close(fig)
    corner=json.loads((out/'corner-sampling.json').read_text())
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for r in corner['records']:
        label=r['method']+(' (UV off)' if r['uv_enabled'] is False else '')
        ax.plot(r['headings'],r['scores'],label=label,linestyle='--' if r['uv_enabled'] is False else '-')
    ax.axvline(45,color='black',ls=':',label='Taught 45° heading')
    ax.set(xticks=np.arange(-10,71,10),xlabel='Heading (degrees)',ylabel='Best learned-view overlap',
           title='Bend corner: a 10° scan misses the narrow 45° peak')
    ax.grid(axis='x',alpha=.25);ax.legend(fontsize=8)
    fig.savefig(out/'corner-sampling.png',dpi=150);plt.close(fig)
    base.atomic_json(out/'results.json',dict(protocol_sha256=file_sha(study/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),source_archive_sha256=file_sha(study/'source.zip'),
        previous_results_sha256=file_sha(previous),results=results,rows=rows,banks=banks,contacts=contact,movies=movies,
        readiness=json.loads((study/'readiness.json').read_text()),
        surface_validation=json.loads((prototype/'validation.json').read_text()),
        limitation=p['limitation']))
    print(json.dumps(results,indent=2));print('Verified',len(movies),'movies')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--prototype',type=Path,default=ROOT/'apiaviz/output/surface-prototype-v1/check3')
    p.add_argument('--output',type=Path,default=ROOT/'docs/navigation-fidelity-v1')
    a=p.parse_args();run(a.study,a.prototype,a.output)
