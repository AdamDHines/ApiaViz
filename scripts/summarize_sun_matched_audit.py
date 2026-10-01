"""Verify and present the cached sun / same-pose diagnostic, without new trials."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import atomic_json
from audit_sun_and_matched_views import sun_figure


def run(source,out):
    out.mkdir(parents=True,exist_ok=False)
    d=json.loads((source/'audit.json').read_text())
    first=json.loads((source/'first-divergence.json').read_text())
    for data in (d,first):
        for name,digest in data['inputs'].items():
            assert file_sha(Path(name))==digest,name
    unique={}
    for r in d['records']:
        key=(r['world'],tuple(np.round(r['position'],8)),r['nearest_station'])
        if key in unique:
            assert unique[key]['streams']==r['streams']
        unique[key]=r
    summary=[]
    for near in (True,False):
        rows=[r for r in unique.values() if (r['polyline_m']<=.05)==near]
        for name in d['controls']:
            summary.append(dict(near=near,method=name,n=len(rows),
                within20=sum(abs(r['streams'][name]['best_offset_deg'])<=20 for r in rows),
                median_margin=float(np.median([r['streams'][name]['tangent_margin'] for r in rows]))))
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['uv10k','bee_visible8k','linear_rgb8k','sobel_rgb8k','sobel_rgb10k']
    labels=['Apia + UV\n10k','Apia bee B/G\n8k','Apia visible RGB\n8k','Sobel visible RGB\n8k','Sobel visible RGB\n10k']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    rows=[r for r in summary if not r['near'] and r['method'] in names]
    rows=sorted(rows,key=lambda r:names.index(r['method']))
    axes[0].bar(range(5),[r['within20']/r['n'] for r in rows],color=['#168e88']*3+['#cc8735']*2)
    for i,r in enumerate(rows):axes[0].text(i,r['within20']/r['n']+.015,f"{r['within20']}/{r['n']}",ha='center')
    axes[0].set(xticks=range(5),xticklabels=labels,ylim=(0,1),ylabel='Within 20° of nearest taught tangent',title='Identical off-route positions · retrospective retrieval')
    r=next(r for r in first['records'] if r['world']=='bend' and r['phase']==-1)
    method=['apiaviz_uv','linear_rgb8k','sobel_rgb8k','sobel_colour']
    changes=[r['metrics'][m]['all']['change'] for m in method]
    axes[1].bar(range(4),changes,color=['#168e88']*2+['#cc8735']*2)
    axes[1].axhline(0,color='black',lw=.8)
    axes[1].set(xticks=range(4),xticklabels=[labels[i] for i in [0,2,3,4]],ylabel='Change in familiarity, same two images and gaze',title='Bend −1, decision 17 · opposite temporal evidence')
    fig.savefig(out/'matched-diagnosis.png',dpi=150);plt.close(fig)
    env=ROOT/'apiaviz/output/uv-acquisition-v2-check1/meander-64-20261002'
    cfg=json.loads((env/'render.json').read_text());cam=DualCamera(env,cfg)
    p=json.loads((ROOT/'apiaviz/output/coherent-validation-v3/protocol.json').read_text())
    w=next(w for w in p['worlds'] if w['name']=='meander')
    sun_figure(out,cam.frame(w['route'][2]),cfg,w['headings'][2],d['sun'][0])
    atomic_json(out/'summary.json',dict(source_audit_sha256=file_sha(source/'audit.json'),
        source_archive_sha256=file_sha(source/'source.zip'),first_divergence_sha256=file_sha(source/'first-divergence.json'),
        analysis_source_sha256={str(Path(__file__).relative_to(ROOT)):file_sha(Path(__file__)),
            'scripts/audit_sun_and_matched_views.py':file_sha(ROOT/'scripts/audit_sun_and_matched_views.py')},
        records=len(d['records']),unique_positions=len(unique),summary=summary,
        sun=d['sun'],first_divergences=first['records'],controls=d['controls'],
        input_hashes_verified=len(set(d['inputs'])|set(first['inputs'])),
        limitation='Correlated, previously inspected development positions from both models; no new trajectories. Tangent retrieval is not route recovery.',
        figures={name:file_sha(out/name) for name in ['matched-diagnosis.png','sun-sampling.png']}))
    print('Verified',len(unique),'unique positions;',len(set(d['inputs'])|set(first['inputs'])),'input hashes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'apiaviz/output/sun-matched-audit-v2')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
