"""Freeze the pipeline audit and matched controller results, including failures."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research.uv_trials import atomic_json,verify_sources
from apiaviz.research.spectral_input import file_sha


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    paths={name:ROOT/'apiaviz/output'/name for name in ['uv-pipeline-audit-v3','uv-failed-views-v3',
        'uv-precision-v3','uv-validation-v2','coherent-validation-v3']}
    old=paths['uv-validation-v2'];new=paths['coherent-validation-v3']
    p,q=[json.loads((d/'protocol.json').read_text()) for d in [old,new]]
    verify_sources(p);verify_sources(q)
    for key in ['worlds','encoder','render','controller_settings','avoidance_settings','sensor_settings','evaluation','body_radius_m','evaluation_safety']:
        assert p[key]==q[key],key
    audit=json.loads((new/'audit.json').read_text());assert audit['passed'] and audit['trials']==18
    assert audit['protocol_sha256']==file_sha(new/'protocol.json')
    pairs=[]
    for key,digest in audit['traces'].items():
        path=new/'trials'/f'{key}.json';assert file_sha(path)==digest
        a,b=[json.loads((d/'trials'/f'{key}.json').read_text()) for d in [old,new]]
        for field in ['encoder_fingerprint','memory_fingerprint','training_images_sha256']:
            assert a['result'][field]==b['result'][field],(key,field)
        def compact(d):
            r=d['result'];h=np.array([t['heading'] for t in d['microtrace']]);turn=(np.diff(h)+180)%360-180
            blocks=[e for e in d['events'] if e['kind']=='blocked_proposal']
            return dict(arrived=r['reached_nest'],termination=r['termination'],
                blocked_proposals=r['blocked_proposals'],rock_blocks=sum(e['reason']=='rock_contact' for e in blocks),
                field_blocks=sum(e['reason']=='field_boundary' for e in blocks),
                movement_reversals_over_90_deg=int((abs(turn)>90).sum()),
                views=r['observations'],time_s=r['time_s'],path_m=r['path_length_m'],
                final_nest_distance_m=r['final_nest_distance_m'])
        pairs.append(dict(id=key,world=b['result']['world'],method=b['result']['method'],phase=b['result']['phase'],
                          old=compact(a),new=compact(b),old_sha256=file_sha(old/'trials'/f'{key}.json'),new_sha256=digest))
    movies=[]
    for movie in sorted((new/'report').rglob('*.mp4')):
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(movie),'-f','null','-'],check=True,capture_output=True)
        movies.append(dict(file=str(movie.relative_to(new)),sha256=file_sha(movie)))
    assert movies
    stages=json.loads((paths['uv-pipeline-audit-v3']/'audit.json').read_text())
    views=json.loads((paths['uv-failed-views-v3']/'audit.json').read_text())
    precision=json.loads((paths['uv-precision-v3']/'validation.json').read_text())
    result=dict(source_protocols={str(d):file_sha(d/'protocol.json') for d in [old,new]},
        audit_hashes={str(paths[n]/f):file_sha(paths[n]/f) for n,f in [
            ('uv-pipeline-audit-v3','audit.json'),('uv-failed-views-v3','audit.json'),('uv-precision-v3','validation.json')]},
        pairs=pairs,decoded_movies=movies,all_motion_audits_passed=True,
        flat_fields=precision['flat_fields'],changed_acquisition_spike_bits=sum(r['changed_spike_bits'] for r in precision['acquisitions']),
        acquisition_spike_bits=sum(r['bits'] for r in precision['acquisitions']),
        route_retrieval=[],methods=[])
    for near in [True,False]:
        rows=[r for r in views['records'] if (r['polyline_m']<=.05)==near]
        result['route_retrieval'].append(dict(within_5cm=near,n=len(rows),
            **{name:sum(abs(r['streams'][name]['best_offset_deg'])<=20 for r in rows) for name in ['all','visible','uv']}))
    for m in q['methods']:
        rows=[r for r in pairs if r['method']==m]
        result['methods'].append(dict(method=m,n=len(rows),
            **{variant:dict(arrivals=sum(r[variant]['arrived'] for r in rows),
                blocked_proposals=sum(r[variant]['blocked_proposals'] for r in rows),
                rock_blocks=sum(r[variant]['rock_blocks'] for r in rows),
                movement_reversals_over_90_deg=sum(r[variant]['movement_reversals_over_90_deg'] for r in rows))
                for variant in ['old','new']}))
    atomic_json(out/'results.json',result)
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(14,4.5))
    x=np.arange(3);labels=['ApiaViz + UV','Sobel + colour','Ardin-style']
    for variant,offset,label,col in [('old',-.18,'v2 controller','#888888'),('new',.18,'v3 controller','#3978aa')]:
        vals=[r[variant]['arrivals'] for r in result['methods']]
        bars=ax[0].bar(x+offset,vals,.35,label=label,color=col);ax[0].bar_label(bars)
        ax[1].bar(x+offset,[r[variant]['rock_blocks'] for r in result['methods']],.35,color=col)
    ax[0].set(ylim=(0,7),ylabel='Arrivals / 6 aligned trials',title='Same encoding; changed controller')
    ax[0].legend(fontsize=8);ax[1].set(ylabel='Rejected rock-contact proposals',title='Blocked movements remain safe')
    for a in ax[:2]:a.set_xticks(x,labels,rotation=12)
    rows=[r for r in stages['records'] if r['method']=='apiaviz_uv' and r['spp']==64]
    values=[[r['streams'][s]['tangent_margin'] for r in rows] for s in ['all','visible','uv']]
    ax[2].boxplot(values,tick_labels=['Combined','Visible only','UV only'],showfliers=True)
    ax[2].axhline(0,color='.5',linewidth=.8)
    ax[2].set(ylabel='Correct-heading familiarity advantage',title='Independent images at 9 taught stations')
    fig.suptitle('Development diagnostics on three inspected worlds; not a UV-benefit trial',fontsize=12)
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=150)
    print(json.dumps(result['methods'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
