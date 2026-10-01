"""Audit all nine graded navigation trials, including every movie and checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research import uv_trials as base
from apiaviz.research.spectral_input import file_sha


def run(study,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise FileExistsError('Preserve completed summary')
    torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    rows=list(base.load_completed(study,p).values())
    assert len(rows)==p['trials']==9
    audit=json.loads((study/'audit.json').read_text())
    assert audit['passed'] and audit['trials']==9 and audit['protocol_sha256']==file_sha(study/'protocol.json')
    with zipfile.ZipFile(study/'source.zip') as archive:
        for name,sha in p['source_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
    parent=Path(p['parent_study']['path']);parent_p=json.loads((parent/'protocol.json').read_text())
    assert file_sha(parent/'protocol.json')==p['parent_study']['protocol_sha256']
    for key in ('sensor_settings','evaluation','evaluation_safety','avoidance_settings','render'):
        assert parent_p[key]==p[key]
    checks=[];details={}
    for world in p['worlds']:
        env=study/'worlds'/world['name'];base.verify_environment(env)
        assert file_sha(env/'teaching.pt')==file_sha(parent/'worlds'/world['name']/'teaching.pt')
        meta=json.loads((env/'world.json').read_text())
        geometry=base.RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
        bank=torch.load(env/'teaching.pt',weights_only=True,map_location='cpu')
        for row in [r for r in rows if r['world']==world['name']]:
            detail=json.loads((study/row['trace']).read_text());details[row['id']]=detail
            assert file_sha(study/row['trace'])==audit['traces'][row['id']]
            base.audit_trial(row,detail,p,world,geometry)
            checkpoint=torch.load(env/f'encoder-{row["method"]}-{row["seed"]}.pt',weights_only=True,map_location='cpu')
            model=base.make_model(p,row['method'],row['seed'])
            images=bank['uv' if row['method']=='apiaviz_uv' else 'rgb']
            codes=base.encode(model,images);memory,calibration=base.make_memory(model,codes)
            assert base.fingerprint(model)==row['encoder_fingerprint']==checkpoint['encoder_fingerprint']
            assert base.fingerprint(memory)==row['memory_fingerprint']==checkpoint['memory_fingerprint']
            assert calibration==checkpoint['calibration']
            for name,part in [('encoder',model),('memory',memory)]:
                for key,value in part.state_dict().items():assert torch.equal(value,checkpoint[name][key])
            if row['method']!='apiaviz_uv':
                old=torch.load(parent/'worlds'/world['name']/f'encoder-{row["method"]}-{row["seed"]}.pt',weights_only=True,map_location='cpu')
                for key in ('encoder_fingerprint','memory_fingerprint','training_images_sha256','calibration'):
                    assert checkpoint[key]==old[key],(row['id'],key)
            else:
                assert row['active_spikes'] is None and row['active_response_components']>0
                assert torch.unique(memory.memory).numel()>100
            blocked=[e for e in detail['events'] if e['kind']=='blocked_proposal']
            longest=streak=0
            for micro in detail['microtrace']:
                streak=streak+1 if micro.get('blocked') else 0;longest=max(longest,streak)
            positions=[row['initial_position']]+[v['position'] for v in detail['microtrace']]
            closest=min(float(np.linalg.norm(np.asarray(v)-world['route'][-1])) for v in positions)
            turns=sum(e['duration_s'] for e in detail['events'] if e['kind']=='turn')
            checks.append(dict(id=row['id'],closest_nest_distance_m=closest,
                blocked_by_reason=dict(Counter(e['reason'] for e in blocked)),longest_blocked_sequence=longest,
                turning_s=turns,observation_s=row['observations']*p['sensor_settings']['observation_s'],
                commanded_translation_s=row['commanded_path_m']/p['sensor_settings']['speed_m_s'],
                scans=sum('scan' in v for v in detail['decisions']),
                casts=sum(v.get('state')=='cast' for v in detail['decisions']),calibration=calibration,
                checkpoint_reconstructed=True,baseline_identical_to_parent=row['method']!='apiaviz_uv'))
    videos=[]
    movies=list((study/'report/movies').glob('*.mp4'))
    assert {v.stem for v in movies}=={r['id'] for r in rows}
    for movie in sorted(movies):
        record=json.loads(movie.with_suffix('.json').read_text())
        assert record['video_sha256']==file_sha(movie)
        assert record['trace_sha256']==file_sha(study/'trials'/f'{movie.stem}.json')
        subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True,capture_output=True)
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=width,height,nb_read_frames','-of','json',str(movie)]))['streams'][0]
        assert int(info['nb_read_frames'])==record['frames']
        videos.append(dict(path=str(movie.relative_to(ROOT)),sha256=file_sha(movie),decoded=True,**info))
    groups=[]
    for method in p['methods']:
        selected=[r for r in rows if r['method']==method];ids={r['id'] for r in selected}
        reasons=Counter()
        for item in checks:
            if item['id'] in ids:reasons.update(item['blocked_by_reason'])
        groups.append(dict(method=method,n=3,arrivals=sum(r['reached_nest'] for r in selected),
            terminations=dict(Counter(r['termination'] for r in selected)),blocked_by_reason=dict(reasons),
            median_time_s=float(np.median([r['time_s'] for r in selected])),
            median_encoding_s=float(np.median([r['encoding_s'] for r in selected]))))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-graded-nav-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from apiaviz.research.uv_trial_report import NAMES,COLORS,path_with_kicks
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    for i,world in enumerate(p['worlds']):
        geometry=json.loads((study/'worlds'/world['name']/'collision.json').read_text())
        route=np.asarray(world['route'])
        world_rows=[r for r in rows if r['world']==world['name']]
        paths=[path_with_kicks(details[r['id']],r['initial_position'])[0] for r in world_rows]
        all_points=np.concatenate([route,*paths]);all_points=all_points[np.isfinite(all_points).all(1)]
        lo=all_points.min(0)-.4;hi=all_points.max(0)+.4
        for j,method in enumerate(p['methods']):
            ax=axes[i,j];row=next(r for r in world_rows if r['method']==method)
            for rock in geometry['rocks']:ax.add_collection(PolyCollection(rock['triangles_xy_m'],facecolor='#b4b4a7',edgecolor='none'))
            ax.plot(*route.T,'k:',lw=1,label='Taught route');ax.scatter(*route[-1],marker='*',s=70,color='#dc9922')
            path,_=path_with_kicks(details[row['id']],row['initial_position'])
            ax.plot(*path.T,color=COLORS[method],lw=1.5);ax.scatter(*path[-1],s=16,color=COLORS[method])
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='x (m)',ylabel='y (m)',
                title=f'{world["name"]} · {NAMES[method]}\n{row["termination"]} · {row["time_s"]:.1f} s')
    fig.suptitle('All nine graded-navigation smoke trials · matched axes within each world')
    fig.savefig(out/'trajectories.png',dpi=160);plt.close(fig)
    for name in ('arrivals.png','costs.png','terminations.png'):
        shutil.copyfile(study/'report'/name,out/name)
    base.atomic_json(out/'results.json',dict(protocol_sha256=file_sha(study/'protocol.json'),
        source_archive_sha256=file_sha(study/'source.zip'),script_sha256=file_sha(Path(__file__)),
        groups=groups,rows=rows,diagnostics=checks,movies=videos,audit=dict(passed=True,trials=9,movies=9),
        readiness=json.loads((study/'readiness.json').read_text()),limitation=p['limitation']))
    print(json.dumps(groups,indent=2));print('Verified nine checkpoints, traces and decoded movies')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/graded-navigation-smoke-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/graded-navigation-smoke-v1')
    args=parser.parse_args();run(args.study.resolve(),args.output.resolve())
