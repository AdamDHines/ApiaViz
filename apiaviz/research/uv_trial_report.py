"""Complete-case accounting, descriptive paired statistics, figures and movies."""
from collections import Counter
import csv
from functools import lru_cache
import html
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .controller_full_report import paired_effect,retention
from .dual_camera import digest,load_cached,position_key,visible_image
from .route_full import cases,perturbation_analysis
from .spectral_input import file_sha
from .uv_input import sample_receptors

NAMES=dict(apiaviz_uv='ApiaViz + UV',sobel_colour='Sobel + colour',ardin_input='Ardin-style')
COLORS=dict(apiaviz_uv='#168e88',sobel_colour='#cc8735',ardin_input='#7978b9')


def environment_preview(env,world,camera):
    os.environ.setdefault('MPLCONFIGDIR',str(env/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    indices=[0,len(world['route'])//2,len(world['route'])-1]
    fig,axes=plt.subplots(3,3,figsize=(12,7),layout='constrained')
    geometry=json.loads((env/'collision.json').read_text())
    route=np.array(world['route'])
    for row,index in enumerate(tqdm(indices,desc='UV / visible preview poses',leave=False)):
        position=world['route'][index]; heading=world['headings'][index]
        uv=camera.scan(position,[heading],uv=True)[0].permute(1,2,0).numpy()[:,::-1]
        rgb=camera.scan(position,[heading])[0].permute(1,2,0).numpy()[:,::-1]
        axes[row,0].imshow(uv/(uv+.2),aspect='auto'); axes[row,0].axis('off')
        axes[row,0].set_title(f'Station {index}: UV/B/G false colours',fontsize=10)
        axes[row,1].imshow(rgb,aspect='auto'); axes[row,1].axis('off')
        axes[row,1].set_title('Visible-only camera response',fontsize=10)
        ax=axes[row,2]
        for rock in geometry['rocks']: ax.add_collection(PolyCollection(rock['triangles_xy_m'],facecolor='#969b91',edgecolor='none'))
        ax.plot(*route.T,':',color='#397f9a'); ax.scatter(*position,color='#d38424')
        ax.set(xlim=(world['world_bounds_m'][0],world['world_bounds_m'][1]),ylim=(world['world_bounds_m'][2],world['world_bounds_m'][3]),aspect='equal')
    fig.suptitle(world['name']+' · fresh Blender geometry, calibrated spectral materials · geometry is evaluator-only')
    fig.savefig(env/'preview.png',dpi=150); plt.close(fig)


def statistics(rows,p):
    groups=[]
    for world in ['all']+[w['name'] for w in p['worlds']]:
        for method in p['methods']:
            selected=[r for r in rows if r['method']==method and (world=='all' or r['world']==world)]
            if not selected: continue
            def median(key):
                values=[r[key] for r in selected if r.get(key) is not None]
                return float(np.median(values)) if values else None
            groups.append(dict(world=world,method=method,n=len(selected),arrivals=sum(r['reached_nest'] for r in selected),
                recovery_applicable=sum(r['recovery_applicable'] for r in selected),recovered=sum(r['recovered'] for r in selected),
                blocked_trials=sum(r['blocked_proposals']>0 for r in selected),blocked_proposals=sum(r['blocked_proposals'] for r in selected),
                adjusted_perturbations=sum(r['perturbation_adjusted'] for r in selected),
                median_path_m=median('path_length_m'),median_time_s=median('time_s'),median_views=median('observations'),
                median_deviation_m=median('polyline_mean_m'),median_encoding_s=median('encoding_s'),
                median_wall_s=median('wall_time_s'),median_active_spikes=median('active_spikes'),
                terminations=dict(Counter(r['termination'] for r in selected))))
    comparisons=[paired_effect(rows,'reached_nest',('familiarity','apiaviz_uv'),('familiarity',m)) for m in p['methods'][1:]]
    return dict(planned=p['trials'],completed=len(rows),complete=len(rows)==p['trials'],groups=groups,
        paired_arrival_effects=comparisons,perturbations=perturbation_analysis(rows,p,primary='apiaviz_uv'),
        comparison=p['comparison'],limitation=p['limitation'],
        denominator='All completed scheduled trials, including all failures. Pending trials are listed separately; a partial report is not a final result.',
        cost_note='Wall time includes cache misses/rendering and depends on method order. Encoding time, camera counts and spikes are reported separately; equal KC count does not imply equal compute/spike cost.')


def movie_selection(rows,p):
    order={c[0]:i for i,c in enumerate(cases(p))}
    ordered=sorted(rows,key=lambda r:order[r['id']])
    selected={r['id'] for r in ordered if r['seed']==p['movies']['seed'] and r['phase']==p['movies']['phase'] and r['scenario'] in p['movies']['scenarios']}
    seen=set()
    for row in ordered:
        key=(row['world'],row['method'])
        if not row['reached_nest'] and key not in seen:
            selected.add(row['id']); seen.add(key)
    return [r for r in ordered if r['id'] in selected]


def path_with_kicks(detail,start):
    timeline=[(m['time_s'],0,m['position']) for m in detail['microtrace']]
    timeline += [(e['time_s'],1,e['position']) for e in detail['events'] if e['kind']=='displacement']
    points=[start]; times=[0.]
    for t,kind,position in sorted(timeline):
        if kind: points.append([np.nan,np.nan]); times.append(t)
        points.append(position); times.append(t)
    return np.asarray(points),np.asarray(times)


def make_movie(out,p,row):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon
    directory=out/'report/movies'; directory.mkdir(exist_ok=True)
    target=directory/f'{row["id"]}.mp4'; sidecar=target.with_suffix('.json')
    trace_hash=file_sha(out/row['trace'])
    if target.exists() and sidecar.exists():
        saved=json.loads(sidecar.read_text())
        if saved['trace_sha256']==trace_hash and saved['video_sha256']==file_sha(target): return target
        raise ValueError('Existing movie or source trace changed')
    detail=json.loads((out/row['trace']).read_text())
    world=next(w for w in p['worlds'] if w['name']==row['world'])
    env=out/'worlds'/row['world']; render=json.loads((env/'render.json').read_text())
    geometry=json.loads((env/'collision.json').read_text())
    views=[e for e in detail['events'] if e['kind'] in ('observation','avoidance_observation')]
    viewtimes=np.array([v['time_s'] for v in views]); path,times=path_with_kicks(detail,row['initial_position'])
    memories=[e for e in views if e['kind']=='observation']
    fig=plt.figure(figsize=(12,7),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[.8,1.5]); uv=fig.add_subplot(grid[0,0]); rgb=fig.add_subplot(grid[0,1])
    overhead=fig.add_subplot(grid[1,0]); signal=fig.add_subplot(grid[1,1])
    fig.suptitle(f'{NAMES[row["method"]]} · {row["world"]} · {row["scenario"]} · seed {row["seed"]}, phase {row["phase"]}')
    uv_picture=uv.imshow(np.zeros((51,199,3)),aspect='auto'); uv.axis('off')
    uv.set_title('UV / blue / green · false-colour display only',fontsize=10)
    rgb_picture=rgb.imshow(np.zeros((51,199,3)),aspect='auto'); rgb.axis('off')
    rgb.set_title('Visible camera · shared avoidance; Sobel/Ardin input',fontsize=10)
    for rock in geometry['rocks']:
        overhead.add_collection(PolyCollection(rock['triangles_xy_m'],facecolor='#8f968e',edgecolor='none'))
    route=np.array(world['route']); overhead.plot(*route.T,':',color='#55768b',label='Taught route')
    overhead.scatter(*route[-1],marker='*',s=100,color='#d09023')
    line,=overhead.plot([],[],color=COLORS[row['method']],lw=2)
    ant=Polygon([[0,0],[0,0],[0,0]],color='#df8223'); overhead.add_patch(ant)
    finite=path[np.isfinite(path).all(1)]
    lo=np.minimum(finite.min(0),route.min(0))-.5; hi=np.maximum(finite.max(0),route.max(0))+.5
    overhead.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Evaluator trace · never shown to policies')
    if memories:
        signal.plot([v['time_s'] for v in memories],[v['familiarity'] for v in memories],color=COLORS[row['method']])
    signal.set(xlim=(0,max(1,row['time_s'])),ylim=(-.01,1.05),xlabel='Simulation time (s)',ylabel='Visual familiarity')
    cursor=signal.axvline(0,color='#d09023'); status=signal.text(.03,.1,'',transform=signal.transAxes,fontsize=9)
    @lru_cache(maxsize=64)
    def image(key,heading):
        raw,_=load_cached(env/'camera',key,digest(render))
        spectral=sample_receptors(raw[:,:,:3],[heading],elevation=render['elevation_deg'])[0].permute(1,2,0).numpy()[:,::-1]
        visible=sample_receptors(visible_image(raw[:,:,3:],render['visible_white']),[heading],elevation=render['elevation_deg'])[0].permute(1,2,0).numpy()[:,::-1]
        return spectral/(spectral+.2),visible
    fps=p['movies']['fps']; duration=min(p['movies']['max_seconds'],max(1.,row['time_s']/8))
    frames=np.linspace(0,row['time_s'],max(2,int(duration*fps)))
    temporary=target.with_suffix('.tmp.mp4')
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=1800,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    try:
        with writer.saving(fig,str(temporary),dpi=100):
            for t in tqdm(frames,desc='Movie frames',leave=False,unit='frame'):
                vi=int(np.searchsorted(viewtimes,t,side='right')-1)
                if vi>=0:
                    view=views[vi]; a,b=image(position_key(view['position']),view['heading']); uv_picture.set_data(a); rgb_picture.set_data(b)
                index=int(np.searchsorted(times,t,side='right')); shown=path[:index]
                line.set_data(shown[:,0],shown[:,1]); pos=shown[np.isfinite(shown).all(1)][-1]
                angle=np.deg2rad(views[vi]['heading'] if vi>=0 else world['headings'][0])
                rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                ant.set_xy(np.array([[.07,0],[-.04,.04],[-.04,-.04]])@rotation.T+pos)
                cursor.set_xdata([t,t]); blocked=sum(e['kind']=='blocked_proposal' and e['time_s']<=t for e in detail['events'])
                status.set_text(f'{t:.1f} / {row["time_s"]:.1f} s\nBlocked commands: {blocked}\n'
                    f'Outcome: {row["termination"] if t==frames[-1] else "running"}\n'
                    f'Adjusted perturbation: {row["perturbation_adjusted"]}\nAnt marker enlarged; views held between samples')
                writer.grab_frame()
            for _ in range(fps): writer.grab_frame()
        temporary.replace(target)
    finally: plt.close(fig)
    sidecar.write_text(json.dumps(dict(trace_sha256=trace_hash,video_sha256=file_sha(target),fps=fps,
        frames=len(frames)+fps,simulated_seconds=row['time_s'],movie_seconds=(len(frames)+fps)/fps,
        selection=p['movies']['selection']),indent=2)+'\n')
    return target


def report(out,p,rows,*,movies=True):
    from .uv_trials import atomic_json
    dest=out/'report'; dest.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(dest/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    for row in rows:
        detail=json.loads((out/row['trace']).read_text())
        kept=retention(detail['trace'],row['scenario'],p['evaluation']['kick_before_step'])
        if not row['recovery_applicable']: kept=dict(return_found=False,lost_after_return=False)
        assert kept['return_found']==row['recovered']
        row.update(kept)
    summary=statistics(rows,p)
    summary['camera_cache']={w['name']:dict(frames=len(list((out/'worlds'/w['name']/'camera').glob('*.npz'))),
        bytes=sum(f.stat().st_size for f in (out/'worlds'/w['name']/'camera').glob('*.npz'))) for w in p['worlds']}
    summary['movies_requested']=movies
    atomic_json(dest/'summary.json',summary)
    ordered={c[0]:i for i,c in enumerate(cases(p))}; rows=sorted(rows,key=lambda r:ordered[r['id']])
    fields=sorted({k for r in rows for k in r})
    with (dest/'trials.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    atomic_json(dest/'trials.json',rows)
    atomic_json(dest/'pending.json',[key for key in ordered if key not in {r['id'] for r in rows}])
    pictures=[]
    with PdfPages(dest/'figures.pdf') as pdf,tqdm(total=4,desc='Summary figures',unit='figure') as progress:
        def save(fig,name):
            fig.savefig(dest/f'{name}.png',dpi=150); fig.savefig(dest/f'{name}.svg'); pdf.savefig(fig)
            pictures.append(name); plt.close(fig); progress.update(1)
        fig,axes=plt.subplots(len(p['worlds']),1,figsize=(11,3*len(p['worlds'])),squeeze=False,layout='constrained')
        for ax,world in zip(axes[:,0],p['worlds']):
            values=np.zeros((3,len(p['scenarios'])))
            for i,method in enumerate(p['methods']):
                for j,scenario in enumerate(p['scenarios']):
                    selected=[r for r in rows if (r['world'],r['method'],r['scenario'])==(world['name'],method,scenario['name'])]
                    hits=sum(r['reached_nest'] for r in selected); values[i,j]=hits/len(selected) if selected else 0
                    ax.text(j,i,f'{hits}/{len(selected)}',ha='center',va='center')
            ax.imshow(values,vmin=0,vmax=1,cmap='Blues',alpha=.6,aspect='auto')
            ax.set(title=world['name'],xticks=range(len(p['scenarios'])),xticklabels=[s['name'] for s in p['scenarios']],yticks=range(3),yticklabels=[NAMES[m] for m in p['methods']])
        fig.suptitle(f'Arrivals · {len(rows)}/{p["trials"]} scheduled trials complete'); save(fig,'arrivals')
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        outcomes=sorted({r['termination'] for r in rows})
        for ax,method in zip(axes,p['methods']):
            selected=[r for r in rows if r['method']==method]; counts=Counter(r['termination'] for r in selected)
            ax.barh(outcomes,[counts[o] for o in outcomes],color=COLORS[method]); ax.set(title=NAMES[method],xlabel='Trials (all failures retained)')
        save(fig,'terminations')
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for ax,key,label in zip(axes.flat,('time_s','observations','path_length_m','polyline_mean_m','encoding_s','active_spikes'),
            ('Simulated seconds','Camera observations','Walked metres','Mean route deviation (m)','Encoding seconds','Active KC spikes')):
            for i,method in enumerate(p['methods']):
                values=[r[key] for r in rows if r['method']==method and r.get(key) is not None]
                ax.scatter(np.full(len(values),i),values,s=9,alpha=.25,color=COLORS[method])
                if values: ax.plot([i-.2,i+.2],[np.median(values)]*2,color='black',lw=2)
            ax.set(xticks=range(3),xticklabels=['Apia+UV','Sobel','Ardin'],ylabel=label)
        fig.suptitle('All completed trials · early failures affect costs · black line is median'); save(fig,'costs')
        fig,axes=plt.subplots(len(p['worlds']),3,figsize=(12,3.5*len(p['worlds'])),squeeze=False,layout='constrained')
        for i,world in enumerate(p['worlds']):
            route=np.array(world['route'])
            for j,method in enumerate(p['methods']):
                ax=axes[i,j]; ax.plot(*route.T,'k:',lw=1)
                for row in rows:
                    if row['world']!=world['name'] or row['method']!=method: continue
                    detail=json.loads((out/row['trace']).read_text()); path,_=path_with_kicks(detail,row['initial_position'])
                    ax.plot(*path.T,color=COLORS[method],alpha=.18,lw=.6)
                ax.set(title=world['name']+' · '+NAMES[method],aspect='equal',xlabel='x (m)',ylabel='y (m)')
        fig.suptitle('Every completed trajectory · imposed kicks break the walking line'); save(fig,'trajectories')
    videos=[]
    if movies:
        for row in tqdm(movie_selection(rows,p),desc='Summary movies',unit='movie'):
            target=make_movie(out,p,row); videos.append((row,str(target.relative_to(dest))))
    summary['movies']=[dict(id=r['id'],path=path) for r,path in videos]; atomic_json(dest/'summary.json',summary)
    def esc(value): return html.escape(str(value))
    intro=f'<h1>UV / visible navigation results</h1><p>{len(rows)} of {p["trials"]} scheduled trials completed. '+('Complete run.' if summary['complete'] else '<b>Partial report — pending trials remain.</b>')+'</p>'
    body=[intro,f'<p>{esc(p["comparison"])}</p>',f'<p>{esc(p["limitation"])}</p>',f'<p>{esc(summary["cost_note"])}</p>',
        '<p><a href="summary.json">Statistics + dose strata</a> · <a href="trials.csv">Trial CSV</a> · <a href="trials.json">Trial JSON</a> · <a href="figures.pdf">Figures PDF</a> · <a href="../protocol.json">Frozen protocol</a></p>',
        '<h2>Outcomes and costs</h2><table><tr><th>Model</th><th>Arrivals</th><th>Recovered / applicable</th><th>Blocked trials</th><th>Adjusted dose</th><th>Median seconds / views</th></tr>']
    for g in summary['groups']:
        if g['world']=='all': body.append(f'<tr><td>{NAMES[g["method"]]}</td><td>{g["arrivals"]}/{g["n"]}</td><td>{g["recovered"]}/{g["recovery_applicable"]}</td><td>{g["blocked_trials"]}</td><td>{g["adjusted_perturbations"]}</td><td>{g["median_time_s"]:.1f} / {g["median_views"]:.0f}</td></tr>')
    body+=['</table><h2>Paired descriptive arrival differences</h2><p>Average phases, then paired seed/scenario differences within each world. Intervals resample worlds, not individual ants. Three worlds cannot support a strong population significance claim.</p><ul>']
    for c in summary['paired_arrival_effects']:
        if c['mean'] is not None: body.append(f'<li>ApiaViz + UV minus {NAMES[c["right"][1]]}: {100*c["mean"]:+.1f} percentage points; descriptive 95% world-bootstrap interval [{100*c["low"]:+.1f}, {100*c["high"]:+.1f}]; {c["paired_cases"]} paired cases, {c["worlds"]} worlds.</li>')
    body+=['</ul><p>Shortened/skipped/missed perturbations remain in primary results. See dose strata and the trajectory-dependent, whole-block full-dose sensitivity analysis in summary.json.</p>']
    body += [f'<h2>{name.replace("_"," ").title()}</h2><img src="{name}.png" alt="{name}">' for name in pictures]
    body += ['<h2>Movies</h2><p>Prespecified first-seed/positive-phase aligned and kick-right cases, plus the first scheduled failure for each world/model. All outcomes retained; playback is accelerated. Spectral false colours are display-only.</p>']
    for row,path in videos: body.append(f'<details><summary>{esc(row["id"])} — {esc(row["termination"])}</summary><video controls preload="none" src="{esc(path)}"></video></details>')
    body += ['<h2>Every completed trial</h2><table><tr><th>Trial / trace</th><th>Outcome</th><th>Time</th><th>Walked</th><th>Blocked</th><th>Adjusted</th></tr>']
    for r in rows: body.append(f'<tr><td><a href="../{esc(r["trace"])}">{esc(r["id"])}</a></td><td>{esc(r["termination"])}</td><td>{r["time_s"]:.1f}</td><td>{r["path_length_m"]:.2f}</td><td>{r["blocked_proposals"]}</td><td>{r["perturbation_adjusted"]}</td></tr>')
    body+=['</table>']
    (dest/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>ApiaViz UV trials</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172b39}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img,video{max-width:100%}details{margin:12px 0}a{color:#087a82}</style></head><body>'+''.join(body)+'</body></html>')
    atomic_json(dest/'artifacts.json',{str(f.relative_to(dest)):file_sha(f) for f in dest.rglob('*') if f.is_file() and '.mpl-cache' not in f.parts and f.name!='artifacts.json'})
