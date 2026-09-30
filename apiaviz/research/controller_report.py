"""Audit and illustrate saved controller trials; never rerun or tune a policy."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .grassland_smoke import position_key, sample_panorama
from .study import file_hash, write_json

NAMES = dict(linear_colour='ApiaViz',sobel_colour='Sobel + colour',ardin_input='Ardin-style')
COLOURS = dict(linear_colour='#007f86',sobel_colour='#d68627',ardin_input='#6572b8')


def read(out):
    protocol=json.loads((out/'protocol.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    return protocol,rows


def audit(out):
    p,rows=read(out)
    assert len(rows)==p['trials'] and len({r['id'] for r in rows})==len(rows)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert file_hash(out/'protocol.json')==manifest['protocol_sha256']
    for source,digest in manifest['source_sha256'].items():
        assert file_hash(Path(source))==digest, source
    previous=[json.loads(s) for s in Path('apiaviz/output/navigation-regime/stage3/results.jsonl').read_text().splitlines()]
    previous={r['id']:r for r in previous}
    comparisons=observations=baseline_matches=0
    hashes={}
    panoramas={}
    for row in rows:
        saved=json.loads((out/row['trace']).read_text())
        hashes[row['trace']]=file_hash(out/row['trace'])
        observed=[e for e in saved['events'] if e['kind']=='observation']
        turned=[e for e in saved['events'] if e['kind']=='turn']
        assert len(observed)==row['observations']==len(saved['sensor'])
        assert len(saved['trajectory'])==row['steps']
        assert np.isclose(sum(abs(e['angle_deg']) for e in turned),row['rotation_deg'])
        expected_time=(sum(e['duration_s'] for e in turned)+len(observed)*p['sensor_settings']['observation_s']
                       +row['steps']*.1/p['sensor_settings']['speed_m_s'])
        assert np.isclose(expected_time,row['time_s'])
        assert row['time_s']<=p['sensor_settings']['time_budget_s']+1e-8
        assert row['observations']<=p['sensor_settings']['observation_budget']
        assert np.isclose(row['path_length_m'],.1*row['steps'])
        for event,query in zip(observed,saved['sensor']):
            np.testing.assert_allclose(event['position'],query['position'],rtol=0,atol=0)
            assert abs((event['heading']-query['headings'][0]+180)%360-180)<1e-8
            expected=-query['scores'][0] if query['scores'][0] is not None else 0.
            assert event['familiarity']==expected
            environment=Path(next(w['environment'] for w in p['worlds'] if w['name']==row['world']))
            path=environment/'panoramas'/f"{position_key(event['position'])}.png"
            if str(path) not in panoramas:
                panoramas[str(path)]=file_hash(path)
        observations+=len(observed)
        decisions=saved['decisions']
        by_step={d['step']:d for d in decisions}
        for d in decisions:
            if 'comparison' not in d: continue
            c=d['comparison']; before=by_step[d['step']-1]
            assert c['before']==before['familiarity']
            assert c['reference_heading']==before['reference_heading']
            assert c['movement_heading']==before['movement_heading']
            assert np.isclose(c['distance_m'],.1)
            within=[e for e in observed if d['time_s'] < e['time_s'] <= d.get('time_end_s',row['time_s'])+1e-8]
            assert within and c['after']==within[0]['familiarity']
            assert abs((within[0]['heading']-c['reference_heading']+180)%360-180)<1e-8
            calibration=json.loads((out/f"{row['world']}-{row['seed']}-{row['method']}-calibration.json").read_text())['familiarity']
            assert np.isclose(c['normalized_change'],(c['after']-c['before'])/calibration['scale'])
            comparisons+=1
        if row['controller'] in ('exhaustive','active'):
            key=f"{row['world']}-{row['seed']}-{row['method']}-{row['scenario']}-{row['controller']}"
            if key in previous:
                old=previous[key]
                for name in ('termination','steps','observations','memory_fingerprint','encoder_fingerprint'):
                    assert row[name]==old[name], (key,name)
                archived=json.loads((Path('apiaviz/output/navigation-regime/stage3')/old['trace']).read_text())
                assert saved['trajectory']==archived['trajectory']
                assert saved['sensor']==archived['decisions']['sensor']
                assert saved['events']==archived['decisions']['events']
                baseline_matches+=1
    # Independently rebuild memories and re-encode first/middle/last queries.
    from .controller_experiments import deterministic
    from .grassland_resolution import ResolutionWorld, load_model
    from .openloop import acquisition_views
    from .navigation import Scorer
    from .spike_overlap import SpikeOverlapMemory
    from .study import encode, fingerprint
    deterministic()
    checked=0
    for spec in p['worlds']:
        environment=Path(spec['environment'])
        base=json.loads((environment/'protocol.json').read_text())
        positions,headings=acquisition_views(np.array(base['route']),np.array(base['headings']),1,0.)
        world=ResolutionWorld(environment,p['shape'],cache_only=True)
        training=world.render(positions,headings)
        for seed in p['seeds']:
            for method in p['methods']:
                model=load_model(Path(p['encoder_environment']),seed,method,p['modes'][method])
                memory=SpikeOverlapMemory(encode(model,training))
                selected=[r for r in rows if (r['world'],r['seed'],r['method'])==(spec['name'],seed,method)]
                scorer=Scorer(world,model,memory,encode,'clean',0.,base['geometry_seed'])
                for row in selected:
                    assert fingerprint(model)==row['encoder_fingerprint']
                    assert fingerprint(memory)==row['memory_fingerprint']
                    queries=json.loads((out/row['trace']).read_text())['sensor']
                    if not queries:continue
                    for i in sorted({0,len(queries)//2,len(queries)-1}):
                        q=queries[i]
                        values=scorer(q['position'],np.array(q['headings']))
                        actual=[float(v) if np.isfinite(v) else None for v in values]
                        assert actual==q['scores'], (row['id'],i)
                        checked+=len(values)
    result=dict(status='passed',trials=len(rows),observations=observations,
        matched_gaze_comparisons=comparisons,original_baselines_exact=baseline_matches,
        resource_accounting_checked=True,trace_sha256=hashes,
        independently_reencoded_scores=checked,memories_rebuilt=True,
        panorama_sha256=panoramas,
        report_source_sha256=file_hash(Path(__file__)),
        limitation='Re-encodes first/middle/last queries of each trial, not every observation; remaining scores checked against sensor logs.')
    write_json(out/'audit.json',result)
    return result


def report(out,docs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p,rows=read(out)
    docs.mkdir(parents=True,exist_ok=True)
    result=audit(out)
    write_json(docs/'audit.json',result)
    write_json(docs/'protocol.json',p)
    with (docs/'trials.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    paired=[]
    for w in p['worlds']:
        for method in p['methods']:
            for scenario in p['scenarios']:
                for seed in p['seeds']:
                    subset=[r for r in rows if (r['world'],r['method'],r['scenario'],r['seed'])==
                            (w['name'],method,scenario['name'],seed)]
                    policies={c:[r for r in subset if r['controller']==c] for c in ('exhaustive','active','familiarity')}
                    values={c:float(np.mean([r['reached_nest'] for r in rs])) for c,rs in policies.items()}
                    paired.append(dict(world=w['name'],method=method,scenario=scenario['name'],seed=seed,
                        arrival=values,delta_vs_exhaustive=values['familiarity']-values['exhaustive'],
                        delta_vs_active=values['familiarity']-values['active']))
    write_json(docs/'paired.json',dict(rows=paired,unit='Phase-averaged arrival within each world/seed/method/scenario. No significance claim.'))
    labels={'exhaustive':'Full scan','active':'Previous active','familiarity':'New controller'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,scenario in zip(axes,['aligned','left50']):
        for j,method in enumerate(p['methods']):
            for k,controller in enumerate(labels):
                rs=[r for r in rows if (r['method'],r['controller'],r['scenario'])==(method,controller,scenario)]
                x=j+(k-1)*.24
                n=sum(r['reached_nest'] for r in rs)
                ax.bar(x,n/len(rs),width=.22,color=COLOURS[method],alpha=[.35,.65,1][k])
                ax.text(x,n/len(rs)+.03,f'{n}/{len(rs)}',ha='center',fontsize=8)
        ax.set(xticks=range(3),xticklabels=[NAMES[m] for m in p['methods']],ylim=(0,1.17),ylabel='Fraction reaching endpoint',
               title='Taught start' if scenario=='aligned' else 'Start 50 cm off-route')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Development test · bars: full scan / previous active / new controller\n'
                 f"New controller includes both search phases; {len(p['worlds'])} worlds and {len(p['seeds'])} wiring seed(s)",fontsize=11)
    fig.savefig(docs/'performance.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for col,w in enumerate(p['worlds']):
        base=json.loads((Path(w['environment'])/'protocol.json').read_text())
        route=np.array(base['route'])
        rocks=json.loads((Path(w['environment'])/'world.json').read_text())['obstacles']
        for ax,scenario in zip(axes[:,col],['aligned','left50']):
            ax.plot(*route.T,color='black',ls=':',label='Taught route')
            drawn=0
            for rock in rocks:
                ax.add_patch(plt.Circle((rock['x'],rock['y']),rock['conservative_radius_m'],color='#dddddd'))
            for row in rows:
                if row['world']!=w['name'] or row['scenario']!=scenario or row['controller']!='familiarity':continue
                trajectory=json.loads((out/row['trace']).read_text())['trajectory']
                if not trajectory:continue
                xy=np.array([t['position'] for t in trajectory])
                drawn+=1
                ax.plot(*xy.T,color=COLOURS[row['method']],ls='-' if row['phase']==1 else '--',lw=1.3,
                        label=NAMES[row['method']] if row['phase']==1 else None)
            ax.set(xlim=(0,10.4),ylim=(0,10.4),aspect='equal',title=f"{w['name']} · {scenario}",xlabel='metres',ylabel='metres')
            if not drawn:
                case=next(s for s in p['scenarios'] if s['name']==scenario)
                heading=np.radians(base['headings'][0])
                start=route[0]+case['lateral']*np.array([-np.sin(heading),np.cos(heading)])
                ax.plot(*start,marker='x',color='red',ms=8)
                count=sum(r['world']==w['name'] and r['scenario']==scenario and r['controller']=='familiarity' for r in rows)
                ax.text(.5,.88,f'All {count} runs stop\nbefore the first move',transform=ax.transAxes,
                        ha='center',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('New controller paths · solid/dashed lines are the two initial search phases')
    fig.savefig(docs/'trajectories.png',dpi=160);plt.close(fig)
    probe=out/'common-probe/results.json'
    if probe.exists():
        diagnostic=json.loads(probe.read_text())
        write_json(docs/'signal-summary.json',dict(summary=diagnostic['summary'],caveat=diagnostic['caveat']))
        fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
        for ax,method in zip(axes,p['methods']):
            for phase,style,label in [('taught','-','At teaching stations'),('between','--','Between teaching stations')]:
                rs=[r for r in diagnostic['summary'] if r['method']==method and r['phase']==phase]
                ax.plot([r['distance_m']*100 for r in rs],[r['fraction_inward_positive'] for r in rs],
                        style,marker='o',color=COLOURS[method],label=label)
            ax.set(title=NAMES[method],ylim=(0,1.05),xticks=[10,20,50],xlabel='Starting offset (cm)',
                   ylabel='Fraction of inward moves improving familiarity')
            ax.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=8)
        fig.suptitle('Identical 10 cm inward movement probes · three development worlds, three wiring seeds\nCorrelated samples; descriptive proportions, not independent repetitions',fontsize=10)
        fig.savefig(docs/'signal.png',dpi=180);plt.close(fig)
    for name in ('synthetic','ablations'):
        source=out/name/'results.json'
        if source.exists():write_json(docs/f'{name}-summary.json',json.loads(source.read_text()))
    return rows


def video(out,docs,trial_id):
    """Replay actual observations at 4x simulated time; no fabricated views."""
    from .navigation_video import font
    p,rows=read(out)
    row=next(r for r in rows if r['id']==trial_id)
    saved=json.loads((out/row['trace']).read_text())
    spec=next(w for w in p['worlds'] if w['name']==row['world'])
    environment=Path(spec['environment'])
    base=json.loads((environment/'protocol.json').read_text())
    route=np.array(base['route'])
    observations=[e for e in saved['events'] if e['kind']=='observation']
    docs.mkdir(parents=True,exist_ok=True)
    target=docs/f'{trial_id}.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x720','-r','8','-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
    cache={}
    def point(xy):return (int(800+xy[0]*40),int(635-xy[1]*40))
    try:
        for now in np.arange(0,row['time_s']+2,.5):
            seen=[e for e in observations if e['time_s']<=now]
            event=seen[-1] if seen else observations[0]
            frame=Image.new('RGB',(1280,720),'#101c26');draw=ImageDraw.Draw(frame)
            def text(x,y,value,size=20,fill='#eaf2f2'):
                draw.text((x,y),value,font=font(size),fill=fill)
            text(30,20,f"{NAMES[row['method']]} · {row['world']} · {row['scenario']}",28)
            text(30,60,'Recorded observations · playback 4× simulated time',18,'#acc0cb')
            key=(position_key(event['position']),event['heading'])
            if key not in cache:
                with Image.open(environment/'panoramas'/f'{key[0]}.png') as image:
                    panorama=np.asarray(image.convert('RGB'))
                view=sample_panorama(panorama,[key[1]],p['shape'])[0].numpy().transpose(1,2,0)
                cache[key]=Image.fromarray((255*view).clip(0,255).astype('uint8'))
            if seen:
                frame.paste(cache[key].resize((740,190),Image.Resampling.NEAREST),(30,130))
            else:
                text(30,190,'Waiting for the first observation',22)
            text(30,100,'Most recent ant view · actual 199 × 51 network input',18)
            draw.line([point(v) for v in route],fill='#d5dddd',width=3)
            walked=[t for t in saved['trajectory'] if t['time_s']<=now]
            if walked:
                points=[point(t['position']) for t in walked]
                if len(points)>1:draw.line(points,fill='#56d3c2',width=3)
            xy=point(event['position'])
            draw.ellipse((xy[0]-6,xy[1]-6,xy[0]+6,xy[1]+6),fill='#f6bf65')
            heading=np.radians(event['heading'])
            draw.line([xy,(xy[0]+20*np.cos(heading),xy[1]-20*np.sin(heading))],fill='#f6bf65',width=3)
            text(815,105,'Taught route and observed positions',18)
            decisions=[d for d in saved['decisions'] if d.get('time_end_s',row['time_s'])<=now]
            decision=decisions[-1] if decisions else {}
            state=decision.get('state',decision.get('state_before','reorient'))
            text(30,350,f"Last decision: {state.upper()}",27)
            text(30,395,(f"Familiarity: {event['familiarity']:.3f}" if seen else 'Familiarity: awaiting observation')+f"     Observations: {len(seen)}",22)
            c=decision.get('comparison')
            text(30,435,'Change after movement: '+(f"{c['normalized_change']:+.3f} (fixed scale)" if c else 'awaiting a matched pair'),20)
            text(30,475,f"Time: {min(now,row['time_s']):.1f} s     Walked: {len(walked)*.1:.1f} m",22)
            text(30,520,'Route map is for the viewer; the controller cannot access it.',17,'#acc0cb')
            text(30,550,'View held until the next actual observation; turns and sensing are charged.',17,'#acc0cb')
            if now>=row['time_s']:text(30,610,'Trial ended: '+row['termination'].replace('_',' '),26,'#f6bf65')
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
        code=process.wait()
    if code:raise RuntimeError(f'ffmpeg exited {code}')
    write_json(target.with_suffix('.json'),dict(trial=trial_id,trace_sha256=file_hash(out/row['trace']),
        video_sha256=file_hash(target),playback_speed=4,view='Most recent logged observation, held between samples'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/familiarity-controller'))
    parser.add_argument('--docs',type=Path,default=Path('docs/familiarity-controller'))
    parser.add_argument('--video',help='Exact trial id to replay instead of generating report')
    args=parser.parse_args()
    if args.video:video(args.output,args.docs,args.video)
    else:report(args.output,args.docs)
