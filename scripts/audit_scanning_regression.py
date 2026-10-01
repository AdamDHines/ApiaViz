"""Read-only turn accounting, video aliasing and saved-score scan controls.

No renders or navigation trials. Counterfactual scans reuse complete score
tables from stationary recorded poses; they are not counterfactual arrivals.
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research import uv_trials as base
from apiaviz.research.active_navigation import Observations
from apiaviz.research.familiarity_controller import FamiliarityController,Settings,wrap
from apiaviz.research.spectral_input import file_sha
import torch


def yaw_at(events,times,initial):
    """Continuous constant-rate yaw implied by logged finite-duration turns."""
    turns=[e for e in events if e['kind']=='turn']
    starts=np.array([e['time_s']-e['duration_s'] for e in turns])
    result=[]
    for t in times:
        i=int(np.searchsorted(starts,t,side='right')-1)
        if i<0:result.append(initial);continue
        e=turns[i]
        fraction=1. if e['duration_s']==0 else np.clip((t-starts[i])/e['duration_s'],0,1)
        result.append(e['from_heading']+fraction*e['angle_deg'])
    return np.asarray(result)


def metrics(study,p,detail):
    row=detail['result'];events=detail['events']
    turns=[e for e in events if e['kind']=='turn'];angles=np.array([abs(e['angle_deg']) for e in turns])
    np.testing.assert_allclose(sum(angles),row['rotation_deg'],atol=1e-6,rtol=0)
    grouped={}
    for i,e in enumerate(events):
        if e['kind']!='turn':continue
        nxt=events[i+1]['kind'] if i+1<len(events) else 'end'
        tag={'observation':'route_observation','avoidance_observation':'locomotion_gaze'}.get(nxt,nxt)
        group=grouped.setdefault(tag,dict(count=0,rotation_deg=0.,large90=0))
        group['count']+=1;group['rotation_deg']+=abs(e['angle_deg']);group['large90']+=abs(e['angle_deg'])>=90-1e-8
    moved=[v for v in detail['microtrace'] if v['stride_m']>0]
    headings=np.array([v['heading'] for v in moved]);changes=abs(wrap(np.diff(headings)))
    scans=[v['scan'] for v in detail['decisions'] if 'scan' in v]
    full=[s for s in scans if len(s['samples'])>=36]
    duplicate=sum(len(s['samples'])-len({round(wrap(x['heading']),6) for x in s['samples']}) for s in scans)
    views=[e for e in events if e['kind'] in ('observation','avoidance_observation')]
    viewtimes=np.array([e['time_s'] for e in views]);viewyaw=np.array([e['heading'] for e in views])
    duration=min(p['movies']['max_seconds'],max(1.,row['time_s']/8))
    frames=np.linspace(0,row['time_s'],max(2,int(duration*p['movies']['fps'])))
    world=next(w for w in p['worlds'] if w['name']==row['world'])
    initial=world['headings'][0]+next(s for s in p['scenarios'] if s['name']==row['scenario'])['heading']
    ix=np.searchsorted(viewtimes,frames,side='right')-1
    shown=np.where(ix>=0,viewyaw[np.maximum(0,ix)],initial)
    jumps=abs(wrap(np.diff(shown)));exact=yaw_at(events,frames,initial)
    return dict(study=study.name,id=row['id'],world=row['world'],method=row['method'],
        termination=row['termination'],path_m=row['path_length_m'],time_s=row['time_s'],
        total_rotation_deg=float(sum(angles)),equivalent_revolutions=float(sum(angles)/360),
        turning_s=sum(e['duration_s'] for e in turns),max_single_turn_deg=float(max(angles)),
        single_turns_ge90=int(sum(angles>=90-1e-8)),single_turns_ge150=int(sum(angles>=150-1e-8)),
        movement_heading_changes_ge90=int(sum(changes>=90-1e-8)),
        max_movement_heading_change_deg=float(max(changes)) if len(changes) else 0.,
        full_scans=len(full),recorded_scans=len(scans),duplicate_scan_headings=duplicate,
        scan_sizes=dict(Counter(len(s['samples']) for s in scans)),turn_followed_by=grouped,
        video=dict(frames=len(frames),simulated_seconds_per_frame=float(frames[1]-frames[0]),
            speed_factor=float(row['time_s']/duration),jumps_ge90=int(sum(jumps>=90-1e-8)),
            jumps_ge150=int(sum(jumps>=150-1e-8)),max_jump_deg=float(max(jumps)),
            max_heading_hold_error_deg=float(max(abs(wrap(exact-shown))))))


def saved_scan_controls(p,details,study):
    class ReplaySensor(Observations):
        def start_scan(self):self.scan_bouts+=1
    records=[]
    for row,detail in details:
        if row['world'] not in {w['name'] for w in p['worlds']}:raise ValueError(row['world'])
        world=next(w for w in p['worlds'] if w['name']==row['world'])
        checkpoint=study/'worlds'/row['world']/f'encoder-{row["method"]}-{row["seed"]}.pt'
        calibration=torch.load(checkpoint,weights_only=True,map_location='cpu')['calibration']
        anchor=world['headings'][0]
        for decision in detail['decisions']:
            scan=decision.get('scan')
            if scan and len(scan['samples'])>=72:
                samples=scan['samples'];hs=np.array([v['heading'] for v in samples]);ys=np.array([v['familiarity'] for v in samples])
                def scorer(position,headings):
                    result=[]
                    for h in headings:
                        distances=abs(wrap(hs-h));i=int(distances.argmin())
                        if distances[i]>1e-6:raise ValueError(('Unobserved counterfactual heading',row['id'],decision['step'],h,float(distances[i])))
                        result.append(-ys[i])
                    return np.asarray(result)
                variants={}
                for name,step,extents,wide in [('full5',5.,(180.,),True),
                        ('progressive5',5.,(20.,60.,180.),True),('progressive10',10.,(20.,60.,180.),True),
                        ('full10',10.,(180.,),True),('local5',5.,(20.,60.,180.),False),('local10',10.,(20.,60.,180.),False)]:
                    sensor=ReplaySensor(scorer,[0.,0.],anchor,p['sensor_settings'])
                    controller=FamiliarityController(calibration,replace(Settings(**p['controller_settings']),scan_step=step,scan_extents=extents),phase=row['phase'])
                    result=controller._scan(sensor,anchor,wide)
                    assert result is not None
                    sensor.rotate(result['target'])
                    variants[name]=dict(target=result['target'],supported=result['supported'],contrast=result['contrast'],
                        observations=sensor.count,rotation_deg=sensor.rotation_deg,time_s=sensor.time,
                        max_turn_deg=max(abs(e['angle_deg']) for e in sensor.events if e['kind']=='turn'))
                np.testing.assert_allclose(wrap(variants['full5']['target']-scan['target']),0,atol=1e-6)
                assert variants['full5']['supported']==scan['supported']
                records.append(dict(id=row['id'],world=row['world'],method=row['method'],step=decision['step'],
                    anchor=anchor,recorded_samples=len(samples),variants=variants))
            if decision.get('reference_heading') is not None:anchor=decision['reference_heading']
    return records


def run(study,parent,out):
    if (out/'results.json').exists():raise FileExistsError('Use a fresh diagnostic output')
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    inputs={};summary=[];details=[]
    for source in (parent,study):
        protocol=json.loads((source/'protocol.json').read_text());inputs[str(source/'protocol.json')]=file_sha(source/'protocol.json')
        for path in sorted((source/'trials').glob('*familiarity-1.json')):
            detail=json.loads(path.read_text());row=detail['result'];inputs[str(path)]=file_sha(path)
            summary.append(metrics(source,protocol,detail))
            if source==study:details.append((row,detail))
    records=saved_scan_controls(p,details,study)
    comparisons=[]
    for method in p['methods']:
        group=[r for r in records if r['method']==method]
        cases=[]
        for variant in ('progressive5','progressive10','full10'):
            cases.append(dict(variant=variant,n=len(group),
                same_heading=int(sum(abs(wrap(r['variants'][variant]['target']-r['variants']['full5']['target']))<1e-6 for r in group)),
                same_heading_within5=int(sum(abs(wrap(r['variants'][variant]['target']-r['variants']['full5']['target']))<=5+1e-6 for r in group)),
                mean_views=float(np.mean([r['variants'][variant]['observations'] for r in group])),
                mean_rotation=float(np.mean([r['variants'][variant]['rotation_deg'] for r in group])),
                mean_time_s=float(np.mean([r['variants'][variant]['time_s'] for r in group])),
                support_disagreements=sum(r['variants'][variant]['supported']!=r['variants']['full5']['supported'] for r in group)))
        comparisons.append(dict(method=method,n=len(group),variants=cases,
            local10_supported_local5_rejected=sum(r['variants']['local10']['supported'] and not r['variants']['local5']['supported'] for r in group),
            local5_supported_local10_rejected=sum(r['variants']['local5']['supported'] and not r['variants']['local10']['supported'] for r in group)))
    files=['apiaviz/research/familiarity_controller.py','apiaviz/research/active_navigation.py',
        'apiaviz/research/graded_smoke.py','apiaviz/research/uv_trial_report.py',
        'apiaviz/research/navigation_video.py','apiaviz/research/route_navigation_video.py']
    base.atomic_json(out/'results.json',dict(script_sha256=file_sha(Path(__file__)),inputs=inputs,
        sources={f:file_sha(ROOT/f) for f in files},metrics=summary,scan_controls=records,comparisons=comparisons,
        limitations='Post-hoc recorded full-scan positions only; all counterfactual scans start facing the prior anchor, use recorded scores, and include return to chosen direction. No new rendering/movement/arrivals. Failed and successful trials have different durations. Old full pre-UV traces are not present on this workstation.'))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-scan-review-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from apiaviz.research.uv_trial_report import NAMES,COLORS
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    current=[r for r in summary if r['study']==study.name]
    for j,world in enumerate(p['worlds']):
        for k,method in enumerate(p['methods']):
            r=next(r for r in current if r['world']==world['name'] and r['method']==method)
            for ax,field in zip(axes,['equivalent_revolutions','single_turns_ge90']):
                ax.bar(j+(k-1)*.24,r[field],.23,color=COLORS[method],label=NAMES[method] if j==0 else None)
            axes[2].bar(j+(k-1)*.24,r['video']['jumps_ge90'],.23,color=COLORS[method])
    for ax,title in zip(axes,['Angular travel / 360°','Actual turn commands ≥90°','Video heading jumps ≥90°']):
        ax.set(title=title,xticks=range(3),xticklabels=[w['name'] for w in p['worlds']]);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Real repeated scans and coarse video sampling both contribute')
    fig.savefig(out/'rotation-summary.png',dpi=160);plt.close(fig)
    row,detail=next((r,d) for r,d in details if r['world']=='meander' and r['method']=='apiaviz_uv')
    events=detail['events'];times=np.linspace(0,12,1201);initial=p['worlds'][0]['headings'][0]
    angles=yaw_at(events,times,initial)
    views=[e for e in events if e['kind'] in ('observation','avoidance_observation')];ts=np.array([v['time_s'] for v in views])
    yaw=np.array([v['heading'] for v in views]);frame_times=np.linspace(0,row['time_s'],96);ix=np.searchsorted(ts,frame_times,side='right')-1
    shown=np.where(ix>=0,yaw[np.maximum(0,ix)],initial);mask=frame_times<=12
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.plot(times,angles-initial,label='Logged yaw, constant-rate turn interpolation')
    ax.scatter(frame_times[mask],shown[mask]-initial,color='#c66028',label='Frames retained in current 12 s summary movie',zorder=3)
    ax.set(xlabel='Simulation time (s)',ylabel='Accumulated heading relative to start (degrees)',title='First 12 simulated seconds · unchanged ApiaViz meander trial');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'first-scan-sampling.png',dpi=160);plt.close(fig)
    print(json.dumps(comparisons,indent=2))
    print('Audited',len(summary),'traces;',len(records),'full-scan tables')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/graded-navigation-smoke-v1')
    parser.add_argument('--parent',type=Path,default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/scan-regression-review-v1')
    args=parser.parse_args();run(args.study.resolve(),args.parent.resolve(),args.output.resolve())
