"""Verify frozen graded-UV controls and report all factorial outcomes."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import load_cached,digest
from apiaviz.research.spectral_input import file_sha


def delta(a,b):return abs((a-b+180)%360-180)


def summarize(rows):
    keys=list(dict.fromkeys((r['adaptive'],r['mode'],r['readout'],r['gain']) for r in rows))
    index={(r['world'],r['station'],r['kind'],r['seed'],r['direction'],r['adaptive'],r['mode'],r['readout'],r['gain']):r for r in rows}
    assert len(index)==len(rows)
    result=[]
    for world in ['all','meander','bend','hairpin']:
        for adaptive,mode,readout,gain in keys:
            for kind in ['all','taught','midpoint','offroute']:
                group=[r for r in rows if (r['adaptive'],r['mode'],r['readout'],r['gain'])==(adaptive,mode,readout,gain)
                       and (world=='all' or r['world']==world)
                       and (kind=='all' or r['kind']==kind or (kind=='offroute' and '20' in r['kind']))]
                change=[];scan_change=[]
                for r in group:
                    key=(r['world'],r['station'],r['kind'],r['seed'],r['direction'],adaptive,mode,readout,1.)
                    change.append(delta(r['heading'],index[key]['heading']))
                    if r['direction']==1 and mode!='current':
                        reverse=index[r['world'],r['station'],r['kind'],r['seed'],-1,adaptive,mode,readout,gain]
                        scan_change.append(delta(r['heading'],reverse['heading']))
                result.append(dict(world=world,adaptive=adaptive,mode=mode,readout=readout,gain=gain,kind=kind,n=len(group),
                    heading_error_mean=float(np.mean([r['heading_error'] for r in group])),
                    within5=sum(r['heading_error']<=5+1e-6 for r in group),within20=sum(r['heading_error']<=20+1e-6 for r in group),
                    forward_projection_mean_m=float(np.mean([r['forward_projection_m'] for r in group])),
                    positive_forward=sum(r['forward_projection_m']>1e-8 for r in group),
                    route_distance_change_mean_m=float(np.mean([r['route_distance_change_m'] for r in group])),
                    correct_station_score_mean=float(np.mean([r['correct_station_score'] for r in group])),
                    illumination_heading_change_mean=float(np.mean(change)),
                    scan_direction_change_mean=float(np.mean(scan_change)) if scan_change else None))
    # Matched intervention effects: lower heading error is better.
    paired=[]
    for intervention in ['adaptation','opponency','readout','uv']:
        for gain in [.25,1.,4.]:
            changes=[]
            for r in rows:
                if r['gain']!=gain or r['mode']=='current':continue
                a,mode,read=r['adaptive'],r['mode'],r['readout']
                if intervention=='adaptation':
                    if a or read!='spikes':continue
                    target=(True,mode,read)
                elif intervention=='opponency':
                    if mode!='pairwise' or read!='spikes':continue
                    target=(a,'combined',read)
                elif intervention=='readout':
                    if read!='spikes':continue
                    target=(a,mode,'graded')
                else:
                    if read!='spikes_no_uv':continue
                    target=(a,mode,'spikes')
                other=index[r['world'],r['station'],r['kind'],r['seed'],r['direction'],*target,gain]
                changes.append(other['heading_error']-r['heading_error'])
            changes=np.asarray(changes)
            paired.append(dict(intervention=intervention,gain=gain,n=len(changes),
                mean_heading_error_change=float(changes.mean()),better=int((changes < -1e-6).sum()),
                worse=int((changes > 1e-6).sum()),same=int((abs(changes)<=1e-6).sum())))
    return result,paired


def figures(out,summary,stimuli):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-graded-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,gain in zip(axes,[.25,1.,4.]):
        for adaptive,mode,label in [(False,'pairwise','Fixed response, pairwise'),(False,'combined','Fixed response, combined'),
                                    (True,'pairwise','Adaptive, pairwise'),(True,'combined','Adaptive, combined')]:
            rs=[next(r for r in summary if r['world']=='all' and r['kind']=='all' and r['adaptive']==adaptive and r['mode']==mode and r['readout']==readout and r['gain']==gain) for readout in ['graded','spikes']]
            ax.plot([0,1],[r['heading_error_mean'] for r in rs],'o-',label=label)
        ax.set(title=f'{gain:g} × radiance',xticks=[0,1],xticklabels=['Graded currents','Spike identities'],ylabel='Mean heading error (degrees)')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.suptitle('Same inputs and wiring; equal stream weights in both readouts • 84 poses × 3 seeds × 2 scan directions')
    fig.savefig(out/'factorial.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for adaptive,mode,label in [(False,'pairwise','Fixed, pairwise'),(False,'combined','Fixed, combined'),(True,'pairwise','Adaptive, pairwise'),(True,'combined','Adaptive, combined')]:
        rs=[next(r for r in summary if r['world']=='all' and r['kind']=='all' and r['adaptive']==adaptive and r['mode']==mode and r['readout']=='spikes' and r['gain']==gain) for gain in [.25,1.,4.]]
        for ax,field in zip(axes,['illumination_heading_change_mean','scan_direction_change_mean']):
            ax.plot([.25,1.,4.],[r[field] for r in rs],'o-',label=label)
            ax.set_xscale('log',base=4);ax.set_xticks([.25,1.,4.],['0.25','1','4']);ax.set_xlabel('Radiance multiplier')
    axes[0].set_ylabel('Heading change from normal lighting (degrees)')
    axes[1].set_ylabel('Heading disagreement between scan directions (degrees)')
    axes[1].legend(fontsize=8);fig.suptitle('Stability diagnostics, equal-stream spike readout')
    fig.savefig(out/'stability.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for r in stimuli['step_responses']:
        if r['config']['integration_tau_s']!=.02 or r['gain']!=4.:continue
        a=np.asarray(r['response'])
        axes[0].plot(np.arange(1,401)*.01,a[:,0],label=f'Gain time {r["config"]["gain_tau_s"]:g} s')
    axes[0].axhline(.5,color='black',ls=':');axes[0].set(xlabel='Time after 4× brightness step (s)',ylabel='Dimensionless receptor response');axes[0].legend(fontsize=8)
    for r in stimuli['colour_steps']:
        a=np.asarray(r['response']);signed=a[:,0]-a[:,1]
        axes[1].plot(np.arange(1,401)*.01,signed,label=r['stimulated_receptor'])
    axes[1].set(xlabel='Time after one receptor doubles (s)',ylabel='UV − (blue + green)/2');axes[1].legend()
    for r in stimuli['step_responses']:
        if r['config']['gain_tau_s']!=1. or r['gain']!=4.:continue
        axes[2].plot(np.arange(1,21)*.01,np.asarray(r['response'])[:20,0],label=f'{r["config"]["integration_tau_s"]*1000:g} ms')
    axes[2].set(xlabel='Time after brightness step (s)',ylabel='Receptor response');axes[2].legend(title='Integration constant')
    fig.suptitle('Controlled stimuli • model properties and parameter sensitivity, not fits to recorded voltages')
    fig.savefig(out/'stimuli.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    p=json.loads((a.study/'protocol.json').read_text());complete=json.loads((a.study/'complete.json').read_text())
    assert complete['protocol_sha256']==file_sha(a.study/'protocol.json')
    assert complete['stimuli_sha256']==file_sha(a.study/'stimuli.json')
    for name,sha in p['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    parent=Path(p['source']);assert file_sha(parent/'protocol.json')==p['source_protocol_sha256']
    records=[];inputs={};count=0
    for world,sha in complete['results'].items():
        path=a.study/f'{world}.json';assert file_sha(path)==sha;inputs[str(path)]=sha
        d=json.loads(path.read_text());original=json.loads((parent/world/'results.json').read_text())
        assert original['raw_records']==d['raw_records']
        assert original['source_teaching_sha256']==d['teaching_sha256']
        config=json.loads((parent/world/'render.json').read_text())
        for key,sha in d['raw_records'].items():
            assert file_sha(parent/world/'camera'/f'{key}.json')==sha
            load_cached(parent/world/'camera',key,digest(config));count+=1
        assert len(d['records'])==28*(2*3*2*2*3*6+3*3)
        records.extend(d['records'])
    summary,paired=summarize(records)
    stimuli=json.loads((a.study/'stimuli.json').read_text())
    base.atomic_json(out/'results.json',dict(summary=summary,paired=paired,inputs=inputs,
        protocol_sha256=file_sha(a.study/'protocol.json'),script_sha256=file_sha(Path(__file__)),
        audit=dict(passed=True,raw_frames=count,records=len(records),sources='All frozen source hashes verified',
            teacher_and_raw_inputs='Exactly the earlier matched-control bank'),limitations=p['limitations']))
    figures(out,summary,stimuli)
    for r in summary:
        if r['world']=='all' and r['kind']=='all' and r['readout'] in ('graded','spikes'):
            print(r['adaptive'],r['mode'],r['readout'],r['gain'],round(r['heading_error_mean'],3),r['within20'],r['n'],flush=True)


if __name__=='__main__':main()
