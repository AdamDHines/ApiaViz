"""Verify and summarize matched response experiments without ranking by arrivals."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import load_cached, digest
from apiaviz.research.collision_geometry import RockGeometry
from apiaviz.research.spectral_input import file_sha

LABELS=dict(api_uv='ApiaViz + UV',api_bee='ApiaViz, UV stream off',api_rgb='ApiaViz, matched RGB',
    sobel='Sobel, matched RGB',ardin='Ardin-style, matched RGB',uv_only='UV stream alone',
    blue_replaces_uv='Blue substituted for UV',rgb_plus_uv='RGB ApiaViz + UV',
    api_rgb_small='ApiaViz RGB, small image',sobel_small='Sobel RGB, small image',
    sobel10k='Sobel, production 10k',ardin10k='Ardin, production 10k',
    angular_rgb='ApiaViz RGB, restored angular filters',angular_bee='ApiaViz bee B/G, angular filters',
    angular_sobel='Sobel RGB, restored angular filters',angular_bee_plus_uv='Angular bee B/G + current UV',
    angular_rgb_plus_uv='Angular RGB + current UV')


def aggregate(rows):
    result=[]
    for world in ['all','meander','bend','hairpin']:
        for name in dict.fromkeys(r['condition'] for r in rows):
            for kind in ['taught','midpoint','offroute']:
                a=[r for r in rows if r['condition']==name and (world=='all' or r['world']==world)
                   and (('20' in r['kind']) if kind=='offroute' else r['kind']==kind)]
                for grid in ['grid5','grid10','grid10_shift5']:
                    result.append(dict(world=world,condition=name,kind=kind,grid=grid,n=len(a),
                        heading_error_mean=float(np.mean([r[grid]['error'] for r in a])),
                        within5=sum(r[grid]['error']<=5+1e-7 for r in a),
                        inward=sum(r[grid]['inward'] for r in a),
                        mean_route_distance_change_m=float(np.mean([r[grid]['after_m']-r[grid]['before_m'] for r in a])),
                        mean_place_margin=float(np.mean([r['place_margin'] for r in a])),
                        correct_place_beats_far=sum(r['place_margin']>0 for r in a),
                        mean_five_degree_loss=float(np.mean([r['five_degree_loss'] for r in a])),
                        mean_same_pose_recall=float(np.mean([r['correct_station_score'] for r in a]))))
    return result


def comparisons(rows):
    index={(r['condition'],r['world'],r['seed'],r['station'],r['kind']):r for r in rows}
    result=[]
    pairs=[('api_bee','api_uv'),('api_rgb','rgb_plus_uv'),('blue_replaces_uv','api_uv'),
           ('api_rgb','api_bee'),('sobel','api_rgb'),('api_rgb','angular_rgb'),
           ('api_bee','angular_bee'),('sobel','angular_sobel'),('angular_bee','angular_bee_plus_uv')]
    for before,after in pairs:
        for kind in ['taught','midpoint','offroute']:
            a=[r for r in rows if r['condition']==before and (('20' in r['kind']) if kind=='offroute' else r['kind']==kind)]
            b=[index[after,r['world'],r['seed'],r['station'],r['kind']] for r in a]
            delta=np.array([y['grid5']['error']-x['grid5']['error'] for x,y in zip(a,b)])
            distance=np.array([y['grid5']['after_m']-x['grid5']['after_m'] for x,y in zip(a,b)])
            result.append(dict(before=before,after=after,kind=kind,n=len(a),
                heading_better=int((delta < -1e-6).sum()),heading_worse=int((delta > 1e-6).sum()),
                heading_equal=int((abs(delta)<=1e-6).sum()),mean_error_change=float(delta.mean()),
                projected_distance_better=int((distance < -1e-9).sum()),projected_distance_worse=int((distance > 1e-9).sum()),
                mean_projected_distance_change_m=float(distance.mean())))
    return result


def figures(out,rows,summary,stage_rows,source):
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-response-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['api_uv','api_bee','api_rgb','sobel','ardin','blue_replaces_uv','angular_rgb','angular_sobel','angular_bee_plus_uv']
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,kind,field,label in [(axes[0],'midpoint','heading_error_mean','Heading error between teaching positions (degrees)'),
                                (axes[1],'offroute','inward','20 cm displaced: directions pointing inward (%)')]:
        selected=[next(r for r in summary if r['condition']==name and r['world']=='all' and r['kind']==kind and r['grid']=='grid5') for name in names]
        values=[r[field] if field!='inward' else 100*r[field]/r['n'] for r in selected]
        ax.barh(np.arange(len(names)),values,color='#427c91')
        for world,marker in [('meander','o'),('bend','s'),('hairpin','^')]:
            wr=[next(r for r in summary if r['condition']==name and r['world']==world and r['kind']==kind and r['grid']=='grid5') for name in names]
            ax.scatter([r[field] if field!='inward' else 100*r[field]/r['n'] for r in wr],np.arange(len(names)),marker=marker,label=world,edgecolor='white',s=45)
        ax.set(yticks=np.arange(len(names)),yticklabels=[LABELS[n] for n in names],xlabel=label)
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[1].legend(loc='lower right');fig.suptitle('Identical images, full 5-degree scans • offline probes, not navigation success rates')
    fig.savefig(out/'matched-responses.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,world in zip(axes,['meander','bend','hairpin']):
        for name in ['api_uv','api_rgb','sobel','angular_rgb','angular_sobel']:
            a=[r['same_station_yaw_curve'] for r in rows if r['condition']==name and r['world']==world and r['kind']=='taught']
            ax.plot(range(-20,21),np.mean(a,axis=0),label=LABELS[name])
        ax.set(title=world,xlabel='Turn away from taught heading (degrees)',ylabel='Match to that same taught view',ylim=(.3,1.02))
        ax.axvline(5,color='black',lw=.7,ls=':');ax.axvline(-5,color='black',lw=.7,ls=':')
    axes[-1].legend(fontsize=7);fig.suptitle('Angular tolerance • 7 fixed stations × 3 wirings per world • independent recall render')
    fig.savefig(out/'heading-tolerance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,stage in zip(axes,['features','currents','spikes']):
        for method in ['api_uv','api_rgb','sobel']:
            values=[r['curve'] for r in stage_rows if r['method']==method and r['stage']==stage]
            ax.plot(range(-20,21),np.mean(values,axis=0),label=LABELS[method])
        ax.set(title=stage,xlabel='Yaw (degrees)',ylabel='Same-view cosine similarity',ylim=(.3,1.02))
    axes[-1].legend(fontsize=8);fig.suptitle('The sharp response exists before the spiking stage • seed 19, all 21 taught probes')
    fig.savefig(out/'processing-stages.png',dpi=160);plt.close(fig)
    data=torch.load(source/'meander/feature-example.pt',weights_only=True)
    fig,axes=plt.subplots(3,4,figsize=(13,7),layout='constrained')
    rgb=data['rgb'].permute(1,2,0).numpy()
    rgb=np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)
    uv=data['uv']/(data['uv']+1.)
    for i,key in enumerate(['api_maps','sobel_maps','uv_maps']):
        if i<2:
            axes[i,0].imshow(rgb,aspect='auto');axes[i,0].set_title(['ApiaViz: identical visible input','Sobel: identical visible input'][i])
            planes=data[key][0][0].numpy()
            titles=['ON','OFF','Smoothed adapted form'] if i==0 else ['Horizontal gradient','Vertical gradient','Gradient magnitude']
        else:
            axes[i,0].imshow(uv.permute(1,2,0).numpy(),aspect='auto');axes[i,0].set_title('UV/B/G input (false colour)')
            planes=adaptive(data[key]['uv'])[0,:3].numpy();titles=['UV ON','UV OFF','Smoothed adapted UV']
        limit=max(float(abs(planes).max()),1e-12)
        for j,(plane,title) in enumerate(zip(planes,titles),1):
            im=axes[i,j].imshow(plane,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
            axes[i,j].set_title(title)
        fig.colorbar(im,ax=axes[i,1:],shrink=.7,label='Common scale within this row')
        for ax in axes[i]:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Fixed middle teaching station, meander • pooled feature maps before stream normalization')
    fig.savefig(out/'feature-maps.png',dpi=160);plt.close(fig)


def adaptive(x):
    from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize
    return adaptive_avg_pool2d_anysize(x,(8,64))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--angular',type=Path,required=True)
    parser.add_argument('--stages',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((a.source/'protocol.json').read_text());study=Path(parent['study'])
    p=json.loads((study/'protocol.json').read_text())
    rows=[];teaching=[];features={};inputs={};validated=0
    for folder in [a.source,a.angular,a.stages]:
        protocol=json.loads((folder/'protocol.json').read_text())
        for name,sha in protocol['source_sha256'].items():assert file_sha(ROOT/name)==sha,(folder,name)
        inputs[str(folder/'protocol.json')]=file_sha(folder/'protocol.json')
        if (folder/'complete.json').exists():
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['protocol_sha256']==file_sha(folder/'protocol.json')
            for world,sha in complete['results'].items():
                path=folder/world/'results.json' if folder==a.source else folder/f'{world}.json'
                assert file_sha(path)==sha
    for w in p['worlds']:
        world=w['name'];env=study/'worlds'/world
        original=json.loads((a.source/world/'results.json').read_text())
        angular=json.loads((a.angular/f'{world}.json').read_text())
        assert original['raw_records']==angular['raw_records']
        assert original['source_teaching_sha256']==angular['teaching_sha256']==file_sha(env/'teaching.pt')
        cfg=json.loads((env/'render.json').read_text())
        meta=json.loads((env/'world.json').read_text())
        geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
        for pose in parent['poses'][world]:
            assert not geometry.intersects(pose['position'],pose['position'])
        for key,sha in original['raw_records'].items():
            assert file_sha(a.source/world/'camera'/f'{key}.json')==sha
            load_cached(a.source/world/'camera',key,digest(cfg));validated+=1
        assert len(original['records'])==28*3*12
        assert len(angular['records'])==28*3*5
        rows.extend(original['records']+angular['records']);teaching.extend(original['teaching']+angular['teaching'])
        features[world]=original['features']
        for path in [a.source/world/'results.json',a.angular/f'{world}.json']:
            inputs[str(path)]=file_sha(path)
    assert len(rows)==84*3*17
    keys={(r['condition'],r['world'],r['seed'],r['station'],r['kind']) for r in rows}
    assert len(keys)==len(rows)
    stages=json.loads((a.stages/'results.json').read_text())
    assert stages['protocol_sha256']==file_sha(a.stages/'protocol.json')
    inputs[str(a.stages/'results.json')]=file_sha(a.stages/'results.json')
    summary=aggregate(rows)
    stage_summary=[]
    for method,stage in dict.fromkeys((r['method'],r['stage']) for r in stages['records']):
        group=[r for r in stages['records'] if r['method']==method and r['stage']==stage]
        stage_summary.append(dict(method=method,stage=stage,n=len(group),
            mean_repeat=float(np.mean([r['repeat'] for r in group])),
            mean_yaw5_loss=float(np.mean([r['yaw5_loss'] for r in group]))))
    base.atomic_json(out/'results.json',dict(summary=summary,paired_controls=comparisons(rows),
        features=features,teaching=teaching,processing_stages=stage_summary,inputs=inputs,
        script_sha256=file_sha(Path(__file__)),audit=dict(raw_frames_validated=validated,valid_probe_positions=84,
            unique_records=len(rows),all_source_hashes_match=True,matched_inputs_and_wiring=True),
        limitations=parent['limitations']))
    figures(out,rows,summary,stages['records'],a.source)
    for r in summary:
        if r['world']=='all' and r['kind']=='offroute' and r['grid']=='grid5':
            print(r['condition'],r['inward'],r['n'],'mean heading error',round(r['heading_error_mean'],2))


if __name__=='__main__':main()
