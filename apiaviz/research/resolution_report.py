"""Cache-only reproducibility audit and figures for the grassland resolution sweep."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .grassland_resolution import DEFAULT, ResolutionWorld, load_model
from .grassland_smoke import position_key, sample_panorama
from .grassland_report import LABELS, COLOURS
from .navigation import Scorer, evaluate_route
from .mechanisms import polyline_distance
from .spike_overlap import SpikeOverlapMemory
from .study import encode, file_hash, fingerprint, write_json


def read(out):
    protocol=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    return protocol,manifest,rows


def audit(out):
    started=time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol,manifest,rows=read(out)
    assert manifest['status']=='complete'
    assert file_hash(out/'protocol.json')==manifest['protocol_sha256']
    environment=Path(protocol['environment'])
    assert file_hash(environment/'grassland.blend')==protocol['scene_sha256']
    assert file_hash(environment/'protocol.json')==protocol['base_protocol_sha256']
    assert file_hash(environment/'acquisition.json')==protocol['acquisition_sha256']
    base=json.loads((environment/'protocol.json').read_text())
    acquisition=json.loads((environment/'acquisition.json').read_text())
    positions,headings=np.asarray(base['route']),np.asarray(base['headings'])
    expected={(f"{c['shape'][1]}x{c['shape'][0]}",c['mode'],s,m)
              for c in protocol['conditions'] for s in protocol['wiring_seeds'] for m in c['methods']}
    assert len(rows)==len(expected)==protocol['trials']
    assert {(r['resolution'],r['mode'],r['seed'],r['preprocessing']) for r in rows}==expected
    cache_positions={position_key(p):p for p in acquisition['positions']}
    teaching={}
    cardinal_calibration=[]
    with Image.open(environment/'calibration.png') as image:
        compass=np.asarray(image.convert('RGB'))
    reference={(r['seed'],r['preprocessing']):r for r in map(json.loads,(environment/'results.jsonl').read_text().splitlines())}
    for resolution,info in manifest['training'].items():
        assert file_hash(out/f'training-{resolution}.pt')==info['file_sha256']
        images=torch.load(out/f'training-{resolution}.pt',weights_only=True)['training']
        assert hashlib.sha256(images.numpy().tobytes()).hexdigest()==info['images_sha256']
        world=ResolutionWorld(environment,images.shape[-2:],cache_only=True)
        reconstructed=world.render(acquisition['positions'],acquisition['headings'])
        assert torch.equal(reconstructed,images)
        h,w=images.shape[-2:]
        views=sample_panorama(compass,[0,90,180,270],(h,w)).numpy()
        iy=int(np.argmin(abs(np.linspace(60,-15,h))))
        rgb=views[:,:,iy,w//2]
        assert np.argmax(rgb[:3],axis=1).tolist()==[0,1,2]
        assert min(rgb[3,:2])>rgb[3,2]+.2
        cardinal_calibration.append(dict(resolution=resolution,passed=True))
        teaching[resolution]=images
    count=0; max_physical_error=0.; reference_trials=0
    for row in rows:
        world=ResolutionWorld(environment,row['shape'],cache_only=True)
        model=load_model(environment,row['seed'],row['preprocessing'],row['mode'])
        metadata=manifest['encoders'][str(row['seed'])]
        assert file_hash(environment/f"encoder-{row['seed']}.pt")==metadata['checkpoint_sha256']
        assert fingerprint(model)==row['encoder_fingerprint']==metadata['fingerprint']
        images=teaching[row['resolution']]
        assert row['training_images_sha256']==manifest['training'][row['resolution']]['images_sha256']
        assert row['memory_views']==len(images)==693
        codes=encode(model,images)
        memory=SpikeOverlapMemory(codes)
        assert fingerprint(memory)==row['memory_fingerprint']
        scorer=Scorer(world,model,memory,encode,'clean',0.,base['geometry_seed'])
        replay=evaluate_route(positions,headings,scorer,'free',max_steps=base['max_steps'])
        saved=json.loads((out/row['trace']).read_text())
        assert replay.pop('trace')==saved['trajectory']
        assert scorer.decisions==saved['decisions']
        for key,value in replay.items(): assert value==row[key],key
        assert fingerprint(model)==row['encoder_fingerprint'] and fingerprint(memory)==row['memory_fingerprint']
        previous=positions[0]
        for step in saved['trajectory']:
            assert step['reset_to'] is None
            expected_position=previous+.1*np.array([np.cos(np.radians(step['heading'])),np.sin(np.radians(step['heading']))])
            max_physical_error=max(max_physical_error,float(np.abs(expected_position-step['position']).max()))
            previous=np.asarray(step['position'])
        errors=polyline_distance([r['position'] for r in saved['trajectory']],positions)
        assert float(errors.mean())==row['polyline_mean_m']
        for decision in scorer.decisions:
            cache_positions[position_key(decision['position'])]=decision['position']
        if row['resolution']=='74x18':
            old=reference[row['seed'],row['preprocessing']]
            assert saved==json.loads((environment/old['trace']).read_text())
            assert row['memory_fingerprint']==old['memory_fingerprint']
            reference_trials+=1
        count+=len(scorer.decisions)
        print(f"Exact replay: {row['resolution']} {row['mode']} {row['seed']} {row['preprocessing']}",flush=True)
    assert max_physical_error<1e-12
    cache_files=[]
    for key,position in sorted(cache_positions.items()):
        path=environment/'panoramas'/f'{key}.png'
        with Image.open(path) as image:
            image.load()
            assert image.size==(720,152)
        cache_files.append(dict(file=path.name,position=position,sha256=file_hash(path)))
    write_json(out/'panorama-manifest.json',dict(environment=str(environment),scene_sha256=protocol['scene_sha256'],files=cache_files))
    result=dict(status='passed',trials=len(rows),reference_trials_reproduced=reference_trials,
                exact_decisions=count,exact_scan_scores=13*count,teaching_views_per_resolution=693,
                unique_cached_positions=len(cache_files),cache_only_replay=True,
                shared_wiring_and_circuit=True,training_reconstructed_exactly=True,
                physical_cardinal_calibration=cardinal_calibration,
                movement_max_error_m=max_physical_error,elapsed_s=time.monotonic()-started)
    write_json(out/'audit.json',result)
    return result


def summarise(rows):
    groups=[]
    keys=sorted({(r['resolution'],r['mode'],r['preprocessing']) for r in rows})
    for resolution,mode,method in keys:
        subset=sorted([r for r in rows if (r['resolution'],r['mode'],r['preprocessing'])==(resolution,mode,method)],key=lambda r:r['seed'])
        values=[r['route_deviation_mean_m']*100 for r in subset]
        groups.append(dict(resolution=resolution,mode=mode,method=method,n=len(subset),
                           reached=sum(r['reached_nest'] for r in subset),mean_cm=float(np.mean(values)),
                           sd_cm=float(np.std(values,ddof=1)),per_seed_cm=values,
                           polyline_mean_cm=float(np.mean([r['polyline_mean_m']*100 for r in subset])),
                           active_fraction=float(np.mean([r['memory_active_fraction'] for r in subset]))))
    effects=[]
    for group in groups:
        if group['resolution']=='74x18': continue
        base=next(g for g in groups if g['resolution']=='74x18' and g['method']==group['method'])
        effects.append(dict(resolution=group['resolution'],mode=group['mode'],method=group['method'],
                            comparison='minus 74x18',mean_delta_cm=group['mean_cm']-base['mean_cm'],
                            paired_seed_deltas_cm=(np.array(group['per_seed_cm'])-base['per_seed_cm']).tolist()))
        if group['mode']=='angles':
            same=next(g for g in groups if g['resolution']==group['resolution'] and g['method']==group['method'] and g['mode']=='pixels')
            effects.append(dict(resolution=group['resolution'],mode=group['mode'],method=group['method'],
                                comparison='fixed angles minus fixed pixels',mean_delta_cm=group['mean_cm']-same['mean_cm'],
                                paired_seed_deltas_cm=(np.array(group['per_seed_cm'])-same['per_seed_cm']).tolist()))
    return dict(groups=groups,paired_effects=effects)


def report(out,destination):
    protocol,manifest,rows=read(out)
    summary=summarise(rows)
    groups=summary['groups']
    environment=Path(protocol['environment'])
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'.gitignore').write_text('!*.png\n')
    write_json(destination/'summary.json',summary)
    write_json(destination/'protocol.json',protocol)
    write_json(destination/'audit.json',json.loads((out/'audit.json').read_text()))
    fields=['resolution','mode','seed','preprocessing','reached_nest','route_deviation_mean_m','polyline_mean_m','route_deviation_max_m','final_nest_distance_m','steps','silent_scans','memory_active_fraction']
    with (destination/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    resolutions=['74x18','149x39','199x51']
    def group(res,mode,method):
        return next(g for g in groups if (g['resolution'],g['mode'],g['method'])==(res,mode,method))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,mode,title in zip(axes,('pixels','angles'),('Original filters in pixels','Filter offsets held fixed in degrees')):
        methods=list(LABELS) if mode=='pixels' else list(LABELS)[:2]
        for mi,method in enumerate(methods):
            data=[group(res,'pixels' if res=='74x18' else mode,method) for res in resolutions]
            ax.plot(range(3),[g['mean_cm'] for g in data],color=COLOURS[method],lw=2,marker='o',label=LABELS[method])
            for x,g in enumerate(data):
                ax.scatter(x+np.linspace(-.045,.045,3),g['per_seed_cm'],s=24,color=COLOURS[method],alpha=.5)
                lowest=min(group(resolutions[x],'pixels' if x==0 else mode,m)['mean_cm'] for m in methods)
                label_offset=-15 if g['mean_cm']==lowest else 8
                ax.annotate(f"{g['mean_cm']:.2f}",(x,g['mean_cm']),xytext=(0,label_offset),textcoords='offset points',ha='center',fontsize=8,color=COLOURS[method])
        ax.set_title(title,pad=16)
        ax.set_xticks(range(3),['74 × 18','149 × 39','199 × 51'])
        ax.set_xlabel('Input resolution (width × height)')
        ax.set_xlim(-.2,2.2)
        ax.grid(axis='y',alpha=.2)
        ax.legend(fontsize=8,loc='best')
    axes[0].set_ylabel('Mean route deviation (cm; lower is better)')
    axes[0].set_ylim(bottom=0)
    fig.suptitle('Grassland resolution comparison',weight='bold',fontsize=17)
    fig.text(.5,.02,'Lines: means of three wiring seeds. Small points: individual runs. One route, one world; no corrective resets.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94])
    fig.savefig(destination/'performance.png',dpi=180)
    plt.close(fig)
    # Equal display size and field of view; only the actual model sampling changes.
    fig,axes=plt.subplots(3,2,figsize=(12,8.5))
    for r,resolution in enumerate(resolutions):
        images=torch.load(out/f'training-{resolution}.pt',weights_only=True)['training']
        for c,index in enumerate((0,38)):
            axes[r,c].imshow(images[index*9+4].permute(1,2,0).numpy(),interpolation='nearest',aspect='auto')
            axes[r,c].set_xticks([]);axes[r,c].set_yticks([])
            axes[r,c].set_title(f'{resolution.replace("x"," × ")} · {index*.1:.1f} m along route',loc='left',fontsize=10)
    fig.suptitle('Same world and views, sampled at three resolutions',fontsize=16,weight='bold')
    fig.text(.5,.015,'Actual RGB input shown; every model uses G and B. No extra optical blur was added in this experiment.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    fig.savefig(destination/'inputs.png',dpi=160)
    plt.close(fig)
    primary=['| Input | ApiaViz linear colour | Sobel + colour | Ardin-style input |', '|---|---:|---:|---:|']
    control=['| Input | ApiaViz, fixed degrees | Sobel, fixed degrees |', '|---|---:|---:|']
    for res in resolutions:
        cells=[f"{group(res,'pixels',method)['mean_cm']:.2f} cm ({group(res,'pixels',method)['reached']}/3)" for method in LABELS]
        primary.append(f"| {res.replace('x',' × ')} | {' | '.join(cells)} |")
        mode='pixels' if res=='74x18' else 'angles'
        cells=[f"{group(res,mode,method)['mean_cm']:.2f} cm ({group(res,mode,method)['reached']}/3)" for method in list(LABELS)[:2]]
        control.append(f"| {res.replace('x',' × ')} | {' | '.join(cells)} |")
    complete=sum(r['reached_nest'] for r in rows)
    apia_base=group('74x18','pixels','linear_colour')
    apia_fine=group('199x51','angles','linear_colour')
    reduction=100*(1-apia_fine['mean_cm']/apia_base['mean_cm'])
    improved=int(np.sum(np.array(apia_fine['per_seed_cm'])<apia_base['per_seed_cm']))
    a=json.loads((out/'audit.json').read_text())
    text=f'''# Grassland input-resolution comparison

**Follow-up qualification (29 September 2026):** a no-vision agent that keeps its
starting heading also reaches the endpoint on this route, with 26.80 cm mean
deviation. Arrival alone is therefore not evidence of visual guidance here;
the much smaller deviations below remain informative. See the
[training and movement audit](../navigation-regime/README.md).

**{complete}/{len(rows)} runs reached the nest.** The experiment uses the saved
10.4 × 10.3 m grassland and its 7.7 m taught route. All three front ends received
the same teaching poses and shared wiring seeds (19, 31, 43). Only sensory sampling
and the explicitly identified spatial-filter control varied. The
[protocol](protocol.json) was saved before the new navigation outcomes were inspected.

![Resolution comparison](performance.png)

## Main finding

Finer input helped ApiaViz when its spatial filter offsets were retained in
degrees. At 199 × 51, that condition gave **{apia_fine['mean_cm']:.2f} cm** mean
deviation, compared with **{apia_base['mean_cm']:.2f} cm** at 74 × 18: a descriptive
reduction of **{reduction:.1f}%**, with improvement in **{improved}/3 seeds**.
Distance to the continuous route line also improved, from
{apia_base['polyline_mean_cm']:.2f} to {apia_fine['polyline_mean_cm']:.2f} cm.

Simply raising ApiaViz's input resolution with the original pixel-sized kernels
gave {group('149x39','pixels','linear_colour')['mean_cm']:.2f} cm at 149 × 39 and
{group('199x51','pixels','linear_colour')['mean_cm']:.2f} cm at 199 × 51. These results
support treating sampling density and filter scale as separate design choices.
The angular control changes several spatial stages together; it does not identify
one stage as the cause, and includes a numerical interpolation approximation.

The strongest Sobel result in this grid was
{min(g['mean_cm'] for g in groups if g['method']=='sobel_colour'):.2f} cm. Thus the
best tested ApiaViz setting has a small numerical advantage on this route, not
evidence of a large or general superiority. Comparing the best settings after
examining the grid is exploratory. Every method reached the nest in every run.

## Input-only change: original filter sizes in pixels

The numbers below are mean route deviation across three seeds; parentheses give
nest arrivals. Lower deviation is better. Every step of a failed run is included.

{chr(10).join(primary)}

At 74 × 18 the adjacent ray centres are 4.055° horizontally and 4.412° vertically.
At 149 × 39 they are 2.000° and 1.974°; at 199 × 51, 1.495° and 1.500°.
This condition uses the existing frontend code without altering its pixel-sized
filters. Thus finer sampling also makes each filter cover a smaller visual angle.

## Filter-scale control

{chr(10).join(control)}

This control preserves the original spatial tap offsets **in visual degrees**
for ApiaViz's hex neighbourhood, adaptation window and contrast bank, and for
Sobel's Gaussian and derivative filters. Original tap weights, gains and
nonlinearities are retained. Fractional pixel offsets use bilinear interpolation;
the hex row phase is anchored to the reference angular rows. Reflected boundaries
are retained. This is a numerical control on the discrete operators, not a newly
calibrated biological eye. Its interpolation and staggered-grid approximation
should be considered when interpreting differences.

The 74 × 18 reference is shared between tables, so there are **39 unique trials**:
27 input-resolution trials plus 12 higher-resolution angular controls. Ardin-style
preprocessing retains its original 10 × 36 reduction and default native-resolution
CLAHE. It has no separately altered spatial-filter condition here. It is the
adapted preprocessing baseline, not the complete original Ardin spiking model.

![Actual input grids](inputs.png)

## What remained fixed

Each run stored spike codes for the same **693 teaching views**, at 77 route
stations spaced 10 cm apart, with nine lateral views per station over ±20 cm.
The memory was rebuilt from the corresponding resolution's images for every
method. The visual backbone and projection wiring were never trained.

All methods retained 8 × 64 pooling per feature plane, 8,000 spiking cells, the
same saved projection checkpoint within seed, and the previous LIF/inhibition and
normalized spike-overlap memory settings. Navigation used 13 scan headings over
±60°, 10 cm steps, a 20 cm endpoint radius and a 200-step limit, with no corrective
resets or online memory updates. The controller receives no route coordinates.

The renderer, lighting, eye height and field of view were unchanged. Images were
sampled from the same 720 × 152 panoramas, and only newly visited positions were
rendered. No receptor acceptance-angle blur was added: this is a sampling and
filter-scale experiment, **not a full simulation of honeybee retinal optics**.

## Reproducibility and interpretation limits

All nine 74 × 18 runs reproduced the previous smoke-test memories, trajectories
and scan scores exactly. An independent cache-only replay reproduced all
**{a['trials']} runs, {a['exact_decisions']} decisions and {a['exact_scan_scores']} scan scores**.
Teaching images at all resolutions were reconstructed exactly from the cache.
Analytic direction fields, physical compass markers from the smoke test and a
direct fractional-sampling reference validate the renderer mapping and angular
filter implementation. See [audit.json](audit.json).

The primary deviation is distance to the nearest taught route point. The
[results CSV](results.csv) also includes the secondary distance to the continuous
route line, maximum deviation, final nest distance, steps and sparsity.
[summary.json](summary.json) gives individual-seed results and paired changes
against 74 × 18 and between the two filter-scale conditions. These are descriptive
effects, not significance tests: three wiring seeds on one previously inspected
route cannot establish a terrain-wide ranking or a biological optimum.
The 199 × 51 Ardin-style result is especially seed-sensitive: one seed had
9.28 cm mean deviation while the others had 2.22 and 2.99 cm. Its aggregate should
not be read as a uniform loss of accuracy at finer resolution.

The 10° steering increments and the common 8 × 64 feature pooling still limit the
system, even with finer input. A benefit at higher resolution would not isolate
retinal acuity unless optical blur and receptive-field scaling were also addressed;
absence of a benefit would not establish that the honeybee-resolution concern was
mistaken. No settings were selected or changed after seeing these results.

## Files and rerunning

Run data: [`apiaviz/output/grassland-resolution`](../../apiaviz/output/grassland-resolution).
Reusable scene and panorama cache:
[`apiaviz/output/grassland-smoke`](../../apiaviz/output/grassland-smoke).
The run directory contains three teaching banks, every decision trace, a source
archive, checkpoint/image/scene hashes, and a manifest of all cached positions
used by these runs. No original model or evaluation default was changed.

```sh
# In one terminal: resume the saved scene; render only cache misses.
blender --background --factory-startup --python apiaviz/research/grassland_world.py -- \\
  --output apiaviz/output/grassland-smoke --resume

# In another: use a new output directory for a fresh run.
.pixi/envs/default/bin/python -m apiaviz.research.grassland_resolution prepare \\
  --output apiaviz/output/grassland-resolution-repeat
.pixi/envs/default/bin/python -m apiaviz.research.grassland_resolution run \\
  --output apiaviz/output/grassland-resolution-repeat

# Replay and regenerate this report from cached views, without Blender.
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python -m apiaviz.research.resolution_report
```
'''
    (destination/'README.md').write_text(text)
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--docs',type=Path,default=Path('docs/grassland-resolution'))
    parser.add_argument('--skip-audit',action='store_true')
    args=parser.parse_args()
    if not args.skip_audit: audit(args.output)
    report(args.output,args.docs)


if __name__=='__main__': main()
