"""Audit cached grassland runs and export a short, descriptive smoke-test report."""
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

from .frontend_refinements import RefinementEncoder
from .grassland_smoke import DEFAULT, GrasslandWorld, position_key, validate_calibration
from .navigation import Scorer, evaluate_route
from .mechanisms import polyline_distance
from .spike_overlap import SpikeOverlapMemory
from .study import encode, file_hash, fingerprint, write_json

LABELS={'linear_colour':'ApiaViz · linear colour','sobel_colour':'Sobel + colour','ardin_input':'Ardin-style input'}
COLOURS={'linear_colour':'#277d67','sobel_colour':'#d28232','ardin_input':'#6c73ad'}


class CacheOnlyWorld(GrasslandWorld):
    def obtain(self,positions):
        missing=[position_key(p) for p in positions if not (self.out/'panoramas'/f'{position_key(p)}.png').exists()]
        if missing: raise AssertionError(f'Cache miss in replay: {missing}')


def audit(out):
    started=time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='complete'
    protocol=json.loads((out/'protocol.json').read_text())
    assert file_hash(out/'protocol.json')==manifest['protocol_sha256']
    environment=Path(manifest['environment'])
    assert file_hash(environment/'grassland.blend')==manifest['scene_sha256']
    world=CacheOnlyWorld(environment)
    calibration=validate_calibration(environment)
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    expected={(s,m) for s in protocol['wiring_seeds'] for m in protocol['methods']}
    assert len(rows)==len(expected) and {(r['seed'],r['preprocessing']) for r in rows}==expected
    images=torch.load(out/'training.pt',weights_only=True)['training']
    image_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
    assert image_hash==manifest['training_images_sha256']
    acquisition=json.loads((out/'acquisition.json').read_text())
    rerender=world.render(acquisition['positions'],acquisition['headings'])
    assert torch.equal(images,rerender)
    positions,headings=np.asarray(protocol['route']),np.asarray(protocol['headings'])
    decisions=0
    cached_positions={position_key(p):p for p in acquisition['positions']}
    obstacle_footprints=json.loads((environment/'world.json').read_text())['obstacles']
    stone_intersections=0
    for row in rows:
        model=RefinementEncoder(seed=row['seed'],method=row['preprocessing']).eval()
        model.load_state_dict(torch.load(out/f"encoder-{row['seed']}.pt",weights_only=True)['state_dict'])
        assert fingerprint(model)==row['encoder_fingerprint']==manifest['encoders'][str(row['seed'])]['fingerprint']
        codes=encode(model,images)
        memory=SpikeOverlapMemory(codes)
        assert fingerprint(memory)==row['memory_fingerprint']
        assert row['training_images_sha256']==image_hash and row['memory_views']==len(images)
        scorer=Scorer(world,model,memory,encode,'clean',0.,protocol['geometry_seed'])
        result=evaluate_route(positions,headings,scorer,'free',max_steps=protocol['max_steps'])
        trace=json.loads((out/row['trace']).read_text())
        assert result.pop('trace')==trace['trajectory']
        assert scorer.decisions==trace['decisions']
        for record in scorer.decisions:
            p=record['position']
            cached_positions[position_key(p)]=p
            stone_intersections+=any(np.linalg.norm(np.asarray(p)-[o['x'],o['y']]) < o['conservative_radius_m'] for o in obstacle_footprints)
        for key,value in result.items(): assert value==row[key],key
        for record in trace['trajectory']: assert record['reset_to'] is None
        errors=polyline_distance([r['position'] for r in trace['trajectory']],positions)
        assert float(errors.mean())==row['polyline_mean_m']
        decisions+=len(scorer.decisions)
        print(f"Replayed {row['seed']} {row['preprocessing']}: exact",flush=True)
    # A reusable environment cache may contain positions from later experiments.
    # Audit this run's dependencies without assuming exclusive ownership of it.
    files=[environment/'panoramas'/f'{key}.png' for key in sorted(cached_positions)]
    cache_manifest=[]
    for path in files:
        with Image.open(path) as im:
            im.load()
            assert im.size==tuple(protocol['renderer']['panorama_size'])
        assert path.stem in cached_positions
        cache_manifest.append(dict(file=path.name,position=cached_positions[path.stem],sha256=file_hash(path)))
    write_json(environment/'panorama-manifest.json',dict(scene_sha256=manifest['scene_sha256'],files=cache_manifest))
    result=dict(status='passed',trials=len(rows),teaching_views=len(images),exact_replayed_decisions=decisions,
                exact_replayed_scan_scores=decisions*13,panoramas=len(files),calibration=calibration,
                cache_only_replay=True,training_images_identical=True,shared_wiring=True,
                no_corrective_resets=True,scan_positions_inside_conservative_stone_footprints=stone_intersections,
                elapsed_s=time.monotonic()-started)
    write_json(out/'audit.json',result)
    return result


def report(out,destination):
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'.gitignore').write_text('!*.png\n')
    protocol=json.loads((out/'protocol.json').read_text())
    manifest=json.loads((out/'manifest.json').read_text())
    environment=Path(manifest['environment'])
    metadata=json.loads((environment/'world.json').read_text())
    rows=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    route=np.asarray(protocol['route'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,5.8),sharex=True,sharey=True)
    for ax,method in zip(axes,protocol['methods']):
        ax.set_facecolor('#f5f1e7')
        overhead=destination/'world-overhead.png'
        if overhead.exists():
            ax.imshow(Image.open(overhead),extent=protocol['world_bounds_m'],alpha=.7)
        else:
            for obstacle in metadata['obstacles']:
                ax.add_patch(plt.Circle((obstacle['x'],obstacle['y']),obstacle['conservative_radius_m'],color='#c7c0ae',alpha=.5,lw=0))
        ax.plot(*route.T,color='#454545',ls='--',lw=1.7,label='Taught route')
        for index,seed in enumerate(protocol['wiring_seeds']):
            row=next(r for r in rows if r['seed']==seed and r['preprocessing']==method)
            trace=json.loads((out/row['trace']).read_text())['trajectory']
            path=np.vstack([route[0],[r['position'] for r in trace]])
            ax.plot(*path.T,color=COLOURS[method],lw=1.4,alpha=1-index*.22,
                    ls=['-','--',':'][index],label=f"Seed {seed} · {'home' if row['reached_nest'] else 'step limit'}")
            ax.scatter(*path[-1],s=24,color=COLOURS[method],marker='o' if row['reached_nest'] else 'x',zorder=5)
        ax.scatter(*route[0],s=55,marker='^',color='#202a32',zorder=6)
        ax.scatter(*route[-1],s=85,marker='*',color='#202a32',zorder=6)
        ax.set_title(LABELS[method],weight='bold',pad=12)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('Position (m)')
        ax.legend(loc='lower left',fontsize=8,framealpha=.9)
    axes[0].set_ylabel('Position (m)')
    fig.suptitle('Dry grassland · open-loop navigation',fontsize=18,weight='bold')
    fig.text(.5,.925,'One 7.7 m route · three paired wiring seeds · same teaching images, spiking circuit and memory',ha='center',color='#555555')
    fig.tight_layout(rect=[0,0,1,.91])
    fig.savefig(destination/'trajectories.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),sharey=True)
    for ax,seed in zip(axes,protocol['wiring_seeds']):
        for method in protocol['methods']:
            row=next(r for r in rows if r['seed']==seed and r['preprocessing']==method)
            trace=json.loads((out/row['trace']).read_text())['trajectory']
            ax.plot([r['step']*.1 for r in trace],[r['deviation_m']*100 for r in trace],
                    color=COLOURS[method],lw=1.3,label=LABELS[method])
        ax.set_title(f'Wiring seed {seed}')
        ax.set_xlabel('Distance walked (m)')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylim(0,max(r['route_deviation_max_m'] for r in rows)*110)
    axes[0].set_ylabel('Distance to nearest taught point (cm)')
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.32),ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.92])
    fig.savefig(destination/'deviation.png',dpi=170)
    plt.close(fig)
    images=torch.load(out/'training.pt',weights_only=True)['training'].numpy()
    indices=np.linspace(0,len(protocol['headings'])-1,4).astype(int)
    fig,axes=plt.subplots(4,1,figsize=(12,6.8))
    for ax,index in zip(axes,indices):
        image=images[index*9+4].transpose(1,2,0)
        ax.imshow(image,interpolation='nearest',aspect='auto')
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_title(f'Taught route: {index*.1:.1f} m from start',loc='left',fontsize=10)
    fig.suptitle('Actual model input · 74 × 18 RGB pixels',weight='bold',fontsize=16)
    fig.text(.5,.015,'296° horizontal field; elevation −15° to +60°. All three front ends use G and B; R is shown for viewing.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.94])
    fig.savefig(destination/'model-inputs.png',dpi=160)
    plt.close(fig)
    fields=['seed','preprocessing','reached_nest','route_deviation_mean_m','polyline_mean_m','final_nest_distance_m','path_length_m','steps','silent_scans','steps_outside_landmark_field']
    with (destination/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    table=['| Method | Reached nest | Mean route deviation | Per-seed deviation (19 / 31 / 43) |',
           '|---|---:|---:|---:|']
    summaries={}
    for method in protocol['methods']:
        group=[r for r in rows if r['preprocessing']==method]
        values=[r['route_deviation_mean_m']*100 for r in group]
        summaries[method]=dict(success=sum(r['reached_nest'] for r in group),mean_deviation_cm=float(np.mean(values)),
                               per_seed_deviation_cm=values,mean_polyline_cm=float(np.mean([r['polyline_mean_m']*100 for r in group])))
        table.append(f"| {LABELS[method]} | {summaries[method]['success']}/3 | {np.mean(values):.2f} cm | {' / '.join(f'{v:.2f}' for v in values)} cm |")
    write_json(destination/'summary.json',summaries)
    audit_result=json.loads((out/'audit.json').read_text())
    write_json(destination/'audit.json',audit_result)
    write_json(destination/'protocol.json',protocol)
    xmin,xmax,ymin,ymax=protocol['world_bounds_m']
    readme=f'''# Dry-grassland navigation smoke test

**Follow-up qualification (29 September 2026):** a no-vision agent that keeps its
starting heading also reaches the endpoint on this route, with 26.80 cm mean
deviation. Arrival alone is therefore not evidence of visual guidance here;
the much smaller deviations below remain informative. See the
[training and movement audit](../navigation-regime/README.md).

The generated grassland has the original landmark field's exact horizontal bounds:
**{xmax-xmin:.3f} × {ymax-ymin:.3f} m** (x: {xmin:.3f} to {xmax:.3f}; y: {ymin:.3f} to {ymax:.3f}).
It contains curved grass, leafy scrub and solid limestone rocks on mildly uneven
ground. A fixed sun at azimuth 35°, elevation 38° casts shadows. The 7.7 m route
and object placement were fixed before any navigation outcomes were examined.

![The evaluated world, viewed from the starting pose](grassland-view.png)

![All nine trajectories](trajectories.png)

{chr(10).join(table)}

Deviation is the whole-trajectory mean distance to the nearest taught route point,
including every step of failures, averaged equally across wiring seeds. It uses
the same definition as the previous study. Continuous-polyline distances and
all per-run outcomes are also in [results.csv](results.csv). Lower is better.

![Deviation during each run](deviation.png)

## What was held constant

All methods received the same **693 teaching images**: 77 route stations at
10 cm intervals, each with nine lateral views spanning ±20 cm. Route training
stores spike patterns in the existing normalized-overlap memory; it does not
optimize the visual backbone. Each seed used identical projection weights across
methods, 8,000 spiking cells, the existing inhibition rule and the same memory.

ApiaViz uses the selected **linear-colour** variant with its original form pathway.
The Sobel baseline uses smoothed luminance gradients plus simple opponent colour.
Ardin-style input uses inverted luminance, CLAHE, 10 × 36 downsampling and L2
normalization before the common encoder. This is a comparison of preprocessing,
not a reproduction of the complete Ardin 2016 neural network.

Navigation starts at the first route pose. At each 10 cm step the ant scans 13
headings over ±60°, chooses the most familiar image and moves. There are no
corrective resets, route coordinates in the steering input, or online memory
updates. The ant stops within 20 cm of the endpoint or after 200 steps. As in the
previous benchmark, movement is a 2D kinematic model with no obstacle avoidance.
The teaching corridor is kept free of substantial obstacles, without colouring
or marking a trail. Ground height is ray-cast from the mesh for a 10 mm eye height.

![Actual teaching inputs](model-inputs.png)

## Rendering and checks

Cycles renders a **720 × 152 RGB panorama** at each position, using a fixed seed,
32 samples, AgX colour rendering and no denoising. Every scanned heading is sampled
from that same panorama at the previous renderer's **74 × 18 endpoint-inclusive
ray grid** (296° horizontally, elevations +60° to −15°). All front ends use the
same G/B channels. This rendering adapter is new; it has been tested with physical
cardinal-direction markers and an analytic azimuth/elevation field.

The audit reconstructed all teaching inputs from cached panoramas, rebuilt all
memories and replayed all **{audit_result['trials']} trajectories**, reproducing
**{audit_result['exact_replayed_decisions']} decisions / {audit_result['exact_replayed_scan_scores']} scan scores exactly**.
This replay required no Blender rendering. See [audit.json](audit.json).

## Interpretation

All nine runs reached the nest in 76 steps (7.6 m walked, ending within the
20 cm stopping radius). ApiaViz and Sobel + colour were effectively tied on the
primary endpoint: 3.13 versus 3.16 cm. Ardin-style preprocessing had a higher mean
deviation of 4.73 cm. The secondary distance to the continuous route line was
2.00, 2.51 and 3.73 cm respectively; unlike the primary measure, it is not affected
by the 10 cm gaps between taught points.

This is an integration smoke test on **one route in one world**, with three
wiring seeds. It can reveal gross failures and seed sensitivity, but it cannot
establish a reliable ranking across natural terrain or statistical superiority.
No parameters or routes were changed in response to these results. Shadows and
sun direction were held fixed between teaching and recall. A change-of-lighting
experiment has not been run.

The rendered input is dominated by sky, with much of the vegetation compressed
into a few rows near the horizon at 74 × 18 resolution. That is worth examining
before interpreting a small performance gap: this route under fixed illumination
may simply be easy for all three methods. These runs do not isolate the
contributions of landmarks, shading and the directional sky gradient.
The [honeybee acuity note](acuity.md) compares this input grid with measured
receptor spacing. A [39-run resolution follow-up](../grassland-resolution/README.md)
now tests finer inputs and controls for angular filter scale.

## Saved environment and reuse

The scene, protocol and panorama cache are stored at
[`{environment.relative_to(Path.cwd()) if environment.is_relative_to(Path.cwd()) else environment}`](../../{environment.relative_to(Path.cwd()) if environment.is_relative_to(Path.cwd()) else environment}).
The directory includes `grassland.blend`, `world.json`, `protocol.json`,
`panoramas/` and `panorama-manifest.json`. Run artifacts include the training bank,
encoder checkpoints, decision traces, source archive and file hashes. These large
files are retained locally under the repository's ignored output directory.

To replay this smoke test in a new output directory using the saved images:

```sh
.pixi/envs/default/bin/python -m apiaviz.research.grassland_smoke run \\
  --environment {environment.relative_to(Path.cwd()) if environment.is_relative_to(Path.cwd()) else environment} \\
  --output apiaviz/output/grassland-replay
```

For a future run that visits new positions, start the renderer in a separate terminal:

```sh
blender --background --factory-startup --python apiaviz/research/grassland_world.py -- \\
  --output {environment.relative_to(Path.cwd()) if environment.is_relative_to(Path.cwd()) else environment} --resume
```

The renderer loads the saved geometry and fills only missing panorama entries.
The cache belongs to this exact scene and lighting configuration; a different sun
or terrain needs its own cache. Original worlds and evaluation defaults are unchanged.
'''
    (destination/'README.md').write_text(readme)
    print(json.dumps(summaries,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--docs',type=Path,default=Path('docs/grassland-smoke'))
    parser.add_argument('--skip-audit',action='store_true')
    args=parser.parse_args()
    if not args.skip_audit: audit(args.output)
    report(args.output,args.docs)


if __name__=='__main__': main()
