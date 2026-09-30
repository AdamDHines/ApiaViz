"""Common rendered movement probes at taught stations and halfway between them.

Offline geometry labels diagnose the signal; they never enter navigation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .active_navigation import segment_collision
from .controller_experiments import REGIME, deterministic
from .familiarity_controller import acquisition_calibration
from .grassland_resolution import ResolutionWorld, load_model
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, write_json, file_hash


def run(out):
    deterministic()
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads((REGIME/'protocol.json').read_text())
    offsets=[-.5,-.4,-.2,-.1,0.,.1,.2,.4,.5]
    plan=dict(station_fractions=[.25,.5,.75],phases=['taught','between'],offsets_m=offsets,
        heading_offsets_deg=[-20,0,20],seeds=p['wiring_seeds'],worlds=p['worlds'],
        role='Common-pose development probe; no navigation parameters fitted to these results.',
        source_sha256=file_hash(Path(__file__)))
    write_json(out/'protocol.json',plan)
    rows,pairs=[],[]
    for spec in p['worlds']:
        env=Path(spec['environment'])
        base=json.loads((env/'protocol.json').read_text())
        rocks=json.loads((env/'world.json').read_text())['obstacles']
        bounds=base['world_bounds_m']
        route,heading=np.array(base['route']),np.array(base['headings'])
        stations=np.unique(np.round(np.array(plan['station_fractions'])*(len(heading)-1)).astype(int))
        queries=[]
        for i in stations:
            rad=np.radians(heading[i]);normal=np.array([-np.sin(rad),np.cos(rad)])
            for phase in plan['phases']:
                centre=route[i] if phase=='taught' else (route[i]+route[i+1])/2
                for offset in offsets:
                    point=centre+offset*normal
                    valid=(bounds[0]<=point[0]<=bounds[1] and bounds[2]<=point[1]<=bounds[3]
                           and not segment_collision(point,point,rocks))
                    queries.append(dict(station=int(i),phase=phase,offset_m=offset,
                        position=point.tolist(),heading=float(heading[i]),valid=bool(valid)))
        world=ResolutionWorld(env,p['shape'])
        teach_pos,teach_head=acquisition_views(route,heading,1,0.)
        training=world.render(teach_pos,teach_head)
        images=world.render([q['position'] for q in queries for h in plan['heading_offsets_deg']],
            [q['heading']+h for q in queries for h in plan['heading_offsets_deg']])
        for seed in plan['seeds']:
            for method in p['methods']:
                model=load_model(Path(p['encoder_environment']),seed,method,p['modes'][method])
                codes=encode(model,training);memory=SpikeOverlapMemory(codes)
                scale=acquisition_calibration(codes.numpy())['scale']
                scores=(-memory(encode(model,images))).detach().numpy().reshape(-1,3)
                local=[]
                for q,values in zip(queries,scores):
                    row=dict(world=spec['name'],seed=seed,method=method,**q,
                        familiarity=float(values[1]),heading_scores=values.tolist(),
                        directional_contrast=float((values.max()-np.median(values))/scale),scale=scale)
                    rows.append(row);local.append(row)
                lookup={(q['station'],q['phase'],q['offset_m']):q for q in local}
                for start in local:
                    offset=start['offset_m']
                    if round(abs(offset),2) not in (.1,.2,.5):continue
                    end=lookup[start['station'],start['phase'],round(offset-np.sign(offset)*.1,2)]
                    valid=start['valid'] and end['valid'] and not segment_collision(start['position'],end['position'],rocks)
                    pairs.append(dict(world=spec['name'],seed=seed,method=method,station=start['station'],
                        phase=start['phase'],from_offset_m=offset,valid=bool(valid),
                        change=end['familiarity']-start['familiarity'],
                        normalized_change=(end['familiarity']-start['familiarity'])/scale))
        print('Probed',spec['name'],flush=True)
    summary=[]
    for method in p['methods']:
        for phase in plan['phases']:
            for distance in (.1,.2,.5):
                selected=[r['normalized_change'] for r in pairs if r['method']==method and r['phase']==phase and abs(r['from_offset_m'])==distance and r['valid']]
                summary.append(dict(method=method,phase=phase,distance_m=distance,n=len(selected),
                    fraction_inward_positive=float(np.mean(np.array(selected)>0)) if selected else None,
                    median_change=float(np.median(selected)) if selected else None))
    write_json(out/'results.json',dict(rows=rows,pairs=pairs,summary=summary,
        caveat='Pairs share positions, worlds and wiring seeds; no significance claim. Three headings only measure local directional contrast, not the complete familiarity landscape.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/familiarity-controller/common-probe'))
    run(parser.parse_args().output)
