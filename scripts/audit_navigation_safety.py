"""Read archived outcomes and verify safe placements against an exported scene.

No rendering or navigation trials; sampled vectors are evaluator geometry checks.
Requires a fresh output directory. Historical assets are read only.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apiaviz.research.collision_geometry import RockGeometry, sha256
from apiaviz.research.evaluation_safety import EvaluationSafety, proposal_reason, safe_displacement


def run(environment, output):
    if output.exists():
        raise FileExistsError('Choose a fresh versioned audit output')
    base=json.loads((environment/'protocol.json').read_text())
    scene=environment/'grassland.blend'
    geometry=RockGeometry.load(environment/'collision.json',sha256(scene),.005)
    settings=EvaluationSafety()
    archived=sorted((ROOT/'docs/route-continuous-full/archive').glob('*/results.jsonl'))
    rows=[json.loads(line) for path in archived for line in path.read_text().splitlines()]
    records=[]
    invalid_anchors=0
    # Prespecified grid and 8 directions, all with a requested 50 cm kick.
    for x in np.linspace(-.5,2.,6):
        for y in np.linspace(-1.,1.,5):
            origin=np.array([x,y])
            if proposal_reason(origin,origin,base['world_bounds_m'],geometry):
                invalid_anchors+=1
                continue
            for angle in np.arange(0.,360.,45.):
                delta=.5*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
                position,record=safe_displacement(origin,delta,base['world_bounds_m'],geometry,settings)
                assert proposal_reason(origin,position,base['world_bounds_m'],geometry) is None
                np.testing.assert_allclose(position-origin,record['fraction']*delta,atol=1e-12)
                assert 0<=record['fraction']<=1
                records.append(record)
    examples=[]
    for name,origin,delta in (
        ('endpoint_inside_rock',[.65,-.24],[0,.25]),
        ('endpoint_clear_but_sweep_crosses_rock',[.65,-.24],[0,.5]),
        ('unobstructed',[0,-.24],[0,.5]),
        ('field_boundary',[2.,0],[.5,0])):
        position,record=safe_displacement(origin,delta,base['world_bounds_m'],geometry,settings)
        endpoint=np.asarray(origin)+delta
        endpoint_rock=geometry.intersects(endpoint,endpoint)
        if name=='endpoint_inside_rock': assert endpoint_rock and record['adjusted']
        if name=='endpoint_clear_but_sweep_crosses_rock': assert not endpoint_rock and record['adjusted']
        if name=='unobstructed': assert not record['adjusted']
        examples.append(dict(name=name,requested_endpoint_in_rock=endpoint_rock,**record))
    summary=dict(schema=settings.schema,kind='Placement geometry audit; not navigation outcomes',
        archived_completed_trials=len(rows),archived_terminations=dict(Counter(r['termination'] for r in rows)),
        geometry=geometry.provenance,body_radius_m=geometry.body_radius_m,
        sampled_displacements=len(records),invalid_grid_anchors_excluded=invalid_anchors,
        adjusted=sum(r['adjusted'] for r in records),
        limiting_constraints=dict(Counter(r['constraint'] for r in records if r['adjusted'])),
        passed=True,examples=examples,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in archived+[
            Path(__file__).resolve(),ROOT/'apiaviz/research/evaluation_safety.py',
            ROOT/'apiaviz/research/collision_geometry.py',ROOT/'apiaviz/research/active_navigation.py']})
    output.mkdir(parents=True)
    for name,value in [('summary.json',summary),('placements.json',records)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.environment.resolve(),args.output.resolve())
