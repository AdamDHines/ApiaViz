"""Read-only check of conservative collision discs against exported rock geometry.

Uses the archived meander mesh, not an additional renderer or controller input.
The final proposed stride was not saved: subtract the maximum 2 cm stride to
obtain a conservative lower bound on clearance to each implicated rock.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]


def main():
    world = json.loads((ROOT/'apiaviz/output/grassland-smoke/world.json').read_text())
    geometry = ROOT/'apiaviz/output/uv-mitsuba/geometry'
    metadata = json.loads((geometry/'geometry.json').read_text())
    assert metadata['scene_sha256'] == world['scene_sha256']
    item = next(m for m in metadata['meshes'] if m['material'] == 'Limestone')
    mesh = np.load(geometry/item['file'])
    vertices, faces = mesh['vertices'][:,:2], mesh['faces']
    a, b = faces.ravel(), np.roll(faces,1,axis=1).ravel()
    n, labels = connected_components(coo_matrix((np.ones(len(a)),(a,b)),
        shape=(len(vertices),len(vertices))), directed=False)
    centres = np.array([[r['x'],r['y']] for r in world['obstacles']])
    radii = np.array([r['conservative_radius_m'] for r in world['obstacles']])
    assert n == len(centres)
    means = np.array([vertices[labels==i].mean(0) for i in range(n)])
    component, rock = linear_sum_assignment(np.linalg.norm(means[:,None]-centres[None],axis=2))
    mismatch = np.linalg.norm(means[component]-centres[rock],axis=1)/radii[rock]
    assert mismatch.max() < .01
    hulls = {int(k):ConvexHull(vertices[labels==i]) for i,k in zip(component,rock)}
    folder = ROOT/'apiaviz/output/route-continuous-full/worlds/meander'
    results = [json.loads(l) for l in (folder/'results.jsonl').read_text().splitlines() if l.strip()]
    rows = []
    for result in results:
        if result['termination'] != 'rock_collision':
            continue
        trace = json.loads((folder/result['trace']).read_text())
        observation = next(e for e in reversed(trace['events']) if e['kind']=='avoidance_observation')
        # Use the final camera position, which also handles a just-applied kick.
        position = np.array(observation['position'])
        heading = np.deg2rad(observation['heading'])
        delta = .02*np.array([np.cos(heading),np.sin(heading)])
        t = np.clip(((centres-position)@delta)/(delta@delta),0,1)
        collision = np.linalg.norm(centres-(position+t[:,None]*delta),axis=1)<=radii
        assert collision.any(), result['id']
        candidates = []
        for k in np.flatnonzero(collision):
            hull = hulls[k]
            polygon = hull.points[hull.vertices]
            edge = np.roll(polygon,-1,axis=0)-polygon
            t = np.clip(np.sum((position-polygon)*edge,axis=1)/np.sum(edge*edge,axis=1),0,1)
            gap = np.linalg.norm(position-(polygon+t[:,None]*edge),axis=1).min()
            if np.all(hull.equations[:,:2]@position+hull.equations[:,2]<=0):
                gap = 0.
            candidates.append(dict(rock_index=int(k),disc_radius_m=float(radii[k]),
                current_projected_hull_clearance_m=float(gap),
                clearance_lower_bound_after_max_stride_m=float(max(0.,gap-.02))))
        rows.append(dict(trial=result['id'],position=position.tolist(),heading_deg=observation['heading'],
            executed_avoidance_substeps=sum(m['state']=='avoid' for m in trace['microtrace']),
            candidates=candidates))
    out=ROOT/'docs/rock-collision-audit'
    out.mkdir(exist_ok=True)
    report=dict(world='meander',scene_sha256=world['scene_sha256'],
        maximum_stride_m=.02,reflex_clearance_setting_m=.045,
        component_centre_relative_error_max=float(mismatch.max()),trials=rows,
        caveat='2D convex hull covers the complete projected rock, including vertices below ground. This is a conservative geometric comparison, not a rerun. Final proposed stride/decision was not logged. Candidate discs are intersected within the maximum 2 cm step. No new controller outcome is inferred.')
    (out/'meander-clearance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
