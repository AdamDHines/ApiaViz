"""Audit swept placements on the actual frozen trial worlds, without navigation.

Exercises field edges, teaching stations and points around every mesh bound.
This is a geometry regression check, not a claim about camera-policy success.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apiaviz.research.collision_geometry import RockGeometry, sha256
from apiaviz.research.evaluation_safety import proposal_reason, safe_displacement


def run(study, output):
    if output.exists():
        raise FileExistsError('Use a fresh audit output')
    protocol = json.loads((study / 'protocol.json').read_text())
    records, summaries = [], []
    radius = protocol['body_radius_m']
    for world in tqdm(protocol['worlds'], desc='Placement geometry'):
        env = study / 'worlds' / world['name']
        geometry = RockGeometry.load(env / 'collision.json', sha256(env / 'grassland.blend'), radius)
        bounds = world['world_bounds_m']
        x0, x1, y0, y1 = bounds
        anchors = [[x, y] for x in np.linspace(x0+radius, x1-radius, 7)
                   for y in np.linspace(y0+radius, y1-radius, 7)]
        anchors += world['route'][::5]
        for _, _, lo, hi in geometry.rocks:
            mid = (lo+hi)/2
            anchors += [[lo[0]-radius-1e-4, mid[1]], [hi[0]+radius+1e-4, mid[1]],
                        [mid[0], lo[1]-radius-1e-4], [mid[0], hi[1]+radius+1e-4]]
        start, invalid, crossing = len(records), 0, 0
        for origin in anchors:
            origin = np.asarray(origin)
            if proposal_reason(origin, origin, bounds, geometry):
                invalid += 1
                continue
            for length in (.02, .5):
                for angle in np.deg2rad(np.arange(0, 360, 45)):
                    delta = length*np.array([np.cos(angle), np.sin(angle)])
                    final, record = safe_displacement(origin, delta, bounds, geometry)
                    assert proposal_reason(origin, final, bounds, geometry) is None
                    np.testing.assert_allclose(final-origin, record['fraction']*delta, atol=1e-12)
                    assert 0 <= record['fraction'] <= 1
                    reason = proposal_reason(origin, origin+delta, bounds, geometry)
                    assert record['adjusted'] == bool(reason)
                    if not reason:
                        assert record['fraction'] == 1
                    endpoint_clear = proposal_reason(origin+delta, origin+delta, bounds, geometry) is None
                    crossing += bool(endpoint_clear and reason == 'rock_contact')
                    records.append(dict(world=world['name'], endpoint_clear=endpoint_clear, **record))
        subset = records[start:]
        constraints = Counter(r['constraint'] for r in subset if r['adjusted'])
        assert constraints['rock_contact'] and constraints['field_boundary'] and crossing
        summaries.append(dict(world=world['name'], tested=len(subset), invalid_anchors_excluded=invalid,
                              constraints=dict(constraints), clear_endpoints_with_blocked_sweeps=crossing,
                              geometry=geometry.provenance))
    summary = dict(passed=True, purpose='Evaluator geometry regression; no navigation or policy feedback',
                   study_protocol_sha256=sha256(study/'protocol.json'), worlds=summaries,
                   sources={str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__).resolve(),
                       ROOT/'apiaviz/research/collision_geometry.py', ROOT/'apiaviz/research/evaluation_safety.py',
                       ROOT/'apiaviz/research/active_navigation.py']})
    output.mkdir(parents=True)
    for name, data in [('summary.json', summary), ('placements.json', records)]:
        (output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
