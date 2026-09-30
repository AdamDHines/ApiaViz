"""Independent accounting/physics audit for the visual avoidance smoke."""
import argparse
import json
from pathlib import Path

import numpy as np

from .active_navigation import segment_collision
from .grassland_smoke import position_key
from .study import file_hash, write_json


def audit(out):
    p = json.loads((out/'protocol.json').read_text())
    manifest = json.loads((out/'manifest.json').read_text())
    assert file_hash(out/'protocol.json') == manifest['protocol_sha256']
    for name, digest in manifest['source_sha256'].items():
        assert file_hash(Path(name)) == digest, name
    rows = [json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    environments = {w['name']:Path(w['environment']) for w in p['worlds']}
    total_moves = total_views = total_resets = 0
    image_hashes = {}
    for row in rows:
        env = environments[row['world']]
        assert file_hash(env/'grassland.blend') == p['scene_sha256'][row['world']]
        rocks = json.loads((env/'world.json').read_text())['obstacles']
        base = json.loads((env/'protocol.json').read_text())
        scenario = next(s for s in p['scenarios'] if s['name'] == row['scenario'])
        a = np.deg2rad(base['headings'][0])
        start = np.array(base['route'][0])+scenario['lateral']*np.array([-np.sin(a),np.cos(a)])
        d = json.loads((out/row['trace']).read_text())
        # Smoke scenarios have no imposed teleport. Audit those separately if a
        # future protocol adds displacements; never join across a teleport.
        assert not scenario['kick']
        points = [start]+[np.array(t['position']) for t in d['microtrace']]
        distance = 0.
        for before, after, move in zip(points[:-1],points[1:],d['microtrace']):
            assert not segment_collision(before, after, rocks)
            actual = float(np.linalg.norm(after-before))
            assert abs(actual-move['stride_m']) < 1e-10
            assert 0 < actual <= p['avoidance_settings']['stride_m']+1e-10
            distance += actual
        assert abs(distance-row['path_length_m']) < 1e-8
        camera = [e for e in d['events'] if e['kind'] == 'avoidance_observation']
        memory = [e for e in d['events'] if e['kind'] == 'observation']
        assert len(camera) == row['avoidance_observations']
        assert len(camera)+len(memory) == row['observations']
        assert len(memory) == len(d['sensor'])
        elapsed = (sum(e['duration_s'] for e in d['events'] if e['kind']=='turn') +
                   distance/p['sensor_settings']['speed_m_s'] +
                   row['observations']*p['sensor_settings']['observation_s'])
        assert abs(elapsed-row['time_s']) < 1e-7
        resets = [e for e in d['events'] if e['kind'] == 'motor_state_reset']
        assert len(resets) == row['motor_state_resets']
        for dec in d['decisions']:
            if dec['step'] == 1:
                assert 'comparison' not in dec  # No stale pre-detour pairing.
            if 'comparison' in dec:
                assert abs(dec['comparison']['distance_m']-.1) < 1e-8
        for e in camera+memory:
            path = env/'panoramas'/f"{position_key(e['position'])}.png"
            if str(path) not in image_hashes:
                image_hashes[str(path)] = file_hash(path)
        total_moves += len(d['microtrace'])
        total_views += row['observations']
        total_resets += len(resets)
    result = dict(passed=True, trials=len(rows), complete=len(rows)==p['trials'],
        executed_substeps=total_moves, accounted_observations=total_views,
        motor_state_resets=total_resets, cached_panoramas=len(image_hashes),
        checks=['Frozen source/scene/protocol', 'No executed segment intersects collision geometry',
                'Actual displacement and path totals', 'All views and turns charged',
                'No pre-detour familiarity comparison reused', 'Every acquired image cached and hashed'])
    write_json(out/'audit.json', result)
    write_json(out/'camera_sha256.json', image_hashes)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/visual-avoidance-smoke-v2'))
    audit(parser.parse_args().output)
