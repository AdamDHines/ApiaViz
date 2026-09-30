"""Full-suite audit, explicitly separating imposed kicks from commanded motion."""
import argparse
import json
from pathlib import Path

import numpy as np

from .active_navigation import segment_collision
from .controller_experiments import setup
from .grassland_smoke import position_key
from .study import file_hash, write_json
from .visual_avoidance import wrap


def check_motion(start, detail, row, p, base, obstacles, scenario):
    """Merge completed translations and external kicks in simulation-time order.

    The final camera observation timestamps a completed translation; a kick at
    the same time occurs afterwards. Never connect a walking segment across it.
    """
    current = np.asarray(start, dtype=float)
    distance = 0.
    kicks = [e for e in detail['events'] if e['kind'] == 'displacement']
    assert len(kicks) == int(row['disturbance_applied'])
    assert len(kicks) <= int(bool(scenario['kick']))
    timeline = [(m['time_s'], 0, m) for m in detail['microtrace']]
    timeline += [(e['time_s'], 1, e) for e in kicks]
    xmin, xmax, ymin, ymax = base['world_bounds_m']
    for _, kind, item in sorted(timeline, key=lambda x: (x[0], x[1])):
        after = np.asarray(item['position'])
        if kind:
            h = np.deg2rad(base['headings'][len(base['headings'])//2])
            delta = scenario['kick']*np.array([-np.sin(h), np.cos(h)])
            np.testing.assert_allclose(item['displacement'], delta, atol=1e-10, rtol=0)
            np.testing.assert_allclose(after-current, delta, atol=1e-10, rtol=0)
            assert abs(distance-(p['evaluation']['kick_before_step']-1)*.1) < 1e-8
        else:
            assert not segment_collision(current, after, obstacles)
            assert xmin <= after[0] <= xmax and ymin <= after[1] <= ymax
            actual = float(np.linalg.norm(after-current))
            assert abs(actual-item['stride_m']) < 1e-10
            assert 0 < actual <= p['avoidance_settings']['stride_m']+1e-10
            delta = actual*np.array([np.cos(np.deg2rad(item['heading'])), np.sin(np.deg2rad(item['heading']))])
            np.testing.assert_allclose(after-current, delta, atol=1e-10, rtol=0)
            distance += actual
            assert abs(distance-item['path_m']) < 1e-8
        current = after
    assert abs(distance-row['path_length_m']) < 1e-8
    assert abs(np.linalg.norm(current-np.array(base['route'][-1]))-row['final_nest_distance_m']) < 1e-8
    return distance, len(kicks)


def audit(out):
    from .route_full import cases, read_rows
    p, manifest, completed = setup(out)
    rows = read_rows(out)
    assert len(rows) == len(completed)
    expected = {c[0] for c in cases(p)}
    assert set(completed) == expected and manifest['status'] == 'complete'
    environments = {w['name']:Path(w['environment']) for w in p['worlds']}
    image_hashes, training, circuits = {}, {}, {}
    total_moves = total_views = comparisons = diverted = kicks = disturbed_pairs = 0
    for row in rows:
        env = environments[row['world']]
        base = json.loads((env/'protocol.json').read_text())
        obstacles = json.loads((env/'world.json').read_text())['obstacles']
        scenario = next(s for s in p['scenarios'] if s['name'] == row['scenario'])
        a = np.deg2rad(base['headings'][0])
        start = np.array(base['route'][0])+scenario['lateral']*np.array([-np.sin(a), np.cos(a)])
        detail = json.loads((out/row['trace']).read_text())
        distance, n_kicks = check_motion(start, detail, row, p, base, obstacles, scenario)
        kicks += n_kicks
        assert row['motor_state_resets'] == row['corrective_resets'] == 0
        assert row['phase'] in (-1, 1)
        camera = [e for e in detail['events'] if e['kind'] == 'avoidance_observation']
        memory = [e for e in detail['events'] if e['kind'] == 'observation']
        assert len(camera) == row['avoidance_observations']
        assert len(memory) == len(detail['sensor'])
        assert len(camera)+len(memory) == row['observations']
        elapsed = (sum(e['duration_s'] for e in detail['events'] if e['kind'] == 'turn') +
                   distance/p['sensor_settings']['speed_m_s'] +
                   row['observations']*p['sensor_settings']['observation_s'])
        assert abs(elapsed-row['time_s']) < 1e-7
        assert row['observations'] <= p['sensor_settings']['observation_budget']
        assert row['time_s'] <= p['sensor_settings']['time_budget_s']+1e-8
        for d in detail['decisions']:
            if 'comparison' not in d:
                continue
            c, feedback = d['comparison'], d['motor_feedback']
            end = d['self_motion_m']
            begin = end-c['distance_m']
            moves = [m for m in detail['microtrace'] if begin+1e-9 < m['path_m'] <= end+1e-9]
            assert len(moves) == len(feedback['commands']) and len(moves) > 0
            for m, cmd in zip(moves, feedback['commands']):
                assert abs(wrap(m['heading']-cmd['heading'])) < 1e-8
                assert abs(m['stride_m']-cmd['distance_m']) < 1e-10
            vector = np.sum([m['stride_m']*np.array([np.cos(np.deg2rad(m['heading'])),
                             np.sin(np.deg2rad(m['heading']))]) for m in moves], axis=0)
            assert abs(np.linalg.norm(vector)-feedback['net_displacement_m']) < 1e-9
            assert abs(c['distance_m']-feedback['walked_m']) < 1e-9
            assert abs(c['distance_m']-.1) < 1e-8
            assert abs(wrap(c['movement_heading']-feedback['net_heading'])) < 1e-8
            assert feedback['diverted'] == any(abs(wrap(m['heading']-feedback['requested_heading'])) > 1e-8 for m in moves)
            comparisons += 1
            diverted += feedback['diverted']
            disturbed_pairs += bool(n_kicks and d['locomotion_step'] == p['evaluation']['kick_before_step'])
        for event in camera+memory:
            path = env/'panoramas'/f"{position_key(event['position'])}.png"
            if str(path) not in image_hashes:
                image_hashes[str(path)] = file_hash(path)
        values = (row['training_images_sha256'], row['memory_views'])
        assert training.setdefault(row['world'], values) == values
        key = (row['world'], row['seed'], row['method'])
        values = (row['encoder_fingerprint'], row['memory_fingerprint'])
        assert circuits.setdefault(key, values) == values
        total_moves += len(detail['microtrace'])
        total_views += row['observations']
    result = dict(passed=True, complete=True, trials=len(rows), executed_substeps=total_moves,
        accounted_observations=total_views, motor_comparisons_checked=comparisons,
        diverted_comparisons_checked=diverted, imposed_kicks_checked=kicks,
        comparisons_spanning_external_kick=disturbed_pairs, cached_panoramas=len(image_hashes),
        checks=['Frozen source/protocol/scene/checkpoints and complete trial matrix',
                'No executed walking segment intersects collision geometry',
                'Walking distance excludes imposed displacements',
                'All observations, translations and turns charged',
                'Executed motor commands match familiarity accounting; kicks remain external',
                'No navigator resets or privileged geometry inputs added',
                'Same teaching images across frontends; fixed model/memory across conditions'])
    write_json(out/'camera_sha256.json', image_hashes)
    write_json(out/'audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
