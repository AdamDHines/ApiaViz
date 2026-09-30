"""Full-suite audit, explicitly separating imposed kicks from commanded motion."""
import argparse
import json
from pathlib import Path

import numpy as np

from .active_navigation import segment_collision
from .controller_experiments import setup
from .collision_geometry import protocol_geometry
from .evaluation_safety import EvaluationSafety, safe_displacement, proposal_reason, inside_body
from .grassland_smoke import position_key
from .study import file_hash, write_json
from .visual_avoidance import wrap


def check_motion(start, detail, row, p, base, obstacles, scenario):
    """Merge completed translations and external kicks in simulation-time order.

    The final camera observation timestamps a completed translation; a kick at
    the same time occurs afterwards. Never connect a walking segment across it.
    """
    current = np.asarray(start, dtype=float)
    safety=EvaluationSafety(**row['evaluation_safety']) if row.get('evaluation_safety') else None
    if safety:
        h=np.deg2rad(base['headings'][0])
        delta=scenario['lateral']*np.array([-np.sin(h),np.cos(h)])
        np.testing.assert_allclose(current,np.asarray(base['route'][0])+delta,atol=1e-10,rtol=0)
        current,release=safe_displacement(base['route'][0],delta,
                                          base['world_bounds_m'],obstacles,safety)
        assert row['release']==release
        np.testing.assert_allclose(row['initial_position'],current,atol=1e-10,rtol=0)
        releases=[e for e in detail['events'] if e['kind']=='release']
        assert len(releases)==1 and all(releases[0][k]==v for k,v in release.items())
    distance = commanded = 0.
    kicks = [e for e in detail['events'] if e['kind'] == 'displacement']
    assert len(kicks) == int(row.get('disturbance_attempted',row['disturbance_applied']))
    assert len(kicks) <= int(bool(scenario['kick']))
    if safety:
        assert row['perturbation_adjusted']==bool(release['adjusted'] or any(e['adjusted'] for e in kicks))
        if not kicks:
            assert row['displacement'] is None and not row['disturbance_applied']
    timeline = [(m['time_s'], 0, m) for m in detail['microtrace']]
    timeline += [(e['time_s'], 1, e) for e in kicks]
    xmin, xmax, ymin, ymax = base['world_bounds_m']
    for _, kind, item in sorted(timeline, key=lambda x: (x[0], x[1])):
        after = np.asarray(item['position'])
        if kind:
            h = np.deg2rad(base['headings'][len(base['headings'])//2])
            delta = scenario['kick']*np.array([-np.sin(h), np.cos(h)])
            if safety:
                expected,placement=safe_displacement(current,delta,base['world_bounds_m'],obstacles,safety)
                assert all(item[k]==v for k,v in placement.items())
                assert row['displacement']==placement
                assert row['disturbance_applied']==(placement['applied_distance_m']>1e-12)
                delta=np.asarray(placement['displacement'])
            np.testing.assert_allclose(item['displacement'], delta, atol=1e-10, rtol=0)
            np.testing.assert_allclose(after-current, delta, atol=1e-10, rtol=0)
            assert abs(commanded-(p['evaluation']['kick_before_step']-1)*.1) < 1e-8
        else:
            assert not segment_collision(current, after, obstacles)
            assert (inside_body(after,base['world_bounds_m'],obstacles) if safety else
                    xmin <= after[0] <= xmax and ymin <= after[1] <= ymax)
            actual = float(np.linalg.norm(after-current))
            assert abs(actual-item['stride_m']) < 1e-10
            command = item.get('commanded_stride_m', actual)
            assert 0 < command <= p['avoidance_settings']['stride_m']+1e-10
            if item.get('blocked', False):
                proposed = current+command*np.array([np.cos(np.deg2rad(item['heading'])), np.sin(np.deg2rad(item['heading']))])
                assert actual == 0 and (proposal_reason(current,proposed,base['world_bounds_m'],obstacles)
                                       if safety else segment_collision(current,proposed,obstacles))
            else:
                assert abs(actual-command) < 1e-10
            commanded += command
            assert abs(commanded-item.get('commanded_path_m', commanded)) < 1e-8
            delta = actual*np.array([np.cos(np.deg2rad(item['heading'])), np.sin(np.deg2rad(item['heading']))])
            np.testing.assert_allclose(after-current, delta, atol=1e-10, rtol=0)
            distance += actual
            assert abs(distance-item['path_m']) < 1e-8
        current = after
    assert abs(distance-row['path_length_m']) < 1e-8
    assert abs(commanded-row.get('commanded_path_m', distance)) < 1e-8
    blocked = [e for e in detail['events'] if e['kind'] == 'blocked_proposal']
    assert len(blocked) == row.get('blocked_proposals', 0)
    completed_blocks = sum(m.get('blocked', False) for m in detail['microtrace'])
    assert len(blocked)-completed_blocks == int(row.get('termination') == 'rock_collision' and bool(blocked))
    for event in blocked:
        if safety:
            assert proposal_reason(event['position'],event['proposed_position'],base['world_bounds_m'],obstacles)==event['reason']
        else:
            assert segment_collision(event['position'], event['proposed_position'], obstacles)
        assert all(k in event for k in ('decision','visual_points','visual_point_ages_m','commanded_stride_m'))
        delta=event['commanded_stride_m']*np.array([np.cos(np.deg2rad(event['heading'])),np.sin(np.deg2rad(event['heading']))])
        np.testing.assert_allclose(np.asarray(event['proposed_position'])-event['position'],delta,atol=1e-10,rtol=0)
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
    total_moves = total_commands = total_blocked = total_views = comparisons = diverted = kicks = disturbed_pairs = 0
    for row in rows:
        assert row.get('evaluation_safety')==p.get('evaluation_safety')
        env = environments[row['world']]
        base = json.loads((env/'protocol.json').read_text())
        world = next(w for w in p['worlds'] if w['name'] == row['world'])
        obstacles = protocol_geometry(p, world, json.loads((env/'world.json').read_text()))
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
                   row.get('commanded_path_m', distance)/p['sensor_settings']['speed_m_s'] +
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
            moves = [m for m in detail['microtrace'] if begin+1e-9 < m.get('commanded_path_m', m['path_m']) <= end+1e-9]
            assert len(moves) == len(feedback['commands']) and len(moves) > 0
            for m, cmd in zip(moves, feedback['commands']):
                assert abs(wrap(m['heading']-cmd['heading'])) < 1e-8
                assert abs(m.get('commanded_stride_m', m['stride_m'])-cmd['distance_m']) < 1e-10
            vector = np.sum([m.get('commanded_stride_m', m['stride_m'])*np.array([np.cos(np.deg2rad(m['heading'])),
                             np.sin(np.deg2rad(m['heading']))]) for m in moves], axis=0)
            assert abs(np.linalg.norm(vector)-feedback['net_displacement_m']) < 1e-9
            assert abs(c['distance_m']-feedback['walked_m']) < 1e-9
            assert abs(c['distance_m']-.1) < 1e-8
            if feedback.get('direction_defined',True):
                assert abs(wrap(c['movement_heading']-feedback['net_heading'])) < 1e-8
            else:
                assert np.linalg.norm(vector)<=1e-10
                assert c['movement_heading'] is None and feedback['net_heading'] is None
            assert feedback['diverted'] == any(abs(wrap(m['heading']-feedback['requested_heading'])) > 1e-8 for m in moves)
            comparisons += 1
            diverted += feedback['diverted']
            disturbed_pairs += bool(row['disturbance_applied'] and d['locomotion_step'] == p['evaluation']['kick_before_step'])
        for event in camera+memory:
            path = env/'panoramas'/f"{position_key(event['position'])}.png"
            if str(path) not in image_hashes:
                image_hashes[str(path)] = file_hash(path)
        values = (row['training_images_sha256'], row['memory_views'])
        assert training.setdefault(row['world'], values) == values
        key = (row['world'], row['seed'], row['method'])
        values = (row['encoder_fingerprint'], row['memory_fingerprint'])
        assert circuits.setdefault(key, values) == values
        total_moves += sum(m['stride_m']>0 for m in detail['microtrace'])
        total_commands += len(detail['microtrace'])
        total_blocked += row.get('blocked_proposals',0)
        total_views += row['observations']
    result = dict(passed=True, complete=True, trials=len(rows), executed_substeps=total_moves,
        commanded_substeps=total_commands,blocked_proposals=total_blocked,
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
