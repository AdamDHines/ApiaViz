"""Ordered acquisition, displacement and active-sensing experiments.

Uses the saved grassland renderer and frozen spiking front ends. Each phase has
its own append-only results and source snapshot; completed trials can be resumed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from .grassland_resolution import ResolutionWorld, load_model
from .grassland_smoke import METHODS
from .mechanisms import polyline_distance
from .navigation import Scorer, evaluate_route
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json

DEFAULT = Path('apiaviz/output/navigation-regime')
ENVIRONMENT = Path('apiaviz/output/grassland-smoke')
BANKS = ['centre', 'spread_equal', 'corridor3', 'corridor9']


def bent_route(anchors):
    anchors = np.asarray(anchors, dtype=float)
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(anchors, axis=0), axis=1))]
    spline = CubicSpline(distances, anchors, bc_type='clamped')
    dense = spline(np.linspace(0, distances[-1], 10001))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    targets = np.arange(0, arc[-1], .1)
    points = np.column_stack([np.interp(targets, arc, dense[:, i]) for i in (0, 1)])
    d = np.diff(points, axis=0)
    headings = np.round(np.degrees(np.arctan2(d[:, 1], d[:, 0])) / 2) * 2
    steps = .1 * np.column_stack([np.cos(np.radians(headings)), np.sin(np.radians(headings))])
    positions = np.vstack([points[0], points[0] + np.cumsum(steps, axis=0)])
    return positions, headings


def bank_indices(stations, bank):
    grid = np.arange(stations * 9).reshape(stations, 9)
    if bank == 'centre': return grid[:, 4]
    if bank == 'spread_equal': return grid[np.arange(stations), np.arange(stations) % 9]
    if bank == 'corridor3': return grid[:, [0, 4, 8]].ravel()
    if bank == 'corridor9': return grid.ravel()
    raise ValueError(bank)


def prepare(out):
    if (out / 'protocol.json').exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    base = json.loads((ENVIRONMENT / 'protocol.json').read_text())
    worlds = [dict(name='meander', environment=str(ENVIRONMENT.resolve()), new=False)]
    for i, (name, anchors) in enumerate([
        ('bend', [[2.3, 8.0], [2.3, 6.0], [3.0, 4.8], [5.0, 4.7], [7.3, 4.7]]),
        ('hairpin', [[2.5, 8.0], [2.5, 6.0], [3.5, 4.8], [5.3, 4.8], [6.3, 6.0], [6.3, 8.0]])]):
        environment = out / name
        environment.mkdir()
        (environment / 'queue').mkdir()
        (environment / 'panoramas').mkdir()
        route, headings = bent_route(anchors)
        p = deepcopy(base)
        p.update(geometry_seed=20261001 + i, route=route.tolist(), headings=headings.tolist(),
                 route_length_m=len(headings) * .1,
                 route_generation='Fixed spline through declared anchors, .1 m steps and 2 degree heading grid.',
                 route_anchors=anchors, role='Predeclared bent-route navigation regime experiment.')
        write_json(environment / 'protocol.json', p)
        worlds.append(dict(name=name, environment=str(environment.resolve()), new=True))
    scenarios = [dict(name='aligned', lateral=0., heading=0., kick=0.),
                 dict(name='left50', lateral=.5, heading=0., kick=0.),
                 dict(name='right50', lateral=-.5, heading=0., kick=0.),
                 dict(name='yaw_left40', lateral=0., heading=40., kick=0.),
                 dict(name='yaw_right40', lateral=0., heading=-40., kick=0.),
                 dict(name='kick_left50', lateral=0., heading=0., kick=.5),
                 dict(name='kick_right50', lateral=0., heading=0., kick=-.5)]
    p = dict(role='Ordered exploratory experiments; no tuning on evaluated outcomes.',
             encoder_environment=str(ENVIRONMENT.resolve()), shape=[51, 199],
             methods=METHODS, modes=dict(linear_colour='angles', sobel_colour='angles', ardin_input='pixels'),
             wiring_seeds=[19, 31, 43], worlds=worlds,
             stage1=dict(world='meander', banks=BANKS, trials=36, max_steps=200,
                         equal_budget='One view per station: centre versus deterministic nine-offset cycle; neither claims natural acquisition.'),
             stage2=dict(bank='centre', scenarios=scenarios, trials=189, max_steps=200,
                         kick_before_step=36, recovery_radius_m=.1, recovery_consecutive_steps=3,
                         termination='Endpoint .2 m, 200 movement steps, field boundary or conservative rock collision.'),
             stage3=dict(bank='centre', scenarios=['aligned', 'left50', 'kick_right50'],
                         controllers=['exhaustive', 'active'], trials=162, max_steps=200,
                         speed_m_s=.1, yaw_speed_deg_s=180., observation_s=.05,
                         time_budget_s=360., observation_budget=2600,
                         active=dict(min_turn_deg=2., max_turn_deg=30., scan_threshold=.25,
                                     poor_views=3, scan_cooldown_steps=6, scan_offsets=[-60.,-30.,30.,60.]),
                         calibration='Positive: median leave-one-out teaching-code overlap. Negative: median overlap for +/-60 degree teaching rotations at up to 12 evenly spaced stations. Clip linear confidence to [0,1]; minimum range .05.',
                         timing='Both controllers pay for every observed view, physical yaw excursion and translation. No observations or turns past shared time/sensory budgets.',
                         limitations='Finite-rate piecewise turns and straight moves; no full body dynamics, optic-flow controller or continuous spiking state.'),
             primary_metrics=['arrival', 'whole_run_polyline_mean_m', 'recovery_path_m', 'recovery_time_s'],
             secondary_metrics=['historical_nearest_station_mean_m', 'observations', 'scan_bouts', 'time_s', 'path_length_m', 'termination'],
             statistics='Paired effects within scene/seed/scenario. Only three scene-route pairs; report descriptive effects, not significance from technical repeats.',
             scope='Stage 1 on the original world; stages 2 and 3 on it and two new independently seeded worlds, one bent route each. No sun change.',
             checkpoint_sha256={str(s): file_hash(ENVIRONMENT / f'encoder-{s}.pt') for s in (19,31,43)})
    for w in worlds:
        w['protocol_sha256'] = file_hash(Path(w['environment']) / 'protocol.json')
    write_json(out / 'protocol.json', p)
    print(json.dumps(dict(output=str(out),stage1=36,stage2=189,stage3=162,worlds=worlds),indent=2),flush=True)


def setup(out, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p = json.loads((out / 'protocol.json').read_text())
    for w in p['worlds']:
        assert file_hash(Path(w['environment']) / 'protocol.json') == w['protocol_sha256']
    for seed, digest in p['checkpoint_sha256'].items():
        assert file_hash(Path(p['encoder_environment']) / f'encoder-{seed}.pt') == digest
    destination = out / f'stage{stage}'
    destination.mkdir(exist_ok=True)
    manifest_path = destination / 'manifest.json'
    sources = {str(f): file_hash(f) for f in sorted(Path('apiaviz').rglob('*.py')) if 'output' not in f.parts and 'data' not in f.parts}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['protocol_sha256'] == file_hash(out / 'protocol.json')
        # A resumed phase must use exactly its archived implementation.
        for source, digest in manifest['source_sha256'].items():
            assert file_hash(Path(source)) == digest, source
    else:
        with zipfile.ZipFile(destination / 'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for f in sources: z.write(f, f)
        manifest = dict(status='running', protocol_sha256=file_hash(out / 'protocol.json'), source_sha256=sources)
        write_json(manifest_path, manifest)
    rows = [json.loads(line) for line in (destination / 'results.jsonl').read_text().splitlines()] if (destination / 'results.jsonl').exists() else []
    return p, destination, manifest, {r['id']: r for r in rows}


def save_trial(destination, row, trace, decisions):
    row['trace'] = row['id'] + '.json'
    write_json(destination / row['trace'], dict(trajectory=trace, decisions=decisions))
    with (destination / 'results.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps({k: row[k] for k in ('id', 'reached_nest', 'polyline_mean_m', 'steps', 'elapsed_s') if k in row}), flush=True)


def stage1(out):
    p, destination, manifest, completed = setup(out, 1)
    environment = Path(p['encoder_environment'])
    base = json.loads((environment / 'protocol.json').read_text())
    route, headings = np.array(base['route']), np.array(base['headings'])
    positions, yaw = acquisition_views(route, headings, viewpoints=9, width=.2)
    world = ResolutionWorld(environment, p['shape'])
    images = world.render(positions, yaw)
    write_json(destination / 'acquisition.json', dict(positions=positions.tolist(), headings=yaw.tolist(),
               bank_indices={b: bank_indices(len(headings), b).tolist() for b in BANKS},
               image_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest()))
    for seed in p['wiring_seeds']:
        for method in p['methods']:
            model = load_model(environment, seed, method, p['modes'][method])
            frozen = fingerprint(model)
            codes = encode(model, images)
            for bank in p['stage1']['banks']:
                key = f'meander-{seed}-{method}-{bank}'
                if key in completed: continue
                start = time.monotonic()
                memory = SpikeOverlapMemory(codes[bank_indices(len(headings), bank)])
                memory_hash = fingerprint(memory)
                scorer = Scorer(world, model, memory, encode, 'clean', 0., base['geometry_seed'])
                result = evaluate_route(route, headings, scorer, 'free', max_steps=200)
                trace = result.pop('trace')
                assert fingerprint(model) == frozen and fingerprint(memory) == memory_hash
                row = dict(id=key, world='meander', seed=seed, method=method, bank=bank,
                           memory_views=len(memory.memory), encoder_fingerprint=frozen,
                           memory_fingerprint=memory_hash, **result,
                           polyline_mean_m=float(polyline_distance([t['position'] for t in trace], route).mean()),
                           elapsed_s=time.monotonic() - start)
                save_trial(destination, row, trace, scorer.decisions)
                completed[key] = row
    assert len(completed) == p['stage1']['trials']
    manifest.update(status='complete', trials=len(completed))
    write_json(destination / 'manifest.json', manifest)


def stage_navigation(out, stage):
    # Execution order is enforced, including on resume.
    previous = json.loads((out / f'stage{stage-1}' / 'manifest.json').read_text())
    assert previous['status'] == 'complete'
    from .active_navigation import Active, Exhaustive, Straight, calibrate, evaluate
    p, destination, manifest, completed = setup(out, stage)
    settings = p['stage3']
    for spec in p['worlds']:
        environment = Path(spec['environment'])
        base = json.loads((environment / 'protocol.json').read_text())
        metadata = json.loads((environment / 'world.json').read_text())
        assert metadata['scene_sha256'] == file_hash(environment / 'grassland.blend')
        manifest.setdefault('scene_sha256', {})[spec['name']] = metadata['scene_sha256']
        write_json(destination / 'manifest.json', manifest)
        route, headings = np.array(base['route']), np.array(base['headings'])
        positions, yaw = acquisition_views(route, headings, viewpoints=1, width=0.)
        world = ResolutionWorld(environment, p['shape'])
        images = world.render(positions, yaw)
        image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        write_json(destination / f"{spec['name']}-acquisition.json", dict(
            positions=positions.tolist(), headings=yaw.tolist(), image_sha256=image_hash))
        scenarios = p['stage2']['scenarios']
        if stage == 3:
            scenarios = [s for s in scenarios if s['name'] in p['stage3']['scenarios']]
        # Geometry-only null controls have no checkpoint seed and are stored
        # separately, never counted as additional biological repeats.
        if stage == 2:
            nulls = []
            for scenario in scenarios:
                result = evaluate(route, headings, None, Straight(), scenario, settings,
                                  p['stage2'], base['world_bounds_m'], metadata['obstacles'], timed=False)
                result.pop('events')
                nulls.append(dict(scenario=scenario['name'], **result))
            write_json(destination / f"{spec['name']}-no-vision.json", nulls)
            if spec['new']:
                assert not nulls[0]['reached_nest'], 'New routes must defeat straight walking without reference to model outcomes'
        for seed in p['wiring_seeds']:
            for method in p['methods']:
                model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
                frozen = fingerprint(model)
                codes = encode(model, images)
                memory = SpikeOverlapMemory(codes)
                memory_hash = fingerprint(memory)
                calibration = calibrate(model, memory, codes, world, positions, yaw) if stage == 3 else None
                if calibration is not None:
                    write_json(destination / f"{spec['name']}-{seed}-{method}-calibration.json", calibration)
                controllers = p['stage3']['controllers'] if stage == 3 else ['exhaustive']
                for scenario in scenarios:
                    for controller_name in controllers:
                        key = f"{spec['name']}-{seed}-{method}-{scenario['name']}-{controller_name}"
                        if key in completed: continue
                        start = time.monotonic()
                        scorer = Scorer(world, model, memory, encode, 'clean', 0., base['geometry_seed'])
                        controller = Exhaustive() if controller_name == 'exhaustive' else Active(settings['active'], calibration)
                        result = evaluate(route, headings, scorer, controller, scenario, settings,
                                          p['stage2'], base['world_bounds_m'], metadata['obstacles'], timed=stage == 3)
                        trace, events = result.pop('trace'), result.pop('events')
                        assert fingerprint(model) == frozen and fingerprint(memory) == memory_hash
                        row = dict(id=key, world=spec['name'], seed=seed, method=method,
                                   scenario=scenario['name'], controller=controller_name, bank='centre',
                                   memory_views=len(codes), training_images_sha256=image_hash,
                                   encoder_fingerprint=frozen, memory_fingerprint=memory_hash,
                                   **result, elapsed_s=time.monotonic() - start)
                        save_trial(destination, row, trace, dict(sensor=scorer.decisions, events=events))
                        completed[key] = row
    assert len(completed) == p[f'stage{stage}']['trials']
    manifest.update(status='complete', trials=len(completed))
    write_json(destination / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'stage1', 'stage2', 'stage3'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.output)
    elif args.action == 'stage1': stage1(args.output)
    else: stage_navigation(args.output, int(args.action[-1]))


if __name__ == '__main__': main()
