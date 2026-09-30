"""Reproducible development experiments for the familiarity controller.

No original experiment is overwritten. Prepare freezes a protocol; run resumes
only with matching source, scenes and checkpoints. A renderer worker is needed
for novel positions (or use --cache-only to fail immediately on a cache miss).
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .active_navigation import Active, Exhaustive, calibrate, evaluate
from .familiarity_controller import (Settings, FamiliarityController,
                                     ControllerAdapter, acquisition_calibration)
from .grassland_resolution import ResolutionWorld, load_model
from .grassland_smoke import position_key
from .navigation import Scorer
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, file_hash, fingerprint, write_json

DEFAULT = Path('apiaviz/output/familiarity-controller')
REGIME = Path('apiaviz/output/navigation-regime')


def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def diagnose(out):
    """Common cached poses; missing positions are reported, not interpolated."""
    deterministic()
    out.mkdir(parents=True, exist_ok=True)
    p = json.loads((REGIME/'protocol.json').read_text())
    rows, missing = [], []
    for spec in p['worlds']:
        environment = Path(spec['environment'])
        base = json.loads((environment/'protocol.json').read_text())
        route, headings = np.array(base['route']), np.array(base['headings'])
        positions, yaw = acquisition_views(route, headings, 1, 0.)
        lateral, lateral_yaw = acquisition_views(route, headings, 9, .2)
        world = ResolutionWorld(environment, p['shape'], cache_only=True)
        images = world.render(positions, yaw)
        indices = np.unique(np.linspace(0, len(yaw)-1, 12).round().astype(int))
        probes = []
        for i in indices:
            for j in (0, 2, 4, 6, 8):
                pos = lateral[i*9+j]
                if not (environment/'panoramas'/f'{position_key(pos)}.png').exists():
                    missing.append(dict(world=spec['name'], station=int(i), lateral_index=j))
                    continue
                probes.append(dict(station=int(i), offset_m=float((j-4)*.05),
                                   position=pos.tolist(), heading=float(yaw[i])))
        query = world.render([v['position'] for v in probes], [v['heading'] for v in probes])
        for seed in p['wiring_seeds']:
            for method in p['methods']:
                model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
                codes = encode(model, images)
                memory = SpikeOverlapMemory(codes)
                values = -memory(encode(model, query)).detach().numpy()
                calibration = acquisition_calibration(codes.numpy())
                for probe, value in zip(probes, values):
                    rows.append(dict(world=spec['name'], seed=seed, method=method,
                                     **probe, familiarity=float(value), scale=calibration['scale']))
    pairs = []
    lookup = {(r['world'], r['seed'], r['method'], r['station'], round(r['offset_m'], 3)): r for r in rows}
    for key, row in lookup.items():
        if abs(key[-1]) < .05:
            continue
        nearer = (*key[:-1], round(key[-1] - np.sign(key[-1])*.1, 3))
        if nearer in lookup:
            change = lookup[nearer]['familiarity'] - row['familiarity']
            pairs.append(dict(world=key[0], seed=key[1], method=key[2], station=key[3],
                              from_offset_m=key[-1], to_offset_m=nearer[-1],
                              change=change, normalized_change=change/row['scale']))
    summary = []
    for method in p['methods']:
        values = [r['normalized_change'] for r in pairs if r['method'] == method]
        summary.append(dict(method=method, pairs=len(values),
                            fraction_inward_positive=float(np.mean(np.array(values)>0)) if values else None,
                            median_normalized_change=float(np.median(values)) if values else None))
    write_json(out/'signal.json', dict(role='Development diagnostic, same cached poses across methods; correlated stations/seeds, no significance claim.',
        limitation='Lateral cached poses cover only available acquisition offsets, at most 20 cm. Centreline self-matches inflate final inward comparisons; no 50 cm recovery claim.',
        rows=rows, inward_pairs=pairs, missing=missing, summary=summary))
    print(json.dumps(summary), flush=True)


def prepare(out, full=False):
    if (out/'protocol.json').exists():
        raise FileExistsError('Use a new output directory or resume run')
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads((REGIME/'protocol.json').read_text())
    scenarios = old['stage2']['scenarios'] if full else [s for s in old['stage2']['scenarios'] if s['name'] in ('aligned', 'left50')]
    seeds = old['wiring_seeds'] if full else [19]
    controllers = [dict(name='exhaustive', phase=1), dict(name='active', phase=1),
                   dict(name='familiarity', phase=1), dict(name='familiarity', phase=-1)]
    protocol = dict(role='Development comparison; existing worlds, no held-out generalization claim.',
        worlds=old['worlds'], encoder_environment=old['encoder_environment'],
        methods=old['methods'], modes=old['modes'], shape=old['shape'], seeds=seeds,
        scenarios=scenarios, controllers=controllers, controller_settings=asdict(Settings()),
        sensor_settings=old['stage3'], evaluation=old['stage2'],
        trials=len(old['worlds'])*len(seeds)*len(old['methods'])*len(scenarios)*len(controllers),
        checkpoint_sha256=old['checkpoint_sha256'],
        scene_sha256={w['name']:file_hash(Path(w['environment'])/'grassland.blend') for w in old['worlds']},
        calibration='New policy: fixed pairwise teaching-code spread, no extra images. Old active: original acquisition-rotation calibration preserved and recorded.',
        fixed='Centreline teaching every 10 cm, encoder, spiking circuit, memory, 10 cm moves, bounds/collisions and resource budgets unchanged.',
        inference='Descriptive paired development results; phase and wiring repeats are not independent worlds.',
        advance='Do not start held-out study until aligned accuracy and recovery are useful on development scenes.')
    write_json(out/'protocol.json', protocol)
    print(json.dumps(dict(trials=protocol['trials'], output=str(out))), flush=True)


def setup(out):
    deterministic()
    p = json.loads((out/'protocol.json').read_text())
    for w in p['worlds']:
        assert file_hash(Path(w['environment'])/'protocol.json') == w['protocol_sha256']
        assert file_hash(Path(w['environment'])/'grassland.blend') == p['scene_sha256'][w['name']]
    for seed in p['seeds']:
        assert file_hash(Path(p['encoder_environment'])/f'encoder-{seed}.pt') == p['checkpoint_sha256'][str(seed)]
    sources = {str(f):file_hash(f) for f in Path('apiaviz').rglob('*.py') if 'output' not in f.parts and 'data' not in f.parts}
    path = out/'manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        assert manifest['protocol_sha256'] == file_hash(out/'protocol.json')
        for name, digest in manifest['source_sha256'].items():
            assert file_hash(Path(name)) == digest, f'Source changed: {name}; use new output'
    else:
        manifest = dict(status='running', protocol_sha256=file_hash(out/'protocol.json'), source_sha256=sources)
        with zipfile.ZipFile(out/'source.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name in sources: z.write(name, name)
        write_json(path, manifest)
    rows = [json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()] if (out/'results.jsonl').exists() else []
    return p, manifest, {r['id']:r for r in rows}


def run(out, cache_only=False):
    p, manifest, completed = setup(out)
    for spec in p['worlds']:
        environment = Path(spec['environment'])
        base = json.loads((environment/'protocol.json').read_text())
        metadata = json.loads((environment/'world.json').read_text())
        route, headings = np.array(base['route']), np.array(base['headings'])
        positions, yaw = acquisition_views(route, headings, 1, 0.)
        world = ResolutionWorld(environment, p['shape'], cache_only=cache_only)
        images = world.render(positions, yaw)
        image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        for seed in p['seeds']:
            for method in p['methods']:
                model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
                model_hash = fingerprint(model)
                codes = encode(model, images)
                memory = SpikeOverlapMemory(codes)
                memory_hash = fingerprint(memory)
                calibration = acquisition_calibration(codes.numpy())
                old_calibration = calibrate(model, memory, codes, world, positions, yaw)
                write_json(out/f"{spec['name']}-{seed}-{method}-calibration.json", dict(
                    familiarity=calibration, active=old_calibration,
                    encoder_fingerprint=model_hash, memory_fingerprint=memory_hash,
                    training_images_sha256=image_hash))
                for scenario in p['scenarios']:
                    for control in p['controllers']:
                        name, phase = control['name'], control['phase']
                        key = f"{spec['name']}-{seed}-{method}-{scenario['name']}-{name}-{phase}"
                        if key in completed: continue
                        started = time.monotonic()
                        policy = None
                        if name == 'familiarity':
                            policy = FamiliarityController(calibration, Settings(**p['controller_settings']), phase)
                            controller = ControllerAdapter(policy)
                        elif name == 'active':
                            controller = Active(p['sensor_settings']['active'], old_calibration)
                        else:
                            controller = Exhaustive()
                        scorer = Scorer(world, model, memory, encode, 'clean', 0., base['geometry_seed'])
                        result = evaluate(route, headings, scorer, controller, scenario,
                            p['sensor_settings'], p['evaluation'], base['world_bounds_m'], metadata['obstacles'])
                        trace, events = result.pop('trace'), result.pop('events')
                        assert fingerprint(model) == model_hash and fingerprint(memory) == memory_hash
                        assert result['observations'] <= p['sensor_settings']['observation_budget']
                        assert result['time_s'] <= p['sensor_settings']['time_budget_s'] + 1e-8
                        write_json(out/f'{key}.json', dict(trajectory=trace, events=events,
                            sensor=scorer.decisions, decisions=policy.decisions if policy else [],
                            transitions=policy.transitions if policy else []))
                        row = dict(id=key, world=spec['name'], seed=seed, method=method,
                            scenario=scenario['name'], controller=name, phase=phase,
                            memory_views=len(codes), memory_fingerprint=memory_hash,
                            encoder_fingerprint=model_hash, training_images_sha256=image_hash,
                            trace=f'{key}.json', **result, elapsed_s=time.monotonic()-started)
                        with (out/'results.jsonl').open('a') as f:
                            f.write(json.dumps(row, allow_nan=False)+'\n')
                        completed[key] = row
                        print(json.dumps({k:row[k] for k in ('id','termination','steps','polyline_mean_m','elapsed_s')}), flush=True)
    assert len(completed) == p['trials']
    manifest.update(status='complete', trials=len(completed))
    write_json(out/'manifest.json', manifest)


def report(out):
    rows = [json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
    groups = []
    for method in sorted({r['method'] for r in rows}):
        for controller in ('exhaustive', 'active', 'familiarity'):
            for scenario in sorted({r['scenario'] for r in rows}):
                selected = [r for r in rows if (r['method'],r['controller'],r['scenario']) == (method,controller,scenario)]
                if not selected: continue
                groups.append(dict(method=method, controller=controller, scenario=scenario,
                    n=len(selected), arrivals=sum(r['reached_nest'] for r in selected),
                    recoveries=sum(r['recovered'] for r in selected),
                    median_observations=float(np.median([r['observations'] for r in selected])),
                    median_time_s=float(np.median([r['time_s'] for r in selected])),
                    mean_deviation_m=float(np.mean([r['polyline_mean_m'] for r in selected if r['polyline_mean_m'] is not None]))))
    write_json(out/'summary.json', dict(trials=len(rows), groups=groups,
        limitation='Development worlds only. New policy has two phases; phase repeats are not independent samples. Deviation and costs include early failures.'))
    print(json.dumps(groups, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['diagnose','prepare','run','report'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--full', action='store_true', help='Three seeds and all seven scenarios instead of the 72-trial smoke')
    parser.add_argument('--cache-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'diagnose': diagnose(args.output)
    elif args.action == 'prepare': prepare(args.output, args.full)
    elif args.action == 'run': run(args.output, args.cache_only)
    else: report(args.output)


if __name__ == '__main__': main()
