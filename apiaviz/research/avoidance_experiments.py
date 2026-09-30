"""Separate, resumable camera-only avoidance development protocol.

python -m apiaviz.research.avoidance_experiments prepare --world hairpin
python -m apiaviz.research.avoidance_experiments run
python -m apiaviz.research.avoidance_experiments report
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .avoidance_navigation import evaluate
from .controller_experiments import setup
from .familiarity_controller import (Settings as NavigationSettings,
    FamiliarityController, ControllerAdapter, acquisition_calibration)
from .grassland_resolution import ResolutionWorld, load_model
from .navigation import Scorer
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, write_json
from .visual_avoidance import Settings

DEFAULT = Path('apiaviz/output/visual-avoidance-smoke-v2')
OLD = Path('apiaviz/output/familiarity-controller')


def prepare(out, selected_world=None):
    if (out/'protocol.json').exists():
        raise FileExistsError('Choose a new directory or resume run')
    p = json.loads((OLD/'protocol.json').read_text())
    if selected_world:
        p['worlds'] = [w for w in p['worlds'] if w['name'] == selected_world]
        if not p['worlds']:
            raise ValueError(selected_world)
    p.update(role='Camera-only obstacle-avoidance development smoke, not a held-out efficacy study.',
        controllers=[dict(name='familiarity', phase=1)],
        trials=len(p['worlds'])*len(p['methods'])*len(p['scenarios']),
        avoidance_settings=asdict(Settings()),
        fixed='Same RGB 199x51, teaching images, encoder checkpoints, spiking circuit, memory, navigation settings and budgets as original smoke.',
        changed='Shared RGB optic-flow reflex, 5 mm startup/uncertain probes and at most 2 cm substeps. Every extra view/turn charged. Navigator transient state restarted after a diverted macro move; memory unchanged.',
        comparison='Historical no-avoidance results are descriptive; they also differ in sampling frequency and motor resets. Not an isolated reflex ablation.',
        physics='Original conservative rock discs retained for failure checks only, never provided to avoidance. No depth, labels, rangefinder or contact signal.',
        limitation='Parameters developed on the hairpin starting rock. That scene is a development check, not independent validation. Only one wiring seed and passing phase in this smoke.',
        metrics='Path-weighted deviation over all executed substeps, including partial final moves; historical metric was equally weighted 10 cm endpoints. Arrival is directly comparable but computational costs differ.',
        advance='Keep the previous 181-trial incomplete study separate. Validate this policy before launching a new exhaustive protocol.')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', p)
    setup(out)  # Freeze source, scene/checkpoint/protocol hashes before outcomes.
    print(json.dumps(dict(trials=p['trials'], output=str(out))), flush=True)


def run(out):
    p, manifest, completed = setup(out)
    for spec in p['worlds']:
        env = Path(spec['environment'])
        base = json.loads((env/'protocol.json').read_text())
        obstacles = json.loads((env/'world.json').read_text())['obstacles']
        route, headings = np.asarray(base['route']), np.asarray(base['headings'])
        world = ResolutionWorld(env, p['shape'])
        positions, yaw = acquisition_views(route, headings, 1, 0.)
        images = world.render(positions, yaw)
        image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        for seed in p['seeds']:
            for method in p['methods']:
                model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
                codes = encode(model, images)
                memory = SpikeOverlapMemory(codes)
                model_hash, memory_hash = fingerprint(model), fingerprint(memory)
                calibration = acquisition_calibration(codes.numpy())
                for scenario in p['scenarios']:
                    key = f"{spec['name']}-{seed}-{method}-{scenario['name']}-familiarity-1"
                    if key in completed:
                        continue
                    started = time.monotonic()
                    print(json.dumps(dict(starting=key)), flush=True)
                    factory = lambda: ControllerAdapter(FamiliarityController(
                        calibration, NavigationSettings(**p['controller_settings']), 1))
                    scorer = Scorer(world, model, memory, encode, 'clean', 0., base['geometry_seed'])
                    result = evaluate(route, headings, scorer, factory, scenario,
                        p['sensor_settings'], p['evaluation'], base['world_bounds_m'], obstacles,
                        Settings(**p['avoidance_settings']))
                    assert fingerprint(model) == model_hash and fingerprint(memory) == memory_hash
                    assert result['observations'] <= p['sensor_settings']['observation_budget']
                    assert result['time_s'] <= p['sensor_settings']['time_budget_s']+1e-8
                    detail = {k:result.pop(k) for k in ('trace','microtrace','events','decisions')}
                    write_json(out/f'{key}.json', dict(**detail, sensor=scorer.decisions))
                    row = dict(id=key, world=spec['name'], seed=seed, method=method,
                        scenario=scenario['name'], controller='familiarity', phase=1,
                        memory_views=len(codes), memory_fingerprint=memory_hash,
                        encoder_fingerprint=model_hash, training_images_sha256=image_hash,
                        trace=f'{key}.json', **result, elapsed_s=time.monotonic()-started)
                    with (out/'results.jsonl').open('a') as f:
                        f.write(json.dumps(row, allow_nan=False)+'\n')
                    completed[key] = row
                    print(json.dumps(row), flush=True)
    assert len(completed) == p['trials']
    manifest.update(status='complete', trials=len(completed))
    write_json(out/'manifest.json', manifest)


def report(out):
    p = json.loads((out/'protocol.json').read_text())
    rows = [json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    old = {r['id']:r for r in (json.loads(s) for s in (OLD/'results.jsonl').read_text().splitlines())}
    paired = []
    for r in rows:
        b = old[r['id']]
        for field in ('encoder_fingerprint','memory_fingerprint','training_images_sha256','memory_views'):
            assert r[field] == b[field], (r['id'], field)
        paired.append(dict(id=r['id'], before=b['termination'], after=r['termination'],
            before_path_m=b['path_length_m'], after_path_m=r['path_length_m'],
            before_observations=b['observations'], after_observations=r['observations'],
            detours=r['motor_state_resets']))
    write_json(out/'summary.json', dict(completed=len(rows), planned=p['trials'],
        identical_training_and_models=True, paired=paired,
        limitation=p['comparison']+' '+p['limitation']))
    print(json.dumps(paired, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','run','report'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--world', choices=['meander','bend','hairpin'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output, args.world)
    elif args.action == 'run':
        run(args.output)
    else:
        report(args.output)


if __name__ == '__main__':
    main()
