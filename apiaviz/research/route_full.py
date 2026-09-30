"""Frozen full matrix for continuous visual navigation and camera-only avoidance.

prepare freezes the protocol; run starts independent, resumable world workers.
Renderers must already be running. Results, audits and a report are generated
automatically on completion. No original experiment or controller is modified.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .controller_experiments import setup
from .controller_full_report import paired_effect, retention
from .familiarity_controller import (Settings as NavigationSettings,
    FamiliarityController, ControllerAdapter, acquisition_calibration)
from .grassland_resolution import ResolutionWorld, load_model
from .motor_feedback import evaluate
from .navigation import Scorer
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, file_hash, fingerprint, write_json
from .visual_avoidance import Settings as AvoidanceSettings

DEFAULT = Path('apiaviz/output/route-continuous-full')
DOCS = Path('docs/route-continuous-full')
NAMES = dict(linear_colour='ApiaViz', sobel_colour='Sobel + colour', ardin_input='Ardin-style')


def cases(p):
    for w in p['worlds']:
        for seed in p['seeds']:
            for method in p['methods']:
                for s in p['scenarios']:
                    for c in p['controllers']:
                        assert c['name'] == 'familiarity' and c['phase'] in (-1, 1)
                        key = f"{w['name']}-{seed}-{method}-{s['name']}-familiarity-{c['phase']}"
                        yield key, w, seed, method, s, c['phase']


def read_rows(out):
    path = out/'results.jsonl'
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def prepare(out):
    if (out/'protocol.json').exists():
        raise FileExistsError('Use run to resume, or choose a fresh output directory')
    p = json.loads(Path('apiaviz/output/route-motor-feedback-v2/protocol.json').read_text())
    full = json.loads(Path('apiaviz/output/controller-full/protocol.json').read_text())
    for key in ('worlds', 'seeds', 'scenarios'):
        p[key] = full[key]
    p['controllers'] = [dict(name='familiarity', phase=phase) for phase in (1, -1)]
    p.update(role='Frozen expanded development matrix for continuous route guidance with camera-only avoidance.',
        advance='User authorized the full suite after the continuous-guidance smoke tests.',
        comparison='All three image frontends use the identical continuous controller, reflex, teaching images and resource budgets. Original studies remain separate.',
        limitation='Three existing development environments, including scenes used to develop the controller. Seeds and phases are repeated measures, not independent environments. No significance or held-out generalization claim.',
        scheduling='One append-only worker per world. Recompute all trials, including cached smoke conditions. Freeze all settings before outcomes.',
        calibration='Fixed pairwise teaching-code spread; no additional calibration images.',
        analysis=dict(primary='Nest arrival across every scheduled trial, including invalid releases and imposed kicks.',
            secondary=['sustained route return', 'renewed loss after return', 'path-weighted deviation', 'camera observations', 'simulated time', 'termination reason'],
            pairing='Average the two phases within world/seed/scenario, then average paired seed/scenario differences within each world.',
            intervals='Exact world bootstrap (27 size-three samples). Descriptive with only three worlds.',
            perturbation='Imposed kicks are external disturbances, excluded from walked distance. A matched-gaze comparison spanning a kick includes its visual consequences; the controller receives no kick notification.',
            success='Unchanged 20 cm endpoint radius, evaluated at macro-step boundaries. Report all failures; no parameter tuning during the suite.'))
    p['trials'] = len(list(cases(p)))
    assert p['trials'] == 378
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', p)
    setup(out)
    for w in p['worlds']:
        shard = out/'worlds'/w['name']
        shard.mkdir(parents=True)
        q = deepcopy(p)
        q.update(worlds=[w], trials=p['trials']//len(p['worlds']))
        write_json(shard/'protocol.json', q)
        setup(shard)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS/'PROTOCOL.md').write_text(
        '# Continuous route guidance: full development suite\n\n'
        '378 trials: three worlds (meander, bend, hairpin), three frontends '
        '(ApiaViz, Sobel + colour, Ardin-style), three wiring seeds (19, 31, 43), '
        'seven release/disturbance conditions, and both initial search/passing directions.\n\n'
        'The seven conditions are aligned; 50 cm left/right releases; ±40° initial yaw; '
        'and 50 cm left/right imposed displacement before movement step 36. '
        'Both the navigator and the avoidance reflex receive the same scheduled phase.\n\n'
        'The tested continuous controller, RGB-only reflex, 199 × 51 input, 10 cm centreline '
        'teaching, encoders and mushroom-body memories are unchanged. Every camera view and '
        'turn is charged against the original 360 s / 2600 observation budgets; at most '
        '200 macro movements. Original 20 cm arrival radius and collision checks are retained.\n\n'
        'Primary outcome: arrival among all scheduled trials. Secondary outcomes: sustained '
        'route return, renewed loss, deviation, observation/time costs and failure reasons. '
        'Invalid imposed displacements remain in the primary denominator and are identified '
        'separately. Paired method comparisons average phases, then seeds/scenarios within '
        'each world. World-bootstrap intervals are descriptive; three development worlds '
        'do not support a strong significance or held-out generalization claim.\n\n'
        'Original results remain separate. Sources, protocol, scenes and checkpoints are '
        'hashed before execution. Each world writes resumable trials; final audits, summary '
        'and figures are generated automatically. Rendered panoramas remain cached.\n')
    print(json.dumps(dict(output=str(out), trials=p['trials'])), flush=True)


def worker(out):
    p, manifest, completed = setup(out)
    assert len(p['worlds']) == 1
    expected = {c[0] for c in cases(p)}
    assert set(completed) <= expected and len(read_rows(out)) == len(completed)
    w = p['worlds'][0]
    env = Path(w['environment'])
    base = json.loads((env/'protocol.json').read_text())
    obstacles = json.loads((env/'world.json').read_text())['obstacles']
    route, headings = np.asarray(base['route']), np.asarray(base['headings'])
    world = ResolutionWorld(env, p['shape'])
    positions, yaw = acquisition_views(route, headings, 1, 0.)
    images = world.render(positions, yaw)
    image_hash = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    loaded = None
    for key, _, seed, method, scenario, phase in cases(p):
        if key in completed:
            continue
        if loaded != (seed, method):
            model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
            codes = encode(model, images)
            memory = SpikeOverlapMemory(codes)
            model_hash, memory_hash = fingerprint(model), fingerprint(memory)
            calibration = acquisition_calibration(codes.numpy())
            loaded = (seed, method)
        started = time.monotonic()
        print(json.dumps(dict(starting=key)), flush=True)
        factory = lambda: ControllerAdapter(FamiliarityController(
            calibration, NavigationSettings(**p['controller_settings']), phase))
        scorer = Scorer(world, model, memory, encode, 'clean', 0., base['geometry_seed'])
        result = evaluate(route, headings, scorer, factory, scenario,
            p['sensor_settings'], p['evaluation'], base['world_bounds_m'], obstacles,
            AvoidanceSettings(**p['avoidance_settings']), phase=phase)
        assert fingerprint(model) == model_hash and fingerprint(memory) == memory_hash
        assert result['observations'] <= p['sensor_settings']['observation_budget']
        assert result['time_s'] <= p['sensor_settings']['time_budget_s']+1e-8
        detail = {k:result.pop(k) for k in ('trace', 'microtrace', 'events', 'decisions')}
        write_json(out/f'{key}.json', dict(**detail, sensor=scorer.decisions))
        row = dict(id=key, world=w['name'], seed=seed, method=method,
            scenario=scenario['name'], controller='familiarity', phase=phase,
            memory_views=len(codes), memory_fingerprint=memory_hash,
            encoder_fingerprint=model_hash, training_images_sha256=image_hash,
            trace=f'{key}.json', **result, elapsed_s=time.monotonic()-started)
        with (out/'results.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
            f.flush()
        completed[key] = row
        print(json.dumps({k:row[k] for k in ('id', 'termination', 'elapsed_s')}), flush=True)
    assert set(completed) == expected
    manifest.update(status='complete', trials=len(completed))
    write_json(out/'manifest.json', manifest)
    from .route_full_audit import audit
    audit(out)


def status(out):
    p = json.loads((out/'protocol.json').read_text())
    counts = {w['name']:len(read_rows(out/'worlds'/w['name'])) for w in p['worlds']}
    return dict(completed=sum(counts.values()), planned=p['trials'], worlds=counts)


def collect(out):
    p, manifest, _ = setup(out)
    rows = []
    for w in p['worlds']:
        shard = out/'worlds'/w['name']
        _, child, completed = setup(shard)
        assert child['status'] == 'complete'
        assert json.loads((shard/'audit.json').read_text())['passed']
        assert child['source_sha256'] == manifest['source_sha256']
        for r in completed.values():
            rows.append(dict(r, trace=str(Path('worlds')/w['name']/r['trace'])))
    assert {r['id'] for r in rows} == {c[0] for c in cases(p)}
    assert len(rows) == p['trials']
    temporary = out/'results.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    temporary.replace(out/'results.jsonl')
    manifest.update(status='complete', trials=len(rows))
    write_json(out/'manifest.json', manifest)
    report(out)


def report(out):
    p, manifest, completed = setup(out)
    assert manifest['status'] == 'complete' and len(completed) == p['trials']
    rows = list(completed.values())
    for row in rows:
        detail = json.loads((out/row['trace']).read_text())
        kept = retention(detail['trace'], row['scenario'], p['evaluation']['kick_before_step'])
        assert kept['return_found'] == row['recovered']
        row.update(kept)
    comparisons = [paired_effect(rows, 'reached_nest', ('familiarity', 'linear_colour'),
                    ('familiarity', method)) for method in ('sobel_colour', 'ardin_input')]
    groups = []
    for world in ['all']+[w['name'] for w in p['worlds']]:
        for method in p['methods']:
            selected = [r for r in rows if r['method'] == method and (world == 'all' or r['world'] == world)]
            errors = [r['polyline_mean_m'] for r in selected if r['polyline_mean_m'] is not None]
            groups.append(dict(world=world, method=method, n=len(selected),
                arrivals=sum(r['reached_nest'] for r in selected),
                recoveries=sum(r['recovered'] for r in selected),
                lost_after_return=sum(r['lost_after_return'] for r in selected),
                mean_deviation_m=float(np.mean(errors)) if errors else None,
                median_observations=float(np.median([r['observations'] for r in selected])),
                median_time_s=float(np.median([r['time_s'] for r in selected])),
                terminations=dict(Counter(r['termination'] for r in selected))))
    summary = dict(trials=len(rows), groups=groups, paired_arrival_effects=comparisons,
                   limitation=p['limitation'], analysis=p['analysis'])
    write_json(out/'summary.json', summary)
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(DOCS/'results.json', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), layout='constrained')
    for ax, w in zip(axes, p['worlds']):
        matrix = []
        for method in p['methods']:
            matrix.append([sum(r['reached_nest'] for r in rows if
                (r['world'], r['method'], r['scenario']) == (w['name'], method, s['name']))
                for s in p['scenarios']])
        ax.imshow(matrix, vmin=0, vmax=6, cmap='YlGnBu', aspect='auto')
        for i in range(3):
            for j in range(7):
                ax.text(j, i, f'{matrix[i][j]}/6', ha='center', va='center',
                        color='white' if matrix[i][j] > 3 else '#20313c')
        ax.set(xticks=range(7), xticklabels=[s['name'] for s in p['scenarios']],
               yticks=range(3), yticklabels=[NAMES[m] for m in p['methods']], title=w['name'])
    fig.suptitle('Nest arrivals · continuous route guidance and camera-only avoidance\nThree wiring seeds × two initial search directions per cell')
    fig.savefig(DOCS/'arrivals.png', dpi=170)
    plt.close(fig)
    lines = ['# Full continuous-navigation suite', '', f'All {len(rows)} scheduled trials completed.', '',
        '![Arrivals by world and release](arrivals.png)', '',
        '| Input | Arrivals | Mean deviation (cm) | Median views | Median time (s) |',
        '|---|---:|---:|---:|---:|']
    for g in groups[:3]:
        lines.append(f"| {NAMES[g['method']]} | {g['arrivals']}/{g['n']} | {100*g['mean_deviation_m']:.1f} | {g['median_observations']:.0f} | {g['median_time_s']:.1f} |")
    lines += ['', 'All failures remain in these denominators. Deviation and cost averages include early failures; '
        'they must be interpreted alongside arrival. Detailed termination counts and paired, world-level '
        'arrival differences are in [results.json](results.json).', '', p['limitation'], '',
        'Each completed world passed checks of source/scene/checkpoint hashes, movement and observation '
        'accounting, collision-free executed segments, external displacements, and the attribution of '
        'familiarity comparisons to commanded movements. Training-image hashes match across frontends, '
        'with fixed encoder/memory hashes across conditions.', '', '[Frozen protocol](PROTOCOL.md)', '']
    (DOCS/'README.md').write_text('\n'.join(lines))


def run(out, stop_renderers=False):
    p, _, _ = setup(out)
    jobs = []
    try:
        for w in p['worlds']:
            shard = out/'worlds'/w['name']
            log = (out/f"{w['name']}.log").open('a')
            process = subprocess.Popen([sys.executable, '-m', 'apiaviz.research.route_full', 'worker', '--output', str(shard)],
                                       stdout=log, stderr=subprocess.STDOUT)
            jobs.append((w['name'], process, log))
        while any(proc.poll() is None for _, proc, _ in jobs):
            failed = [name for name, proc, _ in jobs if proc.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Workers failed: {failed}; inspect world logs')
            progress = dict(status(out), status='running', updated_unix_s=time.time(),
                            workers={name:proc.pid for name, proc, _ in jobs})
            write_json(out/'progress.json', progress)
            print(json.dumps(progress), flush=True)
            time.sleep(30)
        if any(proc.returncode for _, proc, _ in jobs):
            raise RuntimeError('A world worker failed; inspect logs')
        collect(out)
        write_json(out/'progress.json', dict(status(out), status='complete', updated_unix_s=time.time()))
        print('COMPLETE: results, audits and report saved', flush=True)
    except BaseException as error:
        write_json(out/'progress.json', dict(status(out), status='interrupted', error=str(error), updated_unix_s=time.time()))
        raise
    finally:
        for _, proc, log in jobs:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            log.close()
        if stop_renderers:
            for w in p['worlds']:
                (Path(w['environment'])/'stop').touch()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker', 'collect', 'report', 'status'])
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--stop-renderers', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        run(args.output, args.stop_renderers)
    elif args.action == 'status':
        print(json.dumps(status(args.output)))
    else:
        globals()[args.action](args.output)
