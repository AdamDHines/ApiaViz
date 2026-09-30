"""Audit and report the three ordered navigation-regime experiments."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from .active_navigation import Active, Exhaustive, calibrate, evaluate
from .grassland_resolution import ResolutionWorld, load_model
from .grassland_smoke import position_key, validate_calibration
from .mechanisms import polyline_distance
from .navigation import evaluate_route
from .openloop import acquisition_views
from .regime_experiments import DEFAULT, BANKS, bank_indices
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json

LABELS = {'linear_colour': 'ApiaViz', 'sobel_colour': 'Sobel + colour', 'ardin_input': 'Ardin-style input'}
COLOURS = {'linear_colour': '#167e70', 'sobel_colour': '#bd7926', 'ardin_input': '#5b72aa'}


def read(out):
    p = json.loads((out / 'protocol.json').read_text())
    rows = {s: [json.loads(line) for line in (out / f'stage{s}' / 'results.jsonl').read_text().splitlines()] for s in (1, 2, 3)}
    return p, rows


class Replay:
    def __init__(self, decisions, world):
        self.decisions, self.world, self.index = decisions, world, 0

    def __call__(self, pos, headings):
        decision = self.decisions[self.index]
        np.testing.assert_allclose(pos, decision['position'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(headings, decision['headings'], atol=1e-10, rtol=0)
        self.index += 1
        return np.array([v if v is not None else float('inf') for v in decision['scores']])


def audit(out):
    torch.set_num_threads(1)
    p, rows = read(out)
    source_archives, cache_positions = {}, {w['name']: {} for w in p['worlds']}
    for stage in (1, 2, 3):
        path = out / f'stage{stage}'
        manifest = json.loads((path / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['protocol_sha256'] == file_hash(out / 'protocol.json')
        assert len(rows[stage]) == len({r['id'] for r in rows[stage]}) == p[f'stage{stage}']['trials']
        if stage == 1:
            expected = {f'meander-{seed}-{method}-{bank}' for seed in p['wiring_seeds'] for method in p['methods'] for bank in BANKS}
        else:
            scenarios = [s['name'] for s in p['stage2']['scenarios']] if stage == 2 else p['stage3']['scenarios']
            controllers = ['exhaustive'] if stage == 2 else p['stage3']['controllers']
            expected = {f"{w['name']}-{seed}-{method}-{s}-{c}" for w in p['worlds'] for seed in p['wiring_seeds'] for method in p['methods'] for s in scenarios for c in controllers}
        assert {r['id'] for r in rows[stage]} == expected
        with zipfile.ZipFile(path / 'source.zip') as z:
            for source, digest in manifest['source_sha256'].items():
                assert hashlib.sha256(z.read(source)).hexdigest() == digest
        source_archives[str(stage)] = file_hash(path / 'source.zip')
    exact_actions, observed_views, score_checks, maximum_score_error = 0, 0, 0, 0.
    aligned_reproduced, timed_prefixes = 0, 0
    legacy_reproduced = 0
    for spec in p['worlds']:
        environment = Path(spec['environment'])
        base = json.loads((environment / 'protocol.json').read_text())
        metadata = json.loads((environment / 'world.json').read_text())
        assert file_hash(environment / 'protocol.json') == spec['protocol_sha256']
        assert file_hash(environment / 'grassland.blend') == metadata['scene_sha256']
        validate_calibration(environment)
        world = ResolutionWorld(environment, p['shape'], cache_only=True)
        route, headings = np.array(base['route']), np.array(base['headings'])
        images_by_bank = {}
        for count in ([1, 9] if spec['name'] == 'meander' else [1]):
            x, y = acquisition_views(route, headings, viewpoints=count, width=.2 if count == 9 else 0.)
            images = world.render(x, y)
            for pos in x: cache_positions[spec['name']][position_key(pos)] = pos.tolist()
            if count == 1: images_by_bank['centre'] = images
            else:
                for bank in BANKS: images_by_bank[bank] = images[bank_indices(len(headings), bank)]
        for seed in p['wiring_seeds']:
            assert file_hash(Path(p['encoder_environment']) / f'encoder-{seed}.pt') == p['checkpoint_sha256'][str(seed)]
            for method in p['methods']:
                model = load_model(Path(p['encoder_environment']), seed, method, p['modes'][method])
                modelsig = fingerprint(model)
                memories = {bank: SpikeOverlapMemory(encode(model, images)) for bank, images in images_by_bank.items()}
                centre_pos, centre_yaw = acquisition_views(route, headings, viewpoints=1, width=0.)
                calibration_path = out / 'stage3' / f"{spec['name']}-{seed}-{method}-calibration.json"
                rebuilt_calibration = calibrate(model, memories['centre'], encode(model, images_by_bank['centre']), world, centre_pos, centre_yaw)
                assert rebuilt_calibration == json.loads(calibration_path.read_text())
                for stage in (1, 2, 3):
                    for row in rows[stage]:
                        if (row['world'], row['seed'], row['method']) != (spec['name'], seed, method): continue
                        path = out / f'stage{stage}'
                        saved = json.loads((path / row['trace']).read_text())
                        memory = memories[row['bank']]
                        assert modelsig == row['encoder_fingerprint']
                        assert fingerprint(memory) == row['memory_fingerprint']
                        if stage == 1:
                            decisions = saved['decisions']
                            replay = Replay(decisions, world)
                            result = evaluate_route(route, headings, replay, 'free', max_steps=200)
                            assert result.pop('trace') == saved['trajectory']
                            for key, value in result.items(): assert row[key] == value, (row['id'], key)
                            if row['bank'] == 'corridor9':
                                old = next(r for r in map(json.loads, Path('apiaviz/output/grassland-resolution/results.jsonl').read_text().splitlines())
                                           if r['resolution'] == '199x51' and r['mode'] == p['modes'][method] and r['seed'] == seed and r['preprocessing'] == method)
                                old_trace = json.loads((Path('apiaviz/output/grassland-resolution') / old['trace']).read_text())
                                assert saved == old_trace
                                legacy_reproduced += 1
                        else:
                            assert hashlib.sha256(images_by_bank['centre'].numpy().tobytes()).hexdigest() == row['training_images_sha256']
                            decisions = saved['decisions']['sensor']
                            replay = Replay(decisions, world)
                            scenario = next(s for s in p['stage2']['scenarios'] if s['name'] == row['scenario'])
                            if row['controller'] == 'exhaustive': controller = Exhaustive()
                            else:
                                calibration = json.loads((path / f"{spec['name']}-{seed}-{method}-calibration.json").read_text())
                                controller = Active(p['stage3']['active'], calibration)
                            result = evaluate(route, headings, replay, controller, scenario, p['stage3'], p['stage2'],
                                              base['world_bounds_m'], metadata['obstacles'], timed=stage == 3)
                            assert result.pop('trace') == saved['trajectory']
                            assert result.pop('events') == saved['decisions']['events']
                            for key, value in result.items(): assert row[key] == value, (row['id'], key)
                            if stage == 2 and spec['name'] == 'meander' and row['scenario'] == 'aligned':
                                old_id = f'meander-{seed}-{method}-centre'
                                old_trace = json.loads((out / 'stage1' / f'{old_id}.json').read_text())['trajectory']
                                assert [t['position'] for t in saved['trajectory']] == [t['position'] for t in old_trace]
                                aligned_reproduced += 1
                            if stage == 3 and row['controller'] == 'exhaustive':
                                old_trace = json.loads((out / 'stage2' / row['trace']).read_text())['trajectory']
                                assert [t['position'] for t in saved['trajectory']] == [t['position'] for t in old_trace[:row['steps']]]
                                timed_prefixes += 1
                            expected_time = row['rotation_deg'] / p['stage3']['yaw_speed_deg_s'] + row['observations'] * p['stage3']['observation_s'] + row['path_length_m'] / p['stage3']['speed_m_s']
                            assert abs(row['time_s'] - expected_time) < 1e-8
                            assert sum(e['kind'] == 'observation' for e in saved['decisions']['events']) == row['observations']
                            if stage == 3:
                                assert row['time_s'] <= p['stage3']['time_budget_s'] + 1e-9
                                assert row['observations'] <= p['stage3']['observation_budget']
                        assert replay.index == len(decisions)
                        exact_actions += row['steps']
                        observed_views += sum(len(d['headings']) for d in decisions)
                        for d in decisions:
                            cache_positions[spec['name']][position_key(d['position'])] = d['position']
                        # Independently reconstruct the first, middle and last
                        # sensory queries in every trial. Action/time replay above
                        # covers all queries, but does not re-encode every image.
                        if decisions:
                            for idx in np.unique([0, len(decisions) // 2, len(decisions) - 1]):
                                d = decisions[idx]
                                codes = encode(model, world.scan(d['position'], d['headings']))
                                actual = memory(codes).detach().numpy()
                                actual[codes.abs().sum(1).numpy() == 0] = float('inf')
                                expected = np.array([v if v is not None else float('inf') for v in d['scores']])
                                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
                                finite = np.isfinite(actual)
                                if finite.any(): maximum_score_error = max(maximum_score_error, float(abs(actual[finite] - expected[finite]).max()))
                                score_checks += len(actual)
                        if saved['trajectory']:
                            deviation = float(polyline_distance([t['position'] for t in saved['trajectory']], route).mean())
                            assert abs(deviation - row['polyline_mean_m']) < 1e-12
                print(f"Audited {spec['name']} seed={seed} {method}", flush=True)
    cache_manifest = {}
    for spec in p['worlds']:
        environment = Path(spec['environment'])
        cache_manifest[spec['name']] = [dict(key=k, position=pos, sha256=file_hash(environment / 'panoramas' / f'{k}.png'))
                                       for k, pos in sorted(cache_positions[spec['name']].items())]
    write_json(out / 'cache-manifest.json', cache_manifest)
    result = dict(status='passed', trials=sum(map(len, rows.values())), all_action_replayed_steps=exact_actions,
                  logged_observations=observed_views, independently_reencoded_scores=score_checks,
                  reencoded_score_max_error=maximum_score_error, legacy_full_bank_trials_exact=legacy_reproduced,
                  stage1_to_stage2_aligned_exact=aligned_reproduced,
                  timed_exhaustive_prefixes_exact=timed_prefixes,
                  unique_cache_positions={k: len(v) for k, v in cache_manifest.items()},
                  source_archive_sha256=source_archives, all_memory_fingerprints_rebuilt=True,
                  all_calibrations_rebuilt=True,
                  all_sensor_time_budgets_checked=True,
                  scope='All actions/events replayed from logged scores; first/middle/last queries re-encoded per trial; all memories reconstructed. Not a full re-encoding of every sensory query.')
    write_json(out / 'audit.json', result)
    return result


def aggregate(rows):
    valid = [r for r in rows if not r.get('termination', '').startswith('invalid_release')]
    errors = [r['polyline_mean_m'] for r in valid if r['polyline_mean_m'] is not None]
    eligible = [r for r in valid if r.get('recovery_applicable')]
    return dict(n=len(rows), valid=len(valid), arrived=sum(r['reached_nest'] for r in valid),
                mean_polyline_cm=float(np.mean(errors) * 100) if errors else None,
                recovered=sum(r.get('recovered', False) for r in eligible), recovery_eligible=len(eligible),
                recovered_and_arrived=sum(r.get('recovered', False) and r['reached_nest'] for r in eligible),
                terminations=dict(Counter(r.get('termination', 'arrival' if r['reached_nest'] else 'step_budget') for r in rows)))


def summarise(p, rows):
    result = {'stage1': [], 'stage2': [], 'stage3': [], 'worlds': [], 'paired_controller': []}
    for bank in BANKS:
        for method in p['methods']:
            subset = [r for r in rows[1] if r['bank'] == bank and r['method'] == method]
            result['stage1'].append(dict(bank=bank, method=method, memory_views=subset[0]['memory_views'],
                                         per_seed_polyline_cm=[r['polyline_mean_m'] * 100 for r in subset], **aggregate(subset)))
    for scenario in p['stage2']['scenarios']:
        for method in p['methods']:
            subset = [r for r in rows[2] if r['scenario'] == scenario['name'] and r['method'] == method]
            result['stage2'].append(dict(scenario=scenario['name'], method=method, **aggregate(subset)))
    for controller in p['stage3']['controllers']:
        for method in p['methods']:
            subset = [r for r in rows[3] if r['controller'] == controller and r['method'] == method]
            success = [r for r in subset if r['reached_nest']]
            moving = [r for r in subset if r['path_length_m'] > 0]
            result['stage3'].append(dict(controller=controller, method=method,
                median_views_per_m=float(np.median([r['observations'] / r['path_length_m'] for r in moving])) if moving else None,
                successful_mean_views=float(np.mean([r['observations'] for r in success])) if success else None,
                successful_mean_time_s=float(np.mean([r['time_s'] for r in success])) if success else None,
                **aggregate(subset)))
    for stage in (2, 3):
        for world in p['worlds']:
            for method in p['methods']:
                for controller in (['exhaustive'] if stage == 2 else p['stage3']['controllers']):
                    subset = [r for r in rows[stage] if r['world'] == world['name'] and r['method'] == method and r['controller'] == controller]
                    result['worlds'].append(dict(stage=stage, world=world['name'], method=method, controller=controller, **aggregate(subset)))
    for method in p['methods']:
        pairs = {}
        for r in rows[3]:
            if r['method'] == method: pairs.setdefault((r['world'], r['seed'], r['scenario']), {})[r['controller']] = r
        both = [v for v in pairs.values() if all(r['reached_nest'] for r in v.values())]
        result['paired_controller'].append(dict(method=method, pairs=len(pairs), both_arrived=len(both),
            mean_active_minus_exhaustive_views=float(np.mean([v['active']['observations'] - v['exhaustive']['observations'] for v in both])) if both else None,
            mean_active_minus_exhaustive_time_s=float(np.mean([v['active']['time_s'] - v['exhaustive']['time_s'] for v in both])) if both else None,
            active_only_arrival=sum(v['active']['reached_nest'] and not v['exhaustive']['reached_nest'] for v in pairs.values()),
            exhaustive_only_arrival=sum(v['exhaustive']['reached_nest'] and not v['active']['reached_nest'] for v in pairs.values())))
    # Secondary, explicitly post hoc validity analysis. Exclude the entire
    # paired case if any method/controller receives an impossible perturbation;
    # retain all predeclared trials and outcomes in the primary tables above.
    result['geometric_validity_sensitivity'] = {}
    for stage in (2, 3):
        invalid = [r for r in rows[stage] if r['termination'] in
                   ('invalid_release_rock', 'invalid_release_bounds', 'displacement_into_rock', 'displacement_outside_field')]
        keys = {(r['world'], r['seed'], r['scenario']) for r in invalid}
        valid = [r for r in rows[stage] if (r['world'], r['seed'], r['scenario']) not in keys]
        groups = []
        for method in p['methods']:
            for controller in (['exhaustive'] if stage == 2 else p['stage3']['controllers']):
                groups.append(dict(method=method, controller=controller,
                                   **aggregate([r for r in valid if r['method'] == method and r['controller'] == controller])))
        result['geometric_validity_sensitivity'][str(stage)] = dict(
            invalid_trials=[dict(id=r['id'], reason=r['termination']) for r in invalid],
            paired_cases_excluded=[list(k) for k in sorted(keys)],
            scope='Post hoc sensitivity; conditions on valid imposed displacements, not an unbiased replacement for the full trial analysis.',
            groups=groups)
    return result


def report(out, destination):
    p, rows = read(out)
    summary = summarise(p, rows)
    summary['calibration_diagnostics'] = []
    for spec in p['worlds']:
        for seed in p['wiring_seeds']:
            for method in p['methods']:
                c = json.loads((out / 'stage3' / f"{spec['name']}-{seed}-{method}-calibration.json").read_text())
                summary['calibration_diagnostics'].append(dict(world=spec['name'], seed=seed, method=method,
                    low=c['low'], high=c['high'], range=c['range'], degenerate=c['degenerate']))
    initial_steering = []
    for row in rows[2]:
        scenario = next(s for s in p['stage2']['scenarios'] if s['name'] == row['scenario'])
        if not scenario['lateral']: continue
        spec = next(w for w in p['worlds'] if w['name'] == row['world'])
        base = json.loads((Path(spec['environment']) / 'protocol.json').read_text())
        trace = json.loads((out / 'stage2' / row['trace']).read_text())['trajectory']
        if not trace: continue
        # An offline diagnostic only: ground-truth tangent never enters control.
        angle = np.radians(trace[0]['heading'] - base['headings'][0])
        inward = -np.sign(scenario['lateral']) * np.sin(angle)
        initial_steering.append(dict(id=row['id'], world=row['world'], method=row['method'],
            seed=row['seed'], scenario=row['scenario'], inward_displacement_cm=float(inward * 10),
            classification='inward' if inward > 1e-6 else 'outward' if inward < -1e-6 else 'parallel',
            recovered=row['recovered'], arrived=row['reached_nest']))
    summary['initial_steering_after_lateral_release'] = initial_steering
    destination.mkdir(parents=True, exist_ok=True)
    (destination / '.gitignore').write_text('!*.png\n')
    write_json(destination / 'protocol.json', p)
    write_json(destination / 'runtime.json', json.loads((out / 'runtime.json').read_text()))
    write_json(destination / 'summary.json', summary)
    write_json(destination / 'audit.json', json.loads((out / 'audit.json').read_text()))
    for stage, data in rows.items():
        fields = sorted(set().union(*(r.keys() for r in data)))
        with (destination / f'stage{stage}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(data)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for method in p['methods']:
        groups = [next(g for g in summary['stage1'] if g['bank'] == bank and g['method'] == method) for bank in BANKS]
        axes[0].plot(range(4), [g['mean_polyline_cm'] for g in groups], marker='o', color=COLOURS[method], label=LABELS[method])
        for x, g in enumerate(groups):
            axes[0].scatter(x + np.linspace(-.04, .04, 3), g['per_seed_polyline_cm'], color=COLOURS[method], alpha=.45, s=18)
    axes[0].set(title='1  Teaching coverage', ylabel='Mean distance to continuous route (cm)')
    axes[0].set_xticks(range(4), ['Centre\n77 views', 'Spread\n77 views', '3 lateral\n231 views', '9 lateral\n693 views'])
    axes[0].legend(fontsize=8)
    scenarios = [s['name'] for s in p['stage2']['scenarios']]
    matrix = np.array([[next(g['arrived'] / g['valid'] if g['valid'] else 0 for g in summary['stage2'] if g['method'] == m and g['scenario'] == s)
                        for s in scenarios] for m in p['methods']])
    axes[1].imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    for i, m in enumerate(p['methods']):
        for j, s in enumerate(scenarios):
            g = next(g for g in summary['stage2'] if g['method'] == m and g['scenario'] == s)
            axes[1].text(j, i, f"{g['arrived']}/{g['valid']}", ha='center', va='center', color='white' if matrix[i,j] > .6 else '#222222', fontsize=8)
    axes[1].set_yticks(range(3), [LABELS[m] for m in p['methods']], fontsize=8)
    axes[1].set_xticks(range(7), ['Aligned', 'Left\n0.5 m', 'Right\n0.5 m', 'Yaw\n+40°', 'Yaw\n−40°', 'Kick\nleft', 'Kick\nright'], fontsize=8)
    axes[1].set_title('2  Recovery trials: nest arrivals')
    for i, controller in enumerate(p['stage3']['controllers']):
        groups = [next(g for g in summary['stage3'] if g['controller'] == controller and g['method'] == m) for m in p['methods']]
        axes[2].bar(np.arange(3) + (i - .5) * .35, [g['arrived'] for g in groups], width=.33,
                    color='#354955' if i == 0 else '#a7b7b3', label=controller.capitalize())
    axes[2].set_xticks(range(3), ['ApiaViz', 'Sobel\n+ colour', 'Ardin-style'])
    axes[2].set(title='3  Movement controller', ylabel='Nest arrivals / 27 trials', ylim=(0, 29))
    axes[2].legend(fontsize=8)
    fig.suptitle('Navigation with limited teaching and active sensing', fontsize=17, weight='bold')
    fig.tight_layout(rect=[0, .04, 1, .94])
    fig.text(.5, .02, 'Stage 1: one world, three seeds. Stages 2–3: three scene–route pairs, three seeds; descriptive results.', ha='center', fontsize=9)
    fig.savefig(destination / 'results.png', dpi=180); plt.close(fig)
    trajectory_figure(out, destination, p, rows[3])
    def table(head, lines): return '\n'.join([head, '|' + '|'.join(['---'] * (head.count('|') - 1)) + '|'] + lines)
    t1 = table('| Memory bank | Views | ApiaViz | Sobel + colour | Ardin-style |', [
        '| ' + ' | '.join([bank, str(next(g['memory_views'] for g in summary['stage1'] if g['bank'] == bank))] +
                         [f"{next(g['mean_polyline_cm'] for g in summary['stage1'] if g['bank'] == bank and g['method'] == m):.2f} cm" for m in p['methods']]) + ' |'
        for bank in BANKS])
    t2 = table('| Release / disturbance | ApiaViz | Sobel + colour | Ardin-style |', [
        '| ' + ' | '.join([scenario] + [f"{g['arrived']}/{g['valid']}" for m in p['methods'] for g in summary['stage2'] if g['scenario'] == scenario and g['method'] == m]) + ' |'
        for scenario in scenarios])
    t3 = table('| Method | Exhaustive arrivals | Active arrivals | Views/m, exhaustive | Views/m, active |', [
        '| ' + ' | '.join([LABELS[m]] + [f"{g['arrived']}/{g['valid']}" for c in p['stage3']['controllers'] for g in summary['stage3'] if g['controller'] == c and g['method'] == m] +
                         [f"{g['median_views_per_m']:.1f}" for c in p['stage3']['controllers'] for g in summary['stage3'] if g['controller'] == c and g['method'] == m]) + ' |'
        for m in p['methods']])
    text = f'''# Teaching, recovery and active-sensing experiments

Read the [interpretation and main findings](INTERPRETATION.md) for the research
assessment, including the [post hoc confidence diagnostic](confidence.png).

Three experiments were executed in order, with the [protocol](protocol.json)
saved before their outcomes: 36 teaching trials, 189 recovery trials and 162
controller trials. All methods use 199 × 51 images, the same stored wiring seeds
(19, 31, 43), 8,000 spiking cells and normalized spike-pattern retrieval. ApiaViz
and Sobel retain filter offsets in degrees; Ardin-style preprocessing retains its
10 × 36 reduction. Only the explicitly tested factors change.

![Results of the three experiments](results.png)

**1. Teaching coverage.** These trials use the original meandering route and
unchanged scan-and-step evaluator. Values below are means over three wiring
seeds, measuring distance to the continuous route line. The historical distance
to sampled route stations is also retained in [stage1.csv](stage1.csv).

{t1}

`centre` stores one forward-facing view at each of 77 stations. `spread_equal`
also stores 77 views, choosing one of the nine lateral offsets at each station
in a fixed repeating cycle. It tests coverage at equal memory count, but does
not represent a naturally walked path. The 231- and 693-view banks use three
and nine lateral offsets at every station. Coverage and count both change in
those comparisons. Each bank is built identically for all front ends.

**2. Recovery.** The memory choice was fixed in advance to a single centreline
traversal, independently of stage 1 results. The existing world is supplemented
by two independently seeded grasslands with bent and hairpin routes. They
contain 77 and 91 teaching stations respectively; the original contains 77.
Each cell below is arrivals across three worlds and three wiring seeds.

{t2}

Lateral releases are ±0.5 m, beyond the old ±0.2 m teaching corridor. Heading
errors are tested separately at ±40°. Kicks displace the agent by 0.5 m before
movement step 36, along the normal at the route's midpoint. The kick is applied
to the agent's current position; it does not reset it onto the taught route.
Displacement distance is excluded from walked distance. Recovery requires three
consecutive post-movement positions within 0.1 m of the continuous route line.
Recovery counts, distances and elapsed times are retained in the CSV and summary.

These trials stop at the field boundary or conservative rock-disc collisions,
as well as at the endpoint or the 200-step limit. The taught paths and displaced
starting positions were checked for rock intersections before evaluation.
This collision rule was added for stages 2–3 and is shared by all methods; stage
1 retains the historical evaluator. It is not a complete obstacle-avoidance or
contact model, and does not model collisions with individual grass blades.
Some imposed mid-route displacements can land in rocks; those outcomes are
reported separately and count as failures of the complete trial, rather than
being silently excluded. They are not interpretable as failures of visual
recovery. The summary includes a post hoc sensitivity analysis removing each
affected world/seed/scenario from **all** methods (and both controllers in stage
3), keeping the remaining comparison paired. This conditional analysis does
not replace the complete trial counts. No heading or endpoint direction is supplied to the
controller. Geometry is used only for teaching and evaluation.

**3. Active sensing.** Both controllers use the centreline memories and the same
aligned, left-offset and right-kick scenarios, paired within world and seed.
They have identical upper limits of 360 simulated seconds, 2,600 observations
and 200 movement steps. Each image costs 50 ms, translation is 0.1 m/s, and
stationary rotation is limited to 180°/s. These values define this experiment;
they are not fitted biological measurements. Stage 2 logs the same nominal
costs but imposes no time cap.

{t3}

Views/m is the median over trials with movement, including failures. Reduced
sensing is not evidence of efficiency if navigation fails. The summary also
compares observation count and time within pairs where both controllers arrive.

Exhaustive control physically visits thirteen orientations over ±60° before
choosing a heading, paying for its full yaw excursion and return. Active control
observes the current heading once, then alternates left and right turns with
amplitude `2 + 28 × (1 − confidence)²` degrees. After three consecutive low-
confidence views it may scan four additional orientations (±30°, ±60°), with
at least six movement steps between scan bouts. Every queried orientation is
physically visited and charged. It cannot inspect unseen candidate views or
consult the route coordinates. Parameters were not tuned after seeing failures.
This compares two complete control policies: scan frequency, angular sampling
and movement oscillations all differ. It does not isolate scan frequency alone.

Confidence uses the median leave-one-out overlap of taught codes and the median
overlap of ±60° rotations at up to twelve teaching stations, with a minimum
normalization range of 0.05. These additional 24 calibration views are the same
poses across methods; they are not added to the route memory. The calibration
is shared across controller conditions within each method, seed and world.

The movement model remains deliberately limited: finite-rate stationary turns
and straight 10 cm translations, independent static spiking windows, and no
continuous head/body dynamics or optic-flow control. It tests the consequence
of reducing compulsory scans, not a complete ant or honeybee motor system.

![Aligned trajectories, seed 19 fixed before outcomes](trajectories.png)

**Interpretation limits.** There are only three scene–route pairs; the original
world has already informed development. Seeds and multiple releases are matched
technical repeats, not independent environments. Results are exploratory and
do not establish statistical superiority. Whole-run deviation includes failed
trials but can be low when a run stops early; interpret it alongside arrival,
recovery, path length and termination reason. Arriving at the nest does not by
itself imply recovery of the taught route. The new routes were prescribed before
evaluation and defeat straight walking even without the rock collision rule.

The memory stores separate spike templates, so these experiments still do not
test a compressed mushroom-body output circuit. “Ardin-style” denotes the input
processing control, not the full Ardin neural model. Navigation uses sensory
feedback without external corrective resets.

**Verification and files.** [audit.json](audit.json) describes exact replay of
all actions and timing from recorded observations, reconstruction of every
memory, independent re-encoding of first/middle/last sensory queries in every
trial, and exact reproduction of the nine earlier full-bank reference trials.
It does not claim to re-encode every recorded sensory observation.
[summary.json](summary.json) contains all grouped results, per-world outcomes and
paired controller comparisons. Full records are in [stage1.csv](stage1.csv),
[stage2.csv](stage2.csv) and [stage3.csv](stage3.csv).
Package and platform versions are recorded in [runtime.json](runtime.json).

Scenes, panorama caches, traces, calibration records and source archives are
retained under `apiaviz/output/navigation-regime/`; the original world's new
panoramas extend its existing cache. Training and evaluation defaults elsewhere
in the repository remain unchanged.

```sh
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python \\
  -m apiaviz.research.regime_report --audit
```
'''
    (destination / 'README.md').write_text(text)


def trajectory_figure(out, destination, p, rows):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for col, spec in enumerate(p['worlds']):
        environment = Path(spec['environment'])
        base = json.loads((environment / 'protocol.json').read_text())
        metadata = json.loads((environment / 'world.json').read_text())
        route = np.array(base['route'])
        for row, controller in enumerate(p['stage3']['controllers']):
            ax = axes[row, col]
            for rock in metadata['obstacles']:
                ax.add_patch(Circle((rock['x'], rock['y']), rock['conservative_radius_m'], color='#ddd9cf', alpha=.55))
            ax.plot(*route.T, color='#29363b', ls='--', lw=1.8, label='Taught route')
            ax.scatter(*route[0], color='#29363b', marker='s', s=20)
            ax.add_patch(Circle(route[-1], .2, color='#29363b', fill=False))
            for method in p['methods']:
                r = next(r for r in rows if (r['world'], r['seed'], r['method'], r['scenario'], r['controller']) ==
                         (spec['name'], 19, method, 'aligned', controller))
                trace = json.loads((out / 'stage3' / r['trace']).read_text())['trajectory']
                points = np.vstack([route[0], [t['position'] for t in trace]]) if trace else route[:1]
                ax.plot(*points.T, color=COLOURS[method], lw=1.4, label=LABELS[method])
            ax.set(title=f"{spec['name'].capitalize()} · {controller}", xlim=(-.3, 10.5), ylim=(-.3, 10.4), aspect='equal')
            if col == 0: ax.set_ylabel('y (m)')
            if row == 1: ax.set_xlabel('x (m)')
    axes[0,0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Aligned release: the same memories with two movement controllers\nSeed 19, selected in advance; grey discs are conservative rock boundaries', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(destination / 'trajectories.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--docs', type=Path, default=Path('docs/navigation-experiments'))
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if args.audit: print(json.dumps(audit(args.output), indent=2), flush=True)
    report(args.output, args.docs)
