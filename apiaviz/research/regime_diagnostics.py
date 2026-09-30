"""Post hoc confidence diagnostic for completed active-navigation trials."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .active_navigation import confidence, segment_collision
from .mechanisms import polyline_distance
from .regime_experiments import DEFAULT
from .regime_report import COLOURS, LABELS
from .study import file_hash, write_json


def main():
    out = DEFAULT
    destination = Path('docs/navigation-experiments')
    destination.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((out / 'protocol.json').read_text())
    manifest = json.loads((out / 'stage3/manifest.json').read_text())
    assert manifest['status'] == 'complete'
    rows = list(map(json.loads, (out / 'stage3/results.jsonl').read_text().splitlines()))
    release_actions = []
    for spec in protocol['worlds']:
        environment = Path(spec['environment'])
        base = json.loads((environment / 'protocol.json').read_text())
        metadata = json.loads((environment / 'world.json').read_text())
        heading = float(base['headings'][0])
        normal = np.array([-np.sin(np.radians(heading)), np.cos(np.radians(heading))])
        for scenario in protocol['stage2']['scenarios']:
            if not scenario['lateral']: continue
            pos = np.array(base['route'][0]) + scenario['lateral'] * normal
            clear = []
            for offset in np.arange(-60, 61, 10):
                angle = np.radians(heading + scenario['heading'] + offset)
                target = pos + .1 * np.array([np.cos(angle), np.sin(angle)])
                if not segment_collision(pos, target, metadata['obstacles']): clear.append(int(offset))
            assert clear, 'No collision-free first action at a prescribed lateral release'
            release_actions.append(dict(world=spec['name'], scenario=scenario['name'],
                                        clear_first_step_offsets_deg=clear))
    write_json(destination / 'release-feasibility.json', dict(
        scope='Post hoc geometry check: every lateral release admits at least one rock-free 10 cm move in the exhaustive controller action range. Not used by any policy.',
        releases=release_actions))
    data = []
    for row in rows:
        if row['controller'] != 'active': continue
        spec = next(w for w in protocol['worlds'] if w['name'] == row['world'])
        base = json.loads((Path(spec['environment']) / 'protocol.json').read_text())
        route = np.asarray(base['route'])
        calibration = json.loads((out / 'stage3' / f"{row['world']}-{row['seed']}-{row['method']}-calibration.json").read_text())
        saved = json.loads((out / 'stage3' / row['trace']).read_text())
        # First observation at each consecutive position is the current-view
        # observation which determines whether a scan is triggered. Later scan
        # headings at that same position must not contaminate this diagnostic.
        current = []
        previous = None
        for d in saved['decisions']['sensor']:
            assert len(d['headings']) == 1
            if d['position'] != previous: current.append(d)
            previous = d['position']
        if not current: continue
        distances = polyline_distance([d['position'] for d in current], route)
        for d, distance in zip(current, distances):
            f = -d['scores'][0] if d['scores'][0] is not None else 0.
            c = confidence(f, calibration)
            data.append(dict(trial=row['id'], method=row['method'], world=row['world'], seed=row['seed'],
                             distance_m=float(distance), confidence=c, familiarity=f,
                             scan_threshold_met=c < protocol['stage3']['active']['scan_threshold']))
    summary = []
    for method in protocol['methods']:
        points = [d for d in data if d['method'] == method]
        far = [d for d in points if d['distance_m'] > .2]
        summary.append(dict(method=method, current_view_queries=len(points), queries_outside_20cm=len(far),
                            fraction_outside_20cm_above_scan_threshold=float(np.mean([not d['scan_threshold_met'] for d in far])) if far else None))
    write_json(destination / 'confidence-diagnostic.json', dict(
        scope='Post hoc descriptive diagnostic. Query positions are correlated within trajectories, not independent replicates.',
        definition='Current-view confidence at positions >20 cm from the continuous taught route; scans require confidence <0.25 on three consecutive decisions plus a cooldown.',
        source_sha256=file_hash(Path(__file__)), groups=summary, queries=data))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, method, group in zip(axes, protocol['methods'], summary):
        points = [d for d in data if d['method'] == method]
        ax.scatter([d['distance_m'] for d in points], [d['confidence'] for d in points],
                   color=COLOURS[method], s=7, alpha=.18, edgecolors='none', rasterized=True)
        ax.axhline(.25, color='#555555', ls='--', lw=1)
        ax.axvline(.2, color='#555555', ls=':', lw=1)
        percent = group['fraction_outside_20cm_above_scan_threshold']
        ax.set_title(f"{LABELS[method]}\n{percent:.0%} of off-route queries above scan threshold" if percent is not None else LABELS[method], fontsize=10)
        ax.set_xlabel('Distance from taught route (m)')
        ax.set_ylim(-.03, 1.03)
        ax.set_xlim(left=0)
    axes[0].set_ylabel('Calibrated current-view confidence')
    fig.suptitle('Confidence during active navigation', fontsize=16, weight='bold')
    fig.tight_layout(rect=[0, .05, 1, .91])
    fig.text(.5, .015, 'Above dashed line: low-confidence scan trigger is not met. Right of dotted line: more than 20 cm off-route.', ha='center', fontsize=9)
    fig.savefig(destination / 'confidence.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
