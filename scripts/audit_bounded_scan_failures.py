"""Read-only diagnosis of bounded smoke failures using frozen trial records.

No renderer, new navigation, checkpoint changes or outcome counterfactuals.
The limited-search controls reuse scores at identical saved camera positions.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from apiaviz.research.familiarity_controller import FamiliarityController, Settings, SensorView, wrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_before(bundle, step):
    if step == 1:
        return bundle['result']['initial_position']
    return next(t['position'] for t in bundle['trace'] if t['step'] == step - 1)


def bounded_replay(samples, base, calibration, settings):
    """Only query archived scores within +/-60; no unobserved score inference."""
    calls = []
    def observe(h):
        nearest = min(samples, key=lambda s: abs(wrap(s['heading'] - h)))
        assert abs(wrap(nearest['heading'] - h)) < 1e-6
        assert abs(wrap(h - base)) <= 60 + 1e-6
        calls.append(h)
        return nearest['familiarity'], h, len(calls) * .05
    sensor = SensorView(base, 0., 0., observe, lambda h: (True, h, 0.), lambda: None, lambda _: None)
    policy = FamiliarityController(calibration, replace(settings, scan_extents=(60.,)), phase=1)
    result = policy._scan(sensor, base, wide=True)
    return dict(target=result['target'], supported=result['supported'], contrast=result['contrast'],
                views=len(calls), headings=calls,
                limitation='Fixed-pose search control only; not a new route outcome or a recommended every-step scan.')


def run(study, parent, out):
    if out.exists():
        raise FileExistsError('Preserve prior audits; use a fresh output directory')
    p = json.loads((study / 'protocol.json').read_text())
    oldp = json.loads((parent / 'protocol.json').read_text())
    settings = Settings(**p['controller_settings'])
    excluded = {'controller_settings', 'source_sha256', 'comparison', 'variant',
                'protocol_revision', 'controller_variant'}
    for key in set(p) | set(oldp):
        if key not in excluded:
            assert p.get(key) == oldp.get(key), key
    changed_sources = []
    with zipfile.ZipFile(parent / 'source.zip') as old, zipfile.ZipFile(study / 'source.zip') as new:
        for archive, protocol in ((old, oldp), (new, p)):
            for name, expected in protocol['source_sha256'].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        for name in sorted(set(old.namelist()) | set(new.namelist())):
            if name not in old.namelist() or name not in new.namelist() or old.read(name) != new.read(name):
                changed_sources.append(name)
        assert new.read('apiaviz/research/familiarity_controller.py') == (ROOT / 'apiaviz/research/familiarity_controller.py').read_bytes()
    rows = []
    inputs = {str(path): sha(path) for folder in (study, parent)
              for path in (folder / 'protocol.json', folder / 'source.zip')}
    for path in sorted((study / 'trials').glob('*.json')):
        oldpath = parent / 'trials' / path.name
        inputs[str(path)] = sha(path)
        inputs[str(oldpath)] = sha(oldpath)
        x, old = json.loads(path.read_text()), json.loads(oldpath.read_text())
        r, oldr = x['result'], old['result']
        assert x['protocol_sha256'] == inputs[str(study / 'protocol.json')]
        assert old['protocol_sha256'] == inputs[str(parent / 'protocol.json')]
        equal = {k: r[k] == oldr[k] for k in
                 ('encoder_fingerprint', 'memory_fingerprint', 'training_images_sha256')}
        assert all(equal.values())
        checkpoint = study / 'worlds' / r['world'] / f"encoder-{r['method']}-{r['seed']}.pt"
        inputs[str(checkpoint)] = sha(checkpoint)
        assert inputs[str(checkpoint)] == r['checkpoint_sha256']
        calibration = torch.load(checkpoint, map_location='cpu', weights_only=True)['calibration']
        policy = FamiliarityController(calibration, settings, r['phase'])
        rejected = []
        for d in x['decisions']:
            if 'scan' not in d or d['scan']['supported']:
                continue
            s = d['scan']
            samples = {v['heading']: v['familiarity'] for v in s['samples']}
            # Recorded scan offsets are symmetric about their reference, even
            # across the -180/180 seam. Infer that reference from circular mean.
            base = float(np.degrees(np.angle(np.exp(1j * np.radians(list(samples))).mean())))
            target, contrast, peak_supported = policy._peak(samples, base)
            assert abs(wrap(target - s['target'])) < 1e-6
            assert abs(contrast - s['contrast']) < 1e-9
            extent = max(abs(wrap(h-base)) for h in samples)
            interior = abs(wrap(target-base)) < extent - 1e-6
            assert not (peak_supported and interior)
            competitor = any(abs(wrap(h-target)) > 40 and
                             samples[target]-v <= settings.ambiguity_tolerance*policy.scale
                             for h, v in samples.items())
            rejected.append(dict(step=d['step'], target=target, contrast=contrast,
                                 low_contrast=contrast < settings.contrast_threshold,
                                 competing_peak=competitor, edge_of_scan=not interior,
                                 termination=d.get('termination')))
        first = None
        old_decisions = {d['step']: d for d in old['decisions']}
        for d in x['decisions']:
            od = old_decisions.get(d['step'])
            if od is None or d.get('movement_heading') == od.get('movement_heading'):
                continue
            position = pose_before(x, d['step'])
            same_pose = np.allclose(position, pose_before(old, d['step']), atol=1e-10, rtol=0)
            first = dict(step=d['step'], position=position, same_pose=bool(same_pose),
                         bounded_movement=d.get('movement_heading'), previous_movement=od.get('movement_heading'))
            if same_pose and 'scan' in d and 'scan' in od:
                small, full = d['scan'], od['scan']
                for v in small['samples']:
                    match = min(full['samples'], key=lambda s: abs(wrap(s['heading']-v['heading'])))
                    assert abs(wrap(match['heading']-v['heading'])) < 1e-6
                    assert abs(match['familiarity']-v['familiarity']) < 1e-6
                base = float(np.degrees(np.angle(np.exp(1j*np.radians([v['heading'] for v in small['samples']])).mean())))
                first.update(shared_scores_equal=True, bounded_scan=small,
                             previous_scan_top=sorted(full['samples'], key=lambda v: -v['familiarity'])[:4],
                             complete_bounded_control=bounded_replay(full['samples'], base, calibration, settings))
            break
        turns = [e for e in x['events'] if e['kind'] == 'turn']
        blocked = Counter(e['reason'] for e in x['events'] if e['kind'] == 'blocked_proposal')
        reversals = Counter()
        for a, b in zip(x['microtrace'], x['microtrace'][1:]):
            if abs(wrap(b['heading']-a['heading'])) >= 90 - 1e-6:
                reversals[f"{a['state']} -> {b['state']}"] += 1
        rows.append(dict(id=r['id'], unchanged_fingerprints=equal, previous_termination=oldr['termination'],
                         termination=r['termination'], final_nest_distance_m=r['final_nest_distance_m'],
                         time_s=r['time_s'], rotation_deg=r['rotation_deg'],
                         previous_rotation_deg=oldr['rotation_deg'], blocked_reasons=dict(blocked),
                         turn_commands_at_least_90=sum(abs(e['angle_deg']) >= 90-1e-6 for e in turns),
                         max_turn_command_deg=max(abs(e['angle_deg']) for e in turns),
                         consecutive_motor_heading_changes_at_least_90=dict(reversals),
                         first_different_decision=first, rejected_scans=rejected))
    assert len(rows) == 9
    report = dict(schema='bounded-scan-failure-audit-v1', script_sha256=sha(Path(__file__)),
                  input_sha256=inputs, changed_archived_sources=changed_sources,
                  trials=rows, limitations=[
                      'Retrospective diagnosis of nine development trials; no new route outcomes.',
                      'Only fixed-pose decisions can be counterfactually replayed from these scores.',
                      'Historical full-circle tables supply measured scores; active replay stays within +/-60 degrees.',
                      'No exact pre-UV historical trajectory reconstruction is claimed.'])
    out.mkdir(parents=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    for r in rows:
        d = r['first_different_decision'] or {}
        control = d.get('complete_bounded_control', {})
        print(r['id'], r['termination'], 'first difference', d.get('step'),
              'chosen', d.get('bounded_movement'), 'complete +/-60', control.get('target'), control.get('supported'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'apiaviz/output/graded-navigation-smoke-v2-bounded')
    parser.add_argument('--parent', type=Path, default=ROOT / 'apiaviz/output/graded-navigation-smoke-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/bounded-scan-failure-audit-v1')
    args = parser.parse_args()
    run(args.study.resolve(), args.parent.resolve(), args.output.resolve())
