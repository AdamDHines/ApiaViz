"""Summarize both completed UV validation arms without selecting outcomes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model


def read(study):
    p = json.loads((study/'protocol.json').read_text())
    readiness = json.loads((study/'readiness.json').read_text())
    audit = json.loads((study/'audit.json').read_text())
    assert readiness['complete'] and audit['passed']
    assert readiness['protocol_sha256'] == audit['protocol_sha256'] == file_sha(study/'protocol.json')
    rows, blocks = [], {}
    for key, digest in audit['traces'].items():
        data = (study/'trials'/f'{key}.json').read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        detail = json.loads(data)
        row = detail['result']
        rows.append(row)
        method = row['method']
        counts = blocks.setdefault(method, Counter())
        counts.update(e['reason'] for e in detail['events'] if e['kind'] == 'blocked_proposal')
        counts['blocked_without_supported_flow'] += sum(
            m['blocked'] and m['flow']['supported_columns'] == 0 for m in detail['microtrace'])
    assert len(rows) == p['trials']
    artifacts = json.loads((study/'report/artifacts.json').read_text())
    for name, checksum in artifacts.items():
        assert file_sha(study/'report'/name) == checksum, name
    movies = []
    for movie in sorted((study/'report/movies').glob('*.mp4')):
        meta = json.loads(movie.with_suffix('.json').read_text())
        assert meta['video_sha256'] == file_sha(movie)
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(movie), '-f', 'null', '-'], check=True)
        movies.append(dict(name=movie.name, sha256=meta['video_sha256'], decoded=True))
    assert movies and (study/'report/index.html').exists()
    return p, readiness, rows, blocks, movies


def run(primary, off, output):
    if output.exists():
        raise FileExistsError('Use a fresh summary directory')
    torch.set_num_threads(2)
    p, ready, rows, blocks, movies = read(primary)
    q, off_ready, off_rows, off_blocks, off_movies = read(off)
    # The ablation must use exactly the same teaching bank and acquisition.
    for name in ['worlds', 'controller_settings', 'avoidance_settings', 'sensor_settings', 'render', 'evaluation', 'seeds']:
        assert p[name] == q[name], name
    ablation_checks = []
    for world in p['worlds']:
        assert file_sha(primary/'worlds'/world['name']/'teaching.pt') == file_sha(off/'worlds'/world['name']/'teaching.pt')
        a, b = primary/'worlds'/world['name'], off/'worlds'/world['name']
        on_state = torch.load(a/'encoder-apiaviz_uv-19.pt', map_location='cpu', weights_only=True)
        off_state = torch.load(b/'encoder-apiaviz_uv-19.pt', map_location='cpu', weights_only=True)
        assert on_state['encoder'].keys() == off_state['encoder'].keys()
        assert all(torch.equal(v, off_state['encoder'][k]) for k, v in on_state['encoder'].items())
        on_memory, off_memory = on_state['memory']['memory'], off_state['memory']['memory']
        visible = p['encoder']['visible_code_dim']
        assert torch.equal(on_memory[:, :visible] > 0, off_memory[:, :visible] > 0)
        assert torch.count_nonzero(on_memory[:, visible:]) > 0
        assert torch.count_nonzero(off_memory[:, visible:]) == 0
        # Check the configured recall path on a real cached independent image.
        camera = DualCamera(a, json.loads((a/'render.json').read_text()))
        sample = camera.scan(world['route'][0], [world['headings'][0]], uv=True)
        on_model, off_model = make_model(p, 'apiaviz_uv', 19), make_model(q, 'apiaviz_uv', 19)
        on_model.load_state_dict(on_state['encoder']); off_model.load_state_dict(off_state['encoder'])
        on_code, off_code = on_model(sample), off_model(sample)
        assert torch.equal(on_code[:, :visible], off_code[:, :visible])
        assert torch.count_nonzero(on_code[:, visible:]) > 0
        assert torch.count_nonzero(off_code[:, visible:]) == 0
        ablation_checks.append(dict(world=world['name'], passed=True, identical_encoder_weights=True,
            identical_visible_teaching_spike_patterns=True, identical_visible_recall_codes=True,
            uv_off_teaching_and_recall_silent=True))
    groups = []
    collections = [(m, [r for r in rows if r['method'] == m]) for m in p['methods']]
    collections.append(('apiaviz_uv_off', off_rows))
    for name, selected in collections:
        groups.append(dict(model=name, n=len(selected), arrivals=sum(r['reached_nest'] for r in selected),
            aligned_arrivals=sum(r['reached_nest'] for r in selected if r['scenario'] == 'aligned'),
            kick_arrivals=sum(r['reached_nest'] for r in selected if r['scenario'] == 'kick_right50'),
            terminations=dict(Counter(r['termination'] for r in selected)),
            median_final_distance_m=float(np.median([r['final_nest_distance_m'] for r in selected])),
            median_encoding_s=float(np.median([r['encoding_s'] for r in selected])),
            blocks=dict(off_blocks['apiaviz_uv'] if name == 'apiaviz_uv_off' else blocks[name])))
    lookup = {(r['world'], r['scenario'], r['phase'], r['seed']): r for r in off_rows}
    paired = []
    for row in rows:
        if row['method'] != 'apiaviz_uv': continue
        other = lookup[row['world'], row['scenario'], row['phase'], row['seed']]
        paired.append(dict(world=row['world'], scenario=row['scenario'], phase=row['phase'],
            uv_on_arrival=row['reached_nest'], uv_off_arrival=other['reached_nest'],
            on_final_distance_m=row['final_nest_distance_m'], off_final_distance_m=other['final_nest_distance_m'],
            on_full_kick=row['disturbance_applied'] and not row['perturbation_adjusted'],
            off_full_kick=other['disturbance_applied'] and not other['perturbation_adjusted']))
    historical = None
    history = Path(p['parent_study']['path'])
    if (history/'audit.json').exists():
        old_audit = json.loads((history/'audit.json').read_text())
        assert old_audit['passed'] and old_audit['protocol_sha256'] == file_sha(history/'protocol.json')
        for world in p['worlds']:
            for asset in ['grassland.blend', 'collision.json']:
                assert file_sha(history/'worlds'/world['name']/asset) == file_sha(primary/'worlds'/world['name']/asset)
        old_rows = []
        for row in rows:
            path = history/'trials'/f'{row["id"]}.json'
            assert file_sha(path) == old_audit['traces'][row['id']]
            old_rows.append(json.loads(path.read_text())['result'])
        historical = dict(protocol_sha256=file_sha(history/'protocol.json'), groups=[
            dict(model=m, n=sum(r['method']==m for r in old_rows),
                 arrivals=sum(r['reached_nest'] for r in old_rows if r['method']==m)) for m in p['methods']],
            note='Same worlds, wiring seed, phases and scheduled scenarios. Receptor response and acquisition both changed; descriptive combined-change comparison only.')
    output.mkdir(parents=True)
    summary = dict(groups=groups, paired_uv=paired, ablation_checks=ablation_checks,
        matched_historical_subset=historical,
        primary_readiness=ready, uv_off_readiness=off_ready,
        movies=movies, uv_off_movies=off_movies, script_sha256=file_sha(Path(__file__)),
        report_checksums=dict(primary=file_sha(primary/'report/artifacts.json'),
                              uv_off=file_sha(off/'report/artifacts.json')),
        audit_checksums=dict(primary=file_sha(primary/'audit.json'), uv_off=file_sha(off/'audit.json')),
        limitation='Development worlds; one wiring seed; phases are not independent worlds. Input and acquisition changed together versus v1; no isolated causal attribution from the historical comparison.')
    (output/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    names = ['ApiaViz + UV', 'Sobel + colour', 'Ardin-style', 'ApiaViz UV off']
    x = np.arange(4)
    for offset, key, label in [(-.18, 'aligned_arrivals', 'Aligned (6 trials)'), (.18, 'kick_arrivals', 'Scheduled kick (6 trials)')]:
        bars = axes[0].bar(x+offset, [g[key] for g in groups], width=.36, label=label)
        axes[0].bar_label(bars)
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 7.5), ylabel='Arrivals', title='All 48 completed trials retained')
    axes[0].tick_params(axis='x', labelrotation=15)
    axes[0].legend(fontsize=8)
    for i, (_, selected) in enumerate(collections):
        axes[1].scatter(np.full(len(selected), i)+np.linspace(-.15, .15, len(selected)),
                        [r['final_nest_distance_m'] for r in selected], alpha=.7)
    axes[1].set(xticks=x, xticklabels=names, ylabel='Final goal distance (m)', title='Every individual outcome')
    axes[1].tick_params(axis='x', labelrotation=15)
    fig.savefig(output/'validation.png', dpi=160)
    plt.close(fig)
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--uv-off', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.primary.resolve(), args.uv_off.resolve(), args.output.resolve())
