"""Copy verified completed-trial frames between identical acquisition protocols.

Uses the same per-pose locks as parallel workers. Each destination frame keeps
the original checksum/provenance. Teaching banks and checkpoints are untouched.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apiaviz.research.dual_camera import digest, load_cached
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.uv_parallel import file_lock
from apiaviz.research.uv_trials import atomic_json


def run(source, target):
    manifest = target/'cache-reuse.json'
    if manifest.exists():
        raise FileExistsError('This cache reuse has already been recorded')
    p = json.loads((source/'protocol.json').read_text())
    q = json.loads((target/'protocol.json').read_text())
    assert p['worlds'] == q['worlds'] and p['render'] == q['render']
    references = {w['name']: {} for w in p['worlds']}
    traces = {}
    for trial in sorted((source/'trials').glob('*.json')):
        d = json.loads(trial.read_text())
        assert d['protocol_sha256'] == file_sha(source/'protocol.json')
        traces[trial.name] = file_sha(trial)
        refs = references[d['result']['world']]
        for key, value in d['camera_frames'].items():
            if key in refs: assert refs[key] == value
            refs[key] = value
    copied, existing = 0, 0
    for world, refs in references.items():
        a, b = source/'worlds'/world, target/'worlds'/world
        assert file_sha(a/'render.json') == file_sha(b/'render.json')
        render_hash = digest(json.loads((a/'render.json').read_text()))
        for key, sha in tqdm(refs.items(), desc=f'Reuse {world} frames'):
            assert len(key) == 64 and all(c in '0123456789abcdef' for c in key)
            src, dst = a/'camera'/f'{key}.json', b/'camera'/f'{key}.json'
            assert file_sha(src) == sha
            _, record = load_cached(a/'camera', key, render_hash)
            with file_lock(dst.with_suffix('.lock')):
                if dst.exists():
                    _, old = load_cached(b/'camera', key, render_hash)
                    assert old['array_sha256'] == record['array_sha256']
                    existing += 1
                    continue
                for ext in ('.npz', '.json'):
                    temp = dst.with_suffix('.reuse'+ext)
                    shutil.copy2(src.with_suffix(ext), temp)
                    temp.replace(dst.with_suffix(ext))
                copied += 1
    atomic_json(manifest, dict(source=str(source), source_protocol_sha256=file_sha(source/'protocol.json'),
        target_protocol_sha256=file_sha(target/'protocol.json'), source_traces=traces,
        camera_frames=references, copied=copied, already_present=existing,
        script_sha256=file_sha(Path(__file__)),
        reason='Identical recall acquisitions; exact verified cached arrays reused to avoid redundant rendering.'))
    print(dict(copied=copied, already_present=existing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    run(args.source.resolve(), args.target.resolve())
