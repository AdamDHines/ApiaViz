"""Offline matched-pose retrieval; route labels never enter navigation policies."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.mechanisms import polyline_distance
from apiaviz.research.study import encode
from apiaviz.research.uv_trials import make_model, atomic_json
from apiaviz.research.spectral_input import file_sha


def run(study, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    p = json.loads((study / 'protocol.json').read_text())
    source = study / 'meander-direction-audit.json'
    positions = json.loads(source.read_text())['records']
    w = next(w for w in p['worlds'] if w['name'] == 'meander')
    route = np.asarray(w['route'])
    env = study / 'worlds/meander'
    bank = torch.load(env / 'teaching.pt', weights_only=True)
    angles = np.arange(-180., 180., 10.) + w['headings'][0]
    records = []
    with DualCamera(env, json.loads((env / 'render.json').read_text())) as camera:
        for method in p['methods']:
            model = make_model(p, method, 19)
            uv = method == 'apiaviz_uv'
            conditions = (True, False) if uv else (None,)
            for enabled in conditions:
                def coding(images):
                    result = model(images, uv_enabled=enabled) if uv else encode(model, images)
                    return F.normalize((result > 0).float(), dim=1)
                memory = coding(bank['uv' if uv else 'rgb'])
                for pose in positions:
                    pos = np.asarray(pose['position'])
                    similarity = coding(camera.scan(pos, angles, uv=uv)) @ memory.T
                    scores, station = similarity.max(1)
                    best = int(scores.argmax())
                    heading = float(angles[best])
                    proposed = pos + .1 * np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
                    before, after = polyline_distance([pos, proposed], route)
                    records.append(dict(method=method, uv_enabled=enabled, step=pose['step'],
                        position=pos.tolist(), heading=heading, score=float(scores[best]),
                        matched_station=int(station[best]), nearest_station=pose['nearest_station'],
                        tangent_error_deg=abs((heading-pose['tangent']+180)%360-180),
                        distance_before_m=float(before), distance_after_m=float(after)))
    atomic_json(output, dict(protocol_sha256=file_sha(study/'protocol.json'),
        poses_sha256=file_sha(source), script_sha256=file_sha(Path(__file__)),
        limitation='One failed development trajectory, evaluated retrospectively. Projected 10 cm headings are geometric labels, not executed movements or collision checks. No arrival or UV causal claim.',
        records=records))
    for method, enabled in dict.fromkeys((r['method'], r['uv_enabled']) for r in records):
        group = [r for r in records if r['method']==method and r['uv_enabled']==enabled]
        off = [r for r in group if r['distance_before_m'] >= .1]
        print(method, enabled, 'poses', len(group), 'off-route inward',
              sum(r['distance_after_m'] < r['distance_before_m'] for r in off), '/', len(off))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.study, a.output)
