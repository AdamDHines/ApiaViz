"""Isolate lost pre-UV angular filtering on the fixed visual-response probe bank.

No new rendering, parameter fitting, policy changes, or navigation claims.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tqdm import tqdm
from apiaviz.research.angular_frontend import AngularEncoder
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.study import encode, fingerprint
from apiaviz.research.spectral_input import file_sha
from visual_response_controls import binary, metrics, select_poses

CONDITIONS = {
    'angular_rgb': 'Original pre-UV AngularEncoder linear_colour; RGB, 8000 cells',
    'angular_bee': 'Same AngularEncoder on bounded bee green/blue; 8000 cells',
    'angular_sobel': 'Original pre-UV AngularEncoder Sobel; RGB, same 8000-cell wiring',
    'angular_bee_plus_uv': 'angular_bee with unchanged current 2000-cell UV stream appended',
    'angular_rgb_plus_uv': 'angular_rgb with unchanged current 2000-cell UV stream appended',
}


def job(source_string, out_string, w):
    torch.set_num_threads(2)
    source, out = Path(source_string), Path(out_string)
    protocol = json.loads((source/'protocol.json').read_text())
    study = Path(protocol['study'])
    p = json.loads((study/'protocol.json').read_text())
    env = study/'worlds'/w['name']
    camera = DualCamera(source/w['name'], json.loads((env/'render.json').read_text()))
    bank = torch.load(env/'teaching.pt', weights_only=True)
    poses = select_poses(w)
    anchor = float(w['headings'][0])
    angles_list = [np.unique(np.round(np.r_[np.arange(-180., 180., 5.)+anchor,
                   np.arange(-20., 21.)+pose['tangent']], 8)) for pose in poses]
    images = [(camera.scan(pose['position'], angles, uv=True), camera.scan(pose['position'], angles))
              for pose, angles in zip(poses, angles_list)]
    records, teaching = [], []
    for seed in [19, 31, 43]:
        api = AngularEncoder('linear_colour', seed=seed)
        sobel = AngularEncoder('sobel_colour', seed=seed)
        uv_model = base.make_model(p, 'apiaviz_uv', seed)
        for a, b in zip(api.features.legacy, sobel.features.legacy):
            assert torch.equal(a.connection, b.connection)
        frozen = [fingerprint(m) for m in (api, sobel, uv_model)]
        def coding(uv, rgb):
            bee = uv/(uv+1.)
            a = encode(api, rgb)
            b = encode(api, bee[:, [2, 1]])
            u = encode(uv_model, uv)[:, 8000:]
            return dict(angular_rgb=a, angular_bee=b, angular_sobel=encode(sobel, rgb),
                        angular_bee_plus_uv=torch.cat([b, u], 1), angular_rgb_plus_uv=torch.cat([a, u], 1))
        train = coding(bank['uv'], bank['rgb'])
        memory = {name: binary(c) for name, c in train.items()}
        for name, c in train.items():
            sim = memory[name]@memory[name].T
            far = abs(torch.arange(len(c))[:, None]-torch.arange(len(c))[None, :]) >= 10
            teaching.append(dict(seed=seed, condition=name, activity=float((c>0).float().mean()),
                                 far_place_overlap=float(sim[far].mean()), model_fingerprints=frozen))
        for pose, angles, (uv, rgb) in tqdm(list(zip(poses, angles_list, images)), desc=f'angular {w["name"]} {seed}'):
            for name, c in coding(uv, rgb).items():
                record = metrics(binary(c)@memory[name].T, angles, pose, np.asarray(w['route']), anchor)
                records.append(dict(world=w['name'], seed=seed, condition=name, **pose, angles=angles.tolist(), **record))
        assert frozen == [fingerprint(m) for m in (api, sobel, uv_model)]
    base.atomic_json(out/f'{w["name"]}.json', dict(records=records, teaching=teaching,
        raw_records=camera.references, teaching_sha256=file_sha(env/'teaching.pt')))
    return w['name']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parent = json.loads((source/'protocol.json').read_text())
    p = json.loads((Path(parent['study'])/'protocol.json').read_text())
    sources = base.sources()
    for filename in ('visual_response_controls.py', 'visual_angular_controls.py'):
        sources['scripts/'+filename] = file_sha(ROOT/'scripts'/filename)
    base.atomic_json(out/'protocol.json', dict(schema='visual-angular-controls-v1', source=str(source),
        source_protocol_sha256=file_sha(source/'protocol.json'), conditions=CONDITIONS, source_sha256=sources,
        rationale='Earlier navigation and cancelled route-full protocols specify angles mode; UV trial dispatch instead uses pixel kernels. Test that omitted implementation without retuning it.',
        selection='Exactly the parent protocol poses, seeds, scans, inputs and metric definitions.',
        limitations='Offline probes, not full navigation. Appended UV stream retains current pixel filters, isolating only the omitted visible AngularEncoder. No fitted parameters.'))
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sources: z.write(ROOT/name, name)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(job, str(source), str(out), w) for w in p['worlds']]
        for future in futures: print('Completed', future.result(), flush=True)
    base.atomic_json(out/'complete.json', dict(protocol_sha256=file_sha(out/'protocol.json'),
        results={w['name']: file_sha(out/f'{w["name"]}.json') for w in p['worlds']}))


if __name__ == '__main__': main()
