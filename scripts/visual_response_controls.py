"""Matched visual-response controls, not navigation trials or a parameter search.

Fixed route quantiles, both lateral sides and between-teaching positions are
selected before responses are inspected. Geometry only labels offline probes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import DualCamera, sensor_views
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.mechanisms import polyline_distance
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.study import encode, fingerprint
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize

CONDITIONS = {
    'api_uv': 'Current bee UV/B/G ApiaViz, 4000/4000/2000 cells',
    'api_bee': 'Same code and memory with UV stream excluded, 8000 cells',
    'uv_only': 'Current UV stream alone, 2000 cells; includes blue/green opponent information',
    'blue_replaces_uv': 'UV radiance replaced by blue in teaching AND recall; unchanged third-stream wiring',
    'api_rgb': 'Preferred linear-colour ApiaViz on exactly Sobel RGB, 8000 cells',
    'rgb_plus_uv': 'Identical api_rgb visible code plus current UV stream, 10000 cells',
    'sobel': 'RGB Sobel + colour, 8000 cells and identical api_rgb wiring',
    'ardin': 'RGB Ardin-style preprocessing, same 8000-cell wiring',
    'sobel10k': 'Production RGB Sobel, 10000 cells',
    'ardin10k': 'Production RGB Ardin-style, 10000 cells',
    'api_rgb_small': 'api_rgb after area reduction to historical 18x74; same wiring',
    'sobel_small': 'sobel after identical 18x74 area reduction; same wiring',
}


def binary(x):
    return F.normalize((x > 0).float(), dim=1)


def select_poses(w):
    route = np.asarray(w['route'])
    records = []
    for station in np.rint(np.linspace(2, len(route)-3, 7)).astype(int):
        tangent = float(w['headings'][station])
        normal = np.array([-np.sin(np.deg2rad(tangent)), np.cos(np.deg2rad(tangent))])
        for kind, pos in [('taught', route[station]),
                          ('midpoint', .5*(route[station]+route[station+1])),
                          ('left20', route[station]+.2*normal),
                          ('right20', route[station]-.2*normal)]:
            records.append(dict(station=int(station), kind=kind, position=pos.tolist(), tangent=tangent))
    return records


def winner(scores, headings, anchor):
    """Fixed tie rule: least turn, then positive turn. No route-dependent tie."""
    turns = (np.asarray(headings)-anchor+180) % 360-180
    return min(range(len(scores)), key=lambda i: (-float(scores[i]), abs(turns[i]), -turns[i]))


def metrics(matrix, angles, pose, route, anchor):
    a = np.asarray(angles)
    scores = matrix.max(1).values.numpy()
    matches = matrix.argmax(1).numpy()
    turns = (a-anchor+180) % 360-180
    result = {}
    for label, mask in [('grid5', np.isclose(turns/5, np.rint(turns/5))),
                        ('grid10', np.isclose(turns/10, np.rint(turns/10))),
                        ('grid10_shift5', np.isclose((turns-5)/10, np.rint((turns-5)/10)))]:
        indices = np.flatnonzero(mask)
        i = int(indices[winner(scores[indices], a[indices], anchor)])
        pos = np.asarray(pose['position'])
        proposed = pos+.1*np.array([np.cos(np.deg2rad(a[i])), np.sin(np.deg2rad(a[i]))])
        before, after = polyline_distance([pos, proposed], route)
        result[label] = dict(heading=float(a[i]), error=abs(float((a[i]-pose['tangent']+180)%360-180)),
                            score=float(scores[i]), matched_station=int(matches[i]),
                            before_m=float(before), after_m=float(after), inward=bool(after < before-1e-9))
    # Exact correct view vs far-away place, independent recall seed.
    offsets = (a-pose['tangent']+180) % 360-180
    zero = int(np.argmin(abs(offsets)))
    far = np.abs(np.arange(matrix.shape[1])-pose['station']) >= 10
    result['correct_station_score'] = float(matrix[zero, pose['station']])
    result['far_station_score'] = float(matrix[zero, far].max())
    result['place_margin'] = result['correct_station_score']-result['far_station_score']
    # Response to yaw measured against the same taught view, not max over places.
    curve = []
    for offset in range(-20, 21):
        i = int(np.argmin(abs(offsets-offset)))
        assert abs(offsets[i]-offset) < 1e-5
        curve.append(float(matrix[i, pose['station']]))
    result['same_station_yaw_curve'] = curve
    result['five_degree_loss'] = curve[20]-.5*(curve[15]+curve[25])
    result['scores'] = scores.tolist()
    result['matches'] = matches.tolist()
    return result


def codes(models, uv, rgb):
    current = encode(models['uv'], uv)
    replacement = uv.clone()
    replacement[:, 0] = replacement[:, 1]
    rgb_code = encode(models['api'], rgb)
    small = F.interpolate(rgb, size=(18, 74), mode='area')
    return dict(api_uv=current, api_bee=current[:, :8000], uv_only=current[:, 8000:],
                blue_replaces_uv=encode(models['uv'], replacement), api_rgb=rgb_code,
                rgb_plus_uv=torch.cat([rgb_code, current[:, 8000:]], 1),
                sobel=encode(models['sobel'], rgb), ardin=encode(models['ardin'], rgb),
                sobel10k=encode(models['sobel10k'], rgb), ardin10k=encode(models['ardin10k'], rgb),
                api_rgb_small=encode(models['api'], small), sobel_small=encode(models['sobel'], small))


def feature_stats(model, images, uv=False):
    if uv:
        maps = model.backbone(images)
        maps = [adaptive_avg_pool2d_anysize(maps[name], (8, 64)) for name in model.stream_names]
    else:
        maps = model.pooled_features(images)
    result = []
    for x in maps:
        centered = x-x.flatten(1).mean(1)[:, None, None, None]
        energy = centered.square().sum((2, 3))
        fraction = energy/energy.sum(1, keepdim=True).clamp_min(1e-12)
        flat = F.normalize(centered.flatten(1), dim=1)
        sim = flat@flat.T
        far = abs(torch.arange(len(x))[:, None]-torch.arange(len(x))[None, :]) >= 10
        result.append(dict(plane_energy_fraction=fraction.mean(0).tolist(),
                           far_place_feature_cosine=float(sim[far].mean())))
    return result


def world_job(study_string, output_string, w):
    torch.set_num_threads(2)
    study, output = Path(study_string), Path(output_string)
    p = json.loads((study/'protocol.json').read_text())
    env = study/'worlds'/w['name']
    out = output/w['name']; out.mkdir()
    config = json.loads((env/'render.json').read_text())
    shutil.copytree(env/'geometry', out/'geometry')
    shutil.copyfile(env/'calibration.json', out/'calibration.json')
    base.atomic_json(out/'render.json', config)
    poses = select_poses(w)
    anchor = float(w['headings'][0])
    angles_by_pose = [np.unique(np.round(np.r_[np.arange(-180., 180., 5.)+anchor,
                    np.arange(-20., 21.)+pose['tangent']], 8)) for pose in poses]
    queries = []
    with DualCamera(out, config) as camera:
        for pose, angles in tqdm(list(zip(poses, angles_by_pose)), desc=w['name']+' images'):
            key = camera.key(pose['position'])
            if (env/'camera'/f'{key}.json').exists():
                for suffix in ('.json', '.npz'):
                    shutil.copyfile(env/'camera'/f'{key}{suffix}', out/'camera'/f'{key}{suffix}')
            raw = camera.frame(pose['position'])
            queries.append((sensor_views(raw, angles, config, uv=True), sensor_views(raw, angles, config)))
        raw_records = dict(camera.references)
    bank = torch.load(env/'teaching.pt', weights_only=True)
    records, teaching, feature = [], [], {}
    for seed in tqdm([19, 31, 43], desc=w['name']+' seeds'):
        models = dict(uv=base.make_model(p, 'apiaviz_uv', seed),
            **{name: RefinementEncoder(method, seed=seed, code_dim=size)
               for name, method, size in [('api', 'linear_colour', 8000), ('sobel', 'sobel_colour', 8000),
                   ('ardin', 'ardin_input', 8000), ('sobel10k', 'sobel_colour', 10000), ('ardin10k', 'ardin_input', 10000)]})
        for m in ('sobel', 'ardin'):
            for a, b in zip(models['api'].features.legacy, models[m].features.legacy):
                assert torch.equal(a.connection, b.connection)
            assert models['api'].circuit_config == models[m].circuit_config
        for a, b in zip(models['uv'].projections, models['api'].features.legacy):
            assert torch.equal(a.connection, b.connection)
        frozen = {name: fingerprint(m) for name, m in models.items()}
        train = codes(models, bank['uv'], bank['rgb'])
        memory = {name: binary(c) for name, c in train.items()}
        for name, c in train.items():
            sim = memory[name]@memory[name].T
            far = abs(torch.arange(len(c))[:, None]-torch.arange(len(c))[None, :]) >= 10
            teaching.append(dict(seed=seed, condition=name, activity=float((c>0).float().mean()),
                far_place_overlap=float(sim[far].mean()), model_fingerprints=frozen))
        if seed == 19:
            feature = dict(api_uv=feature_stats(models['uv'], bank['uv'], True),
                           api_rgb=feature_stats(models['api'], bank['rgb']),
                           sobel=feature_stats(models['sobel'], bank['rgb']))
            # Direct visible-path implementation identity, holding input bands fixed.
            bee = bank['uv']/(bank['uv']+1.)
            matched = encode(models['api'], bee[:, [2, 1]])
            assert torch.equal(matched>0, train['api_bee']>0), 'Visible pipeline changed beyond its input bands'
            torch.save(dict(uv=bank['uv'][len(bank['uv'])//2], rgb=bank['rgb'][len(bank['rgb'])//2],
                uv_maps=models['uv'].backbone(bank['uv'][len(bank['uv'])//2:len(bank['uv'])//2+1]),
                api_maps=models['api'].pooled_features(bank['rgb'][len(bank['rgb'])//2:len(bank['rgb'])//2+1]),
                sobel_maps=models['sobel'].pooled_features(bank['rgb'][len(bank['rgb'])//2:len(bank['rgb'])//2+1])), out/'feature-example.pt')
        for pose, angles, (uv, rgb) in tqdm(list(zip(poses, angles_by_pose, queries)), desc=f'{w["name"]} {seed}', leave=False):
            query = codes(models, uv, rgb)
            for name, c in query.items():
                record = metrics(binary(c)@memory[name].T, angles, pose, np.asarray(w['route']), anchor)
                records.append(dict(world=w['name'], seed=seed, condition=name, **pose,
                                    angles=angles.tolist(), **record))
        assert frozen == {name: fingerprint(m) for name, m in models.items()}
    result = dict(records=records, teaching=teaching, features=feature, raw_records=raw_records,
        source_teaching_sha256=file_sha(env/'teaching.pt'), render_sha256=file_sha(env/'render.json'),
        visible_implementation_identity='Exact spike bits on all teaching images for seed 19 with matched B/G inputs')
    base.atomic_json(out/'results.json', result)
    return w['name']


def run(study, out):
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((study/'protocol.json').read_text())
    base.verify_sources(p)
    src = base.sources(); src[str(Path(__file__).relative_to(ROOT))] = file_sha(Path(__file__))
    protocol = dict(schema='visual-response-controls-v1', study=str(study), study_sha256=file_sha(study/'protocol.json'),
        source_sha256=src, seeds=[19, 31, 43], conditions=CONDITIONS,
        poses={w['name']: select_poses(w) for w in p['worlds']},
        selection='Seven evenly spaced route station indices, excluding first/last two; at taught position, segment midpoint, and both 20cm lateral sides.',
        scan='Uniform full-circle 5deg grid anchored to initial world heading; both interleaved 10deg grids. Extra +-20deg at 1deg around teaching tangent ONLY for angular response diagnostic.',
        readout='Same normalized binary-spike overlap; each condition learns its own codes from identical teaching poses. Query images use independent recall seed.',
        limitations=['Offline image probes, no movement, arrival claims, or controller budget comparison.',
            'Route coordinates only select/score probes; diagnostic dense tangent sampling is not a deployable policy.',
            'Three development worlds, three paired wirings; do not treat images or seeds as independent environments.',
            'UV-versus-visible ablations have different cell counts; RGB versus Sobel/Ardin 8k controls match wiring exactly.',
            'Blue-for-UV replaces both spectrum and intensity; tests incremental UV signal against a visible-derived stream, not a calibrated biological alternative.',
            'Small-image control changes angular filter scale and loses detail jointly; not an exact old renderer reconstruction.'])
    base.atomic_json(out/'protocol.json', protocol)
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in src: archive.write(ROOT/name, name)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(world_job, str(study), str(out), w) for w in p['worlds']]
        for future in futures: print('Completed', future.result(), flush=True)
    base.atomic_json(out/'complete.json', dict(protocol_sha256=file_sha(out/'protocol.json'),
        results={w['name']: file_sha(out/w['name']/'results.json') for w in p['worlds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study.resolve(), args.output.resolve())
