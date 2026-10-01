"""Read-only stage/heading audit on independent spectral acquisitions.

Uses all nine prespecified acquisition stations, not successful navigation cases.
No renderer is launched. Route headings are evaluator labels, never policy input.
"""
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.uv_trials import make_model, atomic_json, sources
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.familiarity_controller import acquisition_calibration
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize


def similarity(a, b):
    return F.normalize((a > 0).float(), dim=1) @ F.normalize((b > 0).float(), dim=1).T


def run(study, acquisition, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    p = json.loads((study/'protocol.json').read_text())
    manifest = dict(source_protocol_sha256=file_sha(study/'protocol.json'),
                    acquisition_sha256=file_sha(acquisition/'validation.json'),
                    source_sha256=sources(), inputs={}, records=[])
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in manifest['source_sha256']: z.write(ROOT/name, name)
    offsets = np.arange(-180., 180., 10.)
    for w in p['worlds']:
        bank_path = study/'worlds'/w['name']/'teaching.pt'
        bank = torch.load(bank_path, weights_only=True)
        manifest['inputs'][str(bank_path)] = file_sha(bank_path)
        for method in p['methods']:
            model = make_model(p, method, 19)
            key = 'uv' if method == 'apiaviz_uv' else 'rgb'
            taught = encode(model, bank[key])
            calibration = acquisition_calibration(taught)
            for spp in (64, 256):
                env = acquisition/f'{w["name"]}-{spp}-20261002'
                camera = DualCamera(env, json.loads((env/'render.json').read_text()))
                for station in [2, 3, len(w['route'])//2]:
                    images = camera.scan(w['route'][station], w['headings'][station]+offsets,
                                         uv=method == 'apiaviz_uv')
                    query = encode(model, images)
                    slices = {'all': slice(None)}
                    if method == 'apiaviz_uv':
                        slices.update(visible=slice(0, 8000), form=slice(0, 4000),
                                      colour=slice(4000, 8000), uv=slice(8000, 10000))
                    row = dict(world=w['name'], method=method, spp=spp, station=station,
                               calibration=calibration, streams={})
                    for name, sl in slices.items():
                        matrix = similarity(query[:, sl], taught[:, sl])
                        score, match = matrix.max(1)
                        i = int(score.argmax()); zero = int(np.flatnonzero(offsets == 0)[0])
                        row['streams'][name] = dict(best_offset_deg=float(offsets[i]),
                            matched_station=int(match[i]),
                            own_station_at_correct_heading=float(matrix[zero, station]),
                            tangent_margin=float(score[zero]-score[np.abs(offsets) >= 20].max()),
                            scores=score.tolist(), spike_count_at_correct_heading=int((query[zero, sl] > 0).sum()))
                    if method == 'apiaviz_uv':
                        maps = model.backbone(images[zero:zero+1])
                        row['pooled_plane_energy_fraction'] = {}
                        for name, projection in zip(model.stream_names, model.projections):
                            pooled = adaptive_avg_pool2d_anysize(maps[name], projection.pool_hw)
                            energy = pooled.square().sum((0, 2, 3))
                            row['pooled_plane_energy_fraction'][name] = (energy/energy.sum().clamp_min(1e-12)).tolist()
                        raw = camera.frame(w['route'][station])[:, :, :3]
                        # Noncommuting acquisition control: respond at render pixels
                        # before retinal interpolation instead of after interpolation.
                        from apiaviz.research.uv_input import sample_receptors
                        bounded_first = sample_receptors(raw/(raw+1), [w['headings'][station]],
                            elevation=camera.config['elevation_deg'])
                        bounded_last = images[zero:zero+1]/(images[zero:zero+1]+1)
                        row['response_order_abs_difference'] = dict(mean=float((bounded_first-bounded_last).abs().mean()),
                            max=float((bounded_first-bounded_last).abs().max()))
                    manifest['records'].append(row)
                for name, digest in camera.references.items():
                    manifest['inputs'][str(env/'camera'/f'{name}.json')] = digest
    atomic_json(out/'audit.json', manifest)
    for method in p['methods']:
        for spp in (64, 256):
            rows = [r for r in manifest['records'] if r['method']==method and r['spp']==spp]
            for name in rows[0]['streams']:
                errors = [abs(r['streams'][name]['best_offset_deg']) for r in rows]
                margins = [r['streams'][name]['tangent_margin'] for r in rows]
                print(method, spp, name, 'heading errors', errors, 'median margin', np.median(margins), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT/'apiaviz/output/uv-validation-v2')
    parser.add_argument('--acquisition', type=Path, default=ROOT/'apiaviz/output/uv-acquisition-v2-check1')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); run(a.study, a.acquisition, a.output)
