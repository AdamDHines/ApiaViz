"""Cache-only sun visibility and matched-pose retrieval audit. No trial launches.

All six ApiaViz and all six Sobel aligned v3 trajectories contribute positions:
first 20, every tenth subsequent endpoint, and final endpoint. All representations
see every selected pose. These correlated retrospective probes are not new trials.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from apiaviz.research.dual_camera import DualCamera, visible_image
from apiaviz.research.uv_input import sample_receptors
from apiaviz.research.uv_trials import make_model, atomic_json, sources
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.familiarity_controller import acquisition_calibration, FamiliarityController
from apiaviz.research.study import encode
from apiaviz.research.spectral_input import file_sha


def binary(x):
    return F.normalize((x > 0).float(), dim=1)


def run(study, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    p = json.loads((study/'protocol.json').read_text())
    result = dict(protocol_sha256=file_sha(study/'protocol.json'), source_sha256=sources(),
                  script_sha256=file_sha(Path(__file__)), inputs={}, records=[], sun=[],
                  selection=__doc__, controls={
                      'uv10k': 'Unchanged v2 ApiaViz, empirical UV/B/G, 4000/4000/2000 KCs',
                      'bee_visible8k': 'Same ApiaViz code with UV KCs excluded at retrieval',
                      'uv_only2k': 'Same ApiaViz code, UV stream only at retrieval',
                      'linear_rgb8k': 'Preferred ApiaViz linear colour, same visible RGB as Sobel, 4000/4000 KCs',
                      'sobel_rgb8k': 'Visible-only Sobel, 4000/4000 KCs; matched wiring to linear_rgb8k',
                      'sobel_rgb10k': 'Unchanged trial Sobel, visible RGB, 5000/5000 KCs'})
    with zipfile.ZipFile(out/'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in result['source_sha256']: z.write(ROOT/name, name)
    models = dict(uv10k=make_model(p, 'apiaviz_uv', 19),
                  linear_rgb8k=RefinementEncoder('linear_colour', seed=19, code_dim=8000),
                  sobel_rgb8k=RefinementEncoder('sobel_colour', seed=19, code_dim=8000),
                  sobel_rgb10k=make_model(p, 'sobel_colour', 19))
    # The common-RGB comparison changes features only, not projection or circuit.
    for a, b in zip(models['linear_rgb8k'].features.legacy, models['sobel_rgb8k'].features.legacy):
        assert torch.equal(a.connection, b.connection)
    assert models['linear_rgb8k'].circuit_config == models['sobel_rgb8k'].circuit_config
    offsets = np.arange(-180., 180., 10.)
    for w in p['worlds']:
        env = study/'worlds'/w['name']
        bp = env/'teaching.pt'; result['inputs'][str(bp)] = file_sha(bp)
        bank = torch.load(bp, weights_only=True)
        taught = {name: encode(m, bank['uv' if name == 'uv10k' else 'rgb']) for name, m in models.items()}
        taught.update(bee_visible8k=taught['uv10k'][:, :8000], uv_only2k=taught['uv10k'][:, 8000:])
        calibration = {name: acquisition_calibration(c) for name, c in taught.items()}
        memory = {name: binary(c) for name, c in taught.items()}
        camera = DualCamera(env, json.loads((env/'render.json').read_text()))
        result['inputs'][str(env/'render.json')] = file_sha(env/'render.json')
        route = np.array(w['route']); headings = np.array(w['headings'])
        query_cache = {}
        for method in ('apiaviz_uv', 'sobel_colour'):
            for phase in (-1, 1):
                path = study/'trials'/f'{w["name"]}-19-{method}-aligned-familiarity-{phase}.json'
                d = json.loads(path.read_text()); result['inputs'][str(path)] = file_sha(path)
                selected = [r for r in d['trace'] if r['step'] <= 20 or r['step'] % 10 == 0 or r is d['trace'][-1]]
                for r in tqdm(selected, desc=path.stem):
                    pos = r['position']; station = int(np.linalg.norm(route-pos, axis=1).argmin())
                    key = (camera.key(pos), station)
                    if key not in query_cache:
                        angles = headings[station]+offsets
                        uv = camera.scan(pos, angles, uv=True); rgb = camera.scan(pos, angles)
                        query = {name: encode(m, uv if name == 'uv10k' else rgb) for name, m in models.items()}
                        query.update(bee_visible8k=query['uv10k'][:, :8000], uv_only2k=query['uv10k'][:, 8000:])
                        metrics = {}
                        for name, codes in query.items():
                            matrix = binary(codes)@memory[name].T
                            scores, matches = matrix.max(1)
                            i = int(scores.argmax())
                            samples = {float(o): float(s) for o, s in zip(offsets, scores)}
                            target, contrast, supported = FamiliarityController(calibration[name])._peak(samples, 0.)
                            metrics[name] = dict(best_offset_deg=float(offsets[i]), matched_station=int(matches[i]),
                                tangent_margin=float(scores[18]-scores[np.abs(offsets) > 20].max()),
                                scores=scores.tolist(), matches=matches.tolist(),
                                scale=calibration[name]['scale'], contrast=contrast, supported=bool(supported),
                                policy_target_offset=target)
                        query_cache[key] = metrics
                    result['records'].append(dict(world=w['name'], source_method=method, phase=phase,
                        step=r['step'], position=pos, polyline_m=r['polyline_m'], nearest_station=station,
                        streams=query_cache[key]))
        # All three original acquisition stations, in the independent audit cache.
        for key, value in camera.references.items(): result['inputs'][str(env/'camera'/f'{key}.json')] = value
        env = ROOT/'apiaviz/output/uv-acquisition-v2-check1'/f'{w["name"]}-64-20261002'
        camera = DualCamera(env, json.loads((env/'render.json').read_text()))
        result['inputs'][str(env/'render.json')] = file_sha(env/'render.json')
        for station in (2, 3, len(route)//2):
            raw = camera.frame(route[station]); cfg = camera.config
            h, width = raw.shape[:2]
            lum = raw[:, :, 3:].mean(2); y, x = np.unravel_index(lum.argmax(), lum.shape)
            row = dict(world=w['name'], station=station, peak_azimuth=180-(x+.5)*360/width,
                       peak_elevation=cfg['elevation_deg'][0]-(y+.5)*110/h,
                       raw_peak=float(lum.max()), panorama_pixel_deg=[360/width, 110/h])
            scan_heading = np.arange(0., 360.)
            in_view = np.abs((cfg['sun_azimuth']-scan_heading+180)%360-180) < 145
            before = sample_receptors(visible_image(raw[:, :, 3:]), scan_heading)
            linear = sample_receptors(raw[:, :, 3:], scan_heading)
            after = linear/(linear+1)
            # Upper sky only; avoids bright surfaces being called a solar detection.
            skyrows = np.linspace(60, -15, 51) > 25
            peaks_before = before[:, :, skyrows].mean(1).flatten(1).max(1).values.numpy()
            peaks_after = after[:, :, skyrows].mean(1).flatten(1).max(1).values.numpy()
            row.update(headings=scan_heading.tolist(), sun_in_view=in_view.tolist(),
                       peak_response_before=peaks_before.tolist(), peak_response_after=peaks_after.tolist(),
                       in_view_peak_before_range=[float(peaks_before[in_view].min()), float(peaks_before[in_view].max())],
                       in_view_peak_after_range=[float(peaks_after[in_view].min()), float(peaks_after[in_view].max())])
            result['sun'].append(row)
            if w['name']=='meander' and station==2:
                sun_figure(out, raw, cfg, headings[station], row)
        for key, value in camera.references.items(): result['inputs'][str(env/'camera'/f'{key}.json')] = value
    summary = []
    for source in ('all', 'apiaviz_uv', 'sobel_colour'):
        for near in (True, False):
            rows = [r for r in result['records'] if (source=='all' or r['source_method']==source) and (r['polyline_m']<=.05)==near]
            for name in result['controls']:
                summary.append(dict(source=source, near=near, method=name, n=len(rows),
                    within20=sum(abs(r['streams'][name]['best_offset_deg'])<=20 for r in rows),
                    supported=sum(r['streams'][name]['supported'] for r in rows),
                    median_margin=float(np.median([r['streams'][name]['tangent_margin'] for r in rows])),
                    median_station_error=float(np.median([abs(r['streams'][name]['matched_station']-r['nearest_station']) for r in rows]))))
    result['summary'] = summary
    atomic_json(out/'audit.json', result)
    print(json.dumps(summary, indent=2))


def sun_figure(out, raw, cfg, heading, row):
    os.environ.setdefault('MPLCONFIGDIR', str(out/'.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    rgb = visible_image(raw[:, :, 3:])
    # Explicit display-only sRGB transfer, identical for every pixel/view.
    display = np.where(rgb<=.0031308, 12.92*rgb, 1.055*rgb**(1/2.4)-.055)
    ax[0].imshow(display[:, ::-1], extent=(-180, 180, -20, 90), aspect='auto')
    ax[0].plot([cfg['sun_azimuth']], [cfg['sun_elevation']], 'o', mfc='none', mec='red', ms=14)
    ax[0].set(title='Full cached visible panorama · sun ring is an annotation · fixed sRGB display transfer', ylabel='Elevation (°)')
    retina=sample_receptors(rgb,[heading])[0].permute(1,2,0).numpy()[:,::-1]
    # Display columns run positive to negative azimuth after the photograph flip.
    # Extend by half a receptor spacing so the labelled endpoints are pixel centres.
    dx=296/198/2;dy=75/50/2
    ax[1].imshow(retina,extent=(148+dx,-148-dx,-15-dy,60+dy),aspect='auto')
    relative=(cfg['sun_azimuth']-heading+180)%360-180
    ax[1].plot([relative],[cfg['sun_elevation']],'o',mfc='none',mec='red',ms=14)
    ax[1].set(title='Existing movie / visible policy values · 199 × 51 · no display transfer',ylabel='Elevation (°)')
    for key,label in [('peak_response_before','Existing visible: respond, then sample'),('peak_response_after','Control: sample linear radiance, then respond')]:
        values=np.array(row[key]); values[~np.array(row['sun_in_view'])]=np.nan
        ax[2].plot(row['headings'], values,label=label)
    ax[2].set(xlabel='Camera heading (°)',ylabel='Brightest upper-sky response',ylim=(0,1.05),title='Same raw panorama at 360 headings; no sun movement, no new rendering')
    ax[2].legend()
    fig.savefig(out/'sun-sampling.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/coherent-validation-v3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.study,args.output)
