"""Paired flat/detailed renders at the first recorded bend rock-contact approach.

Evaluator trace selects diagnostic poses only. No geometry reaches a policy.
This renders four images, not a navigation trial or a tuned obstacle fixture.
"""
import argparse
import json
from pathlib import Path
import sys
import render as base
sys.path.insert(0,str(base.ROOT))
from apiaviz.research.spectral_input import Calibration,file_sha
from apiaviz.research.dual_camera import visible_weights,BANDS


def run(study,trace,out):
    out.mkdir(parents=True,exist_ok=False)
    np,mi,dr=base.np,base.mi,base.dr
    d=json.loads(trace.read_text());event=next(e for e in d['events'] if e['kind']=='blocked_proposal' and e['reason']=='rock_contact')
    # Last successful translated pair at the same gaze preceding the rejection.
    views=[e for e in d['events'] if e['kind']=='avoidance_observation' and e['time_s']<=event['time_s'] and abs(e['heading']-event['heading'])<1e-8]
    after=views[-1]['position'];before=next(e['position'] for e in reversed(views[:-1]) if np.linalg.norm(np.array(e['position'])-after)>1e-8)
    delta=np.array(after)-before;heading=float(event['heading']);distance=float(np.linalg.norm(delta))
    np.testing.assert_allclose(delta/distance,[np.cos(np.deg2rad(heading)),np.sin(np.deg2rad(heading))],atol=1e-6)
    env=study/'worlds/bend';cfg=json.loads((env/'render.json').read_text());cal=Calibration(env/'calibration.json')
    mi.set_variant('llvm_ad_spectral');dr.set_thread_count(2);base.register_camera()
    weights=np.concatenate([cal.weights,visible_weights(cal.wavelengths)])
    records=[]
    for detailed in (False,True):
        scene,_=base.build_scene(env/'geometry',dict(position=[*before,.01],heading=0),cfg['width'],cfg['height'],cfg['sun_azimuth'],cfg['sun_elevation'],cfg['spp'],
            receptor_weights=weights,material_lookup=cal.material,
            response_names=[f'band{i+1}_{name}' for i,name in enumerate(BANDS)],surface_detail=detailed)
        for name,position in [('before',before),('after',after)]:
            scene.sensors()[0].origin=dr.opaque(mi.Point3f,[*position,.01])
            raw=np.array(mi.render(scene,seed=cfg['seed'],spp=cfg['spp']))
            path=out/f'{"detailed" if detailed else "flat"}-{name}.npy';np.save(path,raw)
            records.append(dict(file=path.name,sha256=file_sha(path),position=position,detailed=detailed))
    (out/'renders.json').write_text(json.dumps(dict(trace_sha256=file_sha(trace),geometry_sha256=file_sha(env/'geometry/geometry.json'),
        script_sha256=file_sha(Path(__file__)),render_config=cfg,heading=heading,distance_m=distance,records=records),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=base.ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--trace',type=Path,default=base.ROOT/'apiaviz/output/uv-validation-v2/trials/bend-19-apiaviz_uv-aligned-familiarity-1.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.study,a.trace,a.output)
