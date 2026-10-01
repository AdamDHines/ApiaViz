"""Persistent CPU spectral camera worker, owned by the trials command."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import render as base
sys.path.insert(0,str(base.ROOT))
from apiaviz.research.spectral_input import Calibration,file_sha
from apiaviz.research.dual_camera import BANDS,digest,position_key,visible_weights,canonical_position,render_seed


def send(value): print('APIAVIZ '+json.dumps(value,allow_nan=False),flush=True)


def run(env):
    config=json.loads((env/'render.json').read_text()); render_hash=digest(config)
    calibration=Calibration(env/'calibration.json')
    if calibration.sha256!=config['calibration_sha256']: raise ValueError('Calibration hash changed')
    for name,value in config['geometry_sha256'].items():
        if file_sha(env/'geometry'/name)!=value: raise ValueError('Spectral geometry changed')
    base.mi.set_variant('llvm_ad_spectral'); base.dr.set_thread_count(config['threads']); base.register_camera()
    weights=base.np.concatenate([calibration.weights,visible_weights(calibration.wavelengths)])
    scene,_=base.build_scene(env/'geometry',dict(position=[0,0,.01],heading=0),
        config['width'],config['height'],config['sun_azimuth'],config['sun_elevation'],config['spp'],
        receptor_weights=weights,material_lookup=calibration.material,
        elevation=tuple(reversed(config['elevation_deg'])),
        response_names=[f'band{i+1}_{name}' for i,name in enumerate(BANDS)],
        surface_detail=config.get('surface_detail')=='source-informed-surfaces-v1')
    cache=env/'camera'; cache.mkdir(exist_ok=True)
    send(dict(ok=True,render_hash=render_hash))
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'): break
        try:
            decimals=config.get('pose_decimals')
            position=canonical_position(request['position'],decimals); key=position_key(position,decimals)
            target=cache/f'{key}.json'
            if target.exists(): raise FileExistsError('Parent requested an already completed cache frame')
            # Runtime parameter, not a new compile-time constant at every pose.
            scene.sensors()[0].origin=base.dr.opaque(base.mi.Point3f,[*position,.01])
            ray,_=scene.sensors()[0].sample_ray(0.,.5,base.mi.Point2f(.5,.5),base.mi.Point2f(0.,0.))
            if not base.np.allclose(base.np.array(ray.o).reshape(3),[*position,.01],atol=1e-6,rtol=0):
                raise AssertionError('Camera origin did not update')
            seed=render_seed(key,config)
            started=time.perf_counter()
            data=base.np.array(base.mi.render(scene,seed=seed,spp=config['spp']))
            elapsed=time.perf_counter()-started
            if data.shape!=(config['height'],config['width'],6) or not base.np.isfinite(data).all() or base.np.any(data<0):
                raise ValueError('Unexpected six-band spectral image')
            array=cache/f'{key}.npz'
            with array.with_suffix('.npz.tmp').open('wb') as handle: base.np.savez_compressed(handle,responses=data)
            array.with_suffix('.npz.tmp').replace(array)
            record=dict(schema='apiaviz-dual-frame-v1',render_hash=render_hash,channels=BANDS,
                position=position,camera_xyz=[*position,.01],heading=0.,seed=seed,shape=list(data.shape),
                array_sha256=file_sha(array),linear=True,units='relative unit-area photon-weighted band radiance',
                render_wall_s=elapsed,versions=dict(mitsuba=base.mi.__version__,drjit=base.dr.__version__))
            if decimals is not None:
                record.update(schema='apiaviz-dual-frame-v2',pose_decimals=decimals,seed_policy=config['seed_policy'])
            temporary=target.with_suffix('.json.tmp'); temporary.write_text(json.dumps(record,indent=2)+'\n'); temporary.replace(target)
            send(dict(ok=True,key=key))
        except Exception:
            send(dict(ok=False,error=traceback.format_exc()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--environment',type=Path,required=True)
    try: run(p.parse_args().environment.resolve())
    except Exception: send(dict(ok=False,error=traceback.format_exc())); raise
