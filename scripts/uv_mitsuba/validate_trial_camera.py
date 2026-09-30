"""Six-band numerical isolation and optional bounded camera/JIT equivalence check."""
import argparse
import json
from pathlib import Path
import sys
import time

import render as base
sys.path.insert(0,str(base.ROOT))
from apiaviz.research.spectral_input import Calibration,file_sha
from apiaviz.research.dual_camera import BANDS,position_key,visible_weights


def validate(environment=None):
    mi,np=base.mi,base.np
    c=Calibration(base.ROOT/'docs/uv-calibration/calibration.json')
    mi.set_variant('llvm_ad_spectral'); base.dr.set_thread_count(2); base.register_camera()
    weights=np.concatenate([c.weights,visible_weights(c.wavelengths)])
    names=[f'band{i+1}_{name}' for i,name in enumerate(BANDS)]
    film=dict(type='specfilm',width=16,height=8,component_format='float32',rfilter=dict(type='box'))
    for name,curve in zip(names,weights): film[name]=base.spectrum(c.wavelengths,curve)
    checks=[]
    for kind in ('flat','uv_only'):
        values=np.ones(len(c.wavelengths))
        if kind=='uv_only': values[c.wavelengths>=400]=0.
        scene=mi.load_dict(dict(type='scene',integrator=dict(type='path'),
            sensor=dict(type='uv_panorama',film=film,sampler=dict(type='independent',sample_count=2048)),
            emitter=dict(type='constant',radiance=base.spectrum(c.wavelengths,values))))
        observed=np.array(mi.render(scene,seed=173,spp=2048)).mean((0,1))
        fine=np.linspace(320,700,38001)
        expected=np.array([np.trapz(np.interp(fine,c.wavelengths,w)*np.interp(fine,c.wavelengths,values),fine) for w in weights])
        np.testing.assert_allclose(observed,expected,atol=.008,rtol=0)
        if kind=='uv_only':
            np.testing.assert_array_equal(observed[3:],0.)
            assert observed[0]>.1
        checks.append(dict(input=kind,expected=expected.tolist(),observed=observed.tolist()))
    result=dict(passed=True,channels=BANDS,checks=checks,calibration_sha256=c.sha256,
        source_sha256={str(p.relative_to(base.ROOT)):file_sha(p) for p in [Path(__file__).resolve(),Path(base.__file__).resolve(),base.ROOT/'apiaviz/research/dual_camera.py']})
    if environment:
        scene,_=base.build_scene(environment/'geometry',dict(position=[0,0,.01],heading=0),240,110,35,38,4,
            receptor_weights=weights,material_lookup=c.material,response_names=names)
        arrays={}; durations={}; differences=[]
        for mode in ('literal','opaque'):
            elapsed=[]
            for x in (.013,.027,.039):
                origin=[x,0.,.01]
                scene.sensors()[0].origin=mi.Point3f(origin) if mode=='literal' else base.dr.opaque(mi.Point3f,origin)
                ray,_=scene.sensors()[0].sample_ray(0.,.5,mi.Point2f(.5,.5),mi.Point2f(0.,0.))
                np.testing.assert_allclose(np.array(ray.o).reshape(3),origin,atol=1e-6,rtol=0)
                seed=(20261001+int(position_key([x,0.])[:8],16))%(2**32)
                start=time.perf_counter(); image=np.array(mi.render(scene,seed=seed,spp=4)); elapsed.append(time.perf_counter()-start)
                if mode=='literal': arrays[x]=image
                else:
                    np.testing.assert_allclose(image,arrays[x],atol=1e-6,rtol=1e-6)
                    differences.append(float(np.max(np.abs(image-arrays[x]))))
            durations[mode]=elapsed
        result['camera_benchmark']=dict(seconds=durations,max_abs_differences=differences,
            poses=3,spp=4,shape=[110,240,6],geometry_sha256=file_sha(environment/'geometry/geometry.json'),
            interpretation='Cold first call and two warm calls; tiny fixture benchmark, not a full-scene throughput forecast')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',type=Path); p.add_argument('--output',type=Path)
    args=p.parse_args(); result=validate(args.environment)
    if args.output:
        with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
