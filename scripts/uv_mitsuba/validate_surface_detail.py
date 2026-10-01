"""Bounded renderer checks: spectral shape, derivatives, normals and paired views."""
import argparse
import json
from pathlib import Path
import sys
import time
import render as base
sys.path.insert(0,str(base.ROOT))
from apiaviz.research.spectral_input import Calibration,file_sha
from apiaviz.research.dual_camera import visible_weights,BANDS
import surface_materials as surface


def run(geometry,out):
    out.mkdir(parents=True,exist_ok=False)
    mi,dr,np=base.mi,base.dr,base.np
    mi.set_variant('llvm_ad_spectral');dr.set_thread_count(2);base.register_camera();surface.register()
    calibration=Calibration(base.ROOT/'docs/uv-calibration/calibration.json')
    meta=json.loads((geometry/'geometry.json').read_text())
    points=mi.Point3f([[.123,-.371,.479],[.287,.591,-.615],[.711,-.231,.899]])
    v,g=surface.noise(points);assert np.all((np.array(v)>=0)&(np.array(v)<=1))
    errors=[]
    for axis in range(3):
        shift=mi.Vector3f(0);shift[axis]=.001
        derivative=(surface.noise(points+shift)[0]-surface.noise(points-shift)[0])/.002
        errors.append(float(np.max(np.abs(np.array(derivative-g[axis])))))
    assert max(errors)<.001,errors
    stone=meta['surface_materials']['Limestone'];rho=calibration.material('Limestone')
    texture=mi.load_dict(dict(type='source_surface_texture',scale=stone['colour_scale'],octaves=5,
        dark=stone['dark_ratio'],nested=base.spectrum(base.WAVELENGTHS,rho)))
    si=dr.zeros(mi.SurfaceInteraction3f,3);si.p=points
    si.wavelengths=mi.Spectrum([340.,440.,540.,640.])
    values=np.array(texture.eval(si));reference=np.interp([340,440,540,640],base.WAVELENGTHS,rho)[:,None]
    gain=values/reference
    assert np.all(gain>=stone['dark_ratio']-1e-5) and np.all(gain<=1+1e-5)
    assert np.max(np.ptp(gain,axis=0))<1e-5,'Spatial detail changed spectral shape'
    # Verify that the leaf branch really transports light through the sheet,
    # and that solid rock does not gain accidental transmission.
    si.wi=mi.Vector3f(0,0,1);si.sh_frame=mi.Frame3f(mi.Vector3f(0,0,1))
    si.n=mi.Vector3f(0,0,1);si.dp_du=mi.Vector3f(1,0,0);si.dp_dv=mi.Vector3f(0,1,0)
    leaf=dict(meta['surface_materials']['Grass 0'],bump_distance_m=0.)
    rock=dict(stone,bump_distance_m=0.)
    leaf_bsdf=surface.bsdf(calibration.material('Grass 0'),leaf)
    rock_bsdf=surface.bsdf(rho,rock)
    transmitted=np.array(leaf_bsdf.eval(mi.BSDFContext(),si,mi.Vector3f(0,0,-1)))
    opaque=np.array(rock_bsdf.eval(mi.BSDFContext(),si,mi.Vector3f(0,0,-1)))
    assert np.all(transmitted>0) and np.all(opaque==0)
    weights=np.concatenate([calibration.weights,visible_weights(calibration.wavelengths)])
    records=[]
    for detailed in (False,True):
        t=time.perf_counter()
        scene,_=base.build_scene(geometry,meta['poses'][0],480,144,35.,38.,64,
            receptor_weights=weights,material_lookup=calibration.material,
            response_names=[f'band{i+1}_{name}' for i,name in enumerate(BANDS)],surface_detail=detailed)
        for i,pose in enumerate(meta['poses']):
            scene.sensors()[0].origin=dr.opaque(mi.Point3f,pose['position'])
            started=time.perf_counter();raw=np.array(mi.render(scene,seed=20261002,spp=64))
            assert raw.shape==(144,480,6) and np.isfinite(raw).all() and raw.min()>=0
            name=f'{"detailed" if detailed else "flat"}-{i}.npy';np.save(out/name,raw)
            records.append(dict(detailed=detailed,pose=pose,file=name,sha256=file_sha(out/name),
                                render_s=time.perf_counter()-started))
        print('rendered',detailed,'seconds',time.perf_counter()-t,flush=True)
    result=dict(geometry_sha256=file_sha(geometry/'geometry.json'),calibration_sha256=calibration.sha256,
        source_sha256={p.name:file_sha(p) for p in (Path(__file__),Path(base.__file__),Path(surface.__file__))},
        records=records,noise_gradient_max_errors=errors,spectral_shape_preserved=True,
        leaf_transmission_nonzero=True,opaque_transmission_zero=True,
        material_parameters=meta['surface_materials'],versions=dict(mitsuba=mi.__version__,drjit=dr.__version__))
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.geometry.resolve(),a.output.resolve())
