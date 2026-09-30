"""Render controlled receptor/material substitutions into a separate output tree."""
import argparse
import json
import time
import importlib.metadata

import render as base
from calibrate import OUT, sha

np, mi, dr = base.np, base.mi, base.dr


def masks(scene, pose, width, height, azimuth, elevation):
    az = np.deg2rad(pose['heading'])+(.5-(np.arange(width)+.5)/width)*2*np.pi
    el = np.deg2rad(90.-110.*(np.arange(height)+.5)/height)
    aa, ee = np.meshgrid(az, el)
    directions = np.stack([np.cos(ee)*np.cos(aa), np.cos(ee)*np.sin(aa), np.sin(ee)], -1)
    ray = mi.Ray3f(mi.Point3f(pose['position']), mi.Vector3f(directions.reshape(-1,3).T))
    hit = np.array(scene.ray_intersect(ray).is_valid()).reshape(height,width)
    saz, sel = np.deg2rad([azimuth,elevation])
    sun = np.array([np.cos(sel)*np.cos(saz),np.cos(sel)*np.sin(saz),np.sin(sel)])
    # Exclude solar disc vicinity, both poles and a two-degree horizon strip.
    sky = (~hit) & (ee>np.deg2rad(2)) & (ee<np.deg2rad(85)) & ((directions@sun)<np.cos(np.deg2rad(2)))
    return dict(sky=sky, terrain=hit, solid_angle_weight=np.cos(ee))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=960)
    parser.add_argument('--height',type=int,default=294)
    parser.add_argument('--spp',type=int,default=128)
    args = parser.parse_args()
    c = json.loads((OUT/'calibration.json').read_text())
    mi.set_variant('llvm_ad_spectral')
    dr.set_thread_count(2)
    base.register_camera()
    original_receptors, original_materials = base.receptor_spectra, base.material_spectrum
    weights = np.array(c['receptor_weights'])
    def measured_material(name):
        if name == 'earth': name = 'Grassland earth'
        return np.array(c['material_reflectance'][c['scene_material_mapping'][name]])
    geometry = base.ROOT/'apiaviz/output/uv-mitsuba/geometry'
    meta = json.loads((geometry/'geometry.json').read_text())
    jobs = []
    for receptors in (False,True):
        for materials in (False,True):
            jobs.append((f'middle-r{int(receptors)}-m{int(materials)}',1,35.,38.,receptors,materials,20260930))
    jobs += [('start-calibrated',0,35.,38.,True,True,20260930),
             ('end-calibrated',2,35.,38.,True,True,20260930),
             ('sun-shift-calibrated',1,145.,20.,True,True,20260930),
             ('middle-repeat',1,35.,38.,True,True,20260931)]
    dest = OUT/'renders'
    dest.mkdir(exist_ok=True)
    try:
        for name,idx,az,el,receptors,materials,seed in jobs:
            start = time.monotonic()
            base.receptor_spectra = (lambda:weights.copy()) if receptors else original_receptors
            base.material_spectrum = measured_material if materials else original_materials
            print('START',name,flush=True)
            scene,_ = base.build_scene(geometry,meta['poses'][idx],args.width,args.height,az,el,args.spp)
            data = np.array(mi.render(scene,seed=seed,spp=args.spp))
            assert data.shape == (args.height,args.width,3)
            assert np.isfinite(data).all() and data.min() >= 0
            np.save(dest/f'{name}.npy',data)
            scene.sensors()[0].film().bitmap().write(str(dest/f'{name}.exr'))
            visibility = masks(scene,meta['poses'][idx],args.width,args.height,az,el)
            np.savez_compressed(dest/f'{name}-masks.npz',**visibility)
            means = {}
            for region in ('sky','terrain'):
                mask = visibility[region]
                means[region] = np.average(data[mask],axis=0,weights=visibility['solid_angle_weight'][mask]).tolist()
            record = dict(name=name, empirical_receptors=receptors, measured_material_proxies=materials,
                pose=meta['poses'][idx],sun_azimuth_deg=az,sun_elevation_deg=el,
                shape=data.shape,spp=args.spp,seed=seed,threads=2,
                calibration_sha256=sha(OUT/'calibration.json'),scene_sha256=meta['scene_sha256'],
                code_sha256={p.name:sha(p) for p in [base.ROOT/'scripts/uv_mitsuba'/s for s in ['render.py','calibrate.py','render_calibrated.py']]},
                versions={p:importlib.metadata.version(p) for p in ['mitsuba','drjit','numpy','pandas','rdata','pyarrow']},
                mean_band_radiance=means,sky_terrain_ratio=(np.array(means['sky'])/np.array(means['terrain'])).tolist(),
                metric_note='Solid-angle-weighted regional means; geometric masks; sky excludes horizon, zenith and solar vicinity. Descriptive render diagnostics, not navigation performance.',
                illumination='Native sunsky, turbidity 2.5; its ground albedo changes with the soil proxy. Unpolarized.',
                elapsed_s=time.monotonic()-start)
            (dest/f'{name}.json').write_text(json.dumps(record,indent=2)+'\n')
            print('DONE',name,round(record['elapsed_s'],2),flush=True)
    finally:
        base.receptor_spectra, base.material_spectrum = original_receptors, original_materials


if __name__ == '__main__':
    main()
