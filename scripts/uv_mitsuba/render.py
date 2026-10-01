"""Spectral UV/blue/green panoramas from exported navigation geometry.

Physical spectral transport and solar illumination, illustrative material
spectra and Gaussian receptor bands. The optional Rayleigh sky polarization
is an explicit analytical approximation, not Mitsuba's native sky model.
"""
import argparse
import json
import os
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
# Prefer the active environment; preserve an explicitly configured LLVM path.
if 'DRJIT_LIBLLVM_PATH' not in os.environ:
    for prefix in (Path(sys.prefix), ROOT/'.pixi/envs/default'):
        candidates = sorted((prefix/'lib').glob('libLLVM*.dylib')) + sorted((prefix/'lib').glob('libLLVM*.so*'))
        if candidates:
            os.environ['DRJIT_LIBLLVM_PATH'] = str(candidates[0])
            break
import drjit as dr
import mitsuba as mi
import numpy as np

WAVELENGTHS = np.arange(320., 701., 5.)
PEAKS = [344., 436., 544.]
WIDTHS = [20., 30., 40.]


def spectrum(wavelengths, values):
    return dict(type='regular', wavelength_min=float(wavelengths[0]),
                wavelength_max=float(wavelengths[-1]), values=','.join(map(str,values)))


def receptor_spectra():
    # Quantum sensitivity: photon count is proportional to lambda times energy.
    # Normalize area for each illustrative band; absolute receptor gains unknown.
    curves = []
    for peak, width in zip(PEAKS, WIDTHS):
        curve = np.exp(-.5*((WAVELENGTHS-peak)/width)**2)
        curve *= WAVELENGTHS/peak
        curve /= np.trapz(curve, WAVELENGTHS)
        curves.append(curve)
    return np.array(curves)


def material_spectrum(name):
    knots = np.array([320,360,400,440,480,540,580,620,660,700.])
    if name.startswith('Grass '):
        i = int(name.split()[-1])
        if i in (0,2):
            values = np.array([.025,.025,.035,.045,.065,.18,.13,.07,.045,.22])
            values *= 1. if i == 0 else .72
        else:
            values = np.array([.045,.055,.095,.15,.22,.32,.38,.40,.40,.44])
            values *= 1. if i == 1 else 1.15
    elif 'Limestone' in name:
        values = np.array([.16,.20,.29,.38,.44,.50,.53,.55,.57,.58])
    elif 'earth' in name:
        values = np.array([.045,.055,.085,.12,.17,.23,.27,.30,.32,.34])
    elif 'Litter' in name:
        values = np.array([.025,.035,.055,.075,.11,.16,.22,.26,.28,.31])
        values *= .8+.12*int(name.split()[-1])
    elif 'stems' in name:
        values = np.array([.03,.035,.05,.06,.08,.15,.14,.11,.09,.20])
    else:
        values = np.array([.025,.03,.05,.07,.10,.14,.18,.21,.24,.27])
    return np.interp(WAVELENGTHS, knots, values)


def register_camera():
    class Panorama(mi.Sensor):
        def __init__(self, props):
            super().__init__(props)
            self.origin = mi.Point3f(props.get('origin', [0.,0.,0.]))
            self.yaw = float(props.get('yaw', 0.))*np.pi/180
            self.low = float(props.get('elevation_min', -20.))*np.pi/180
            self.high = float(props.get('elevation_max', 90.))*np.pi/180
        def sample_ray(self, time, sample1, sample2, sample3, active=True):
            si = dr.zeros(mi.SurfaceInteraction3f)
            si.uv = sample2
            wavelengths, weight = self.sample_wavelengths(si, sample1, active)
            az = self.yaw+(0.5-sample2.x)*2*np.pi
            el = self.high+(self.low-self.high)*sample2.y
            direction = mi.Vector3f(dr.cos(el)*dr.cos(az), dr.cos(el)*dr.sin(az), dr.sin(el))
            return mi.Ray3f(self.origin, direction, time, wavelengths), weight
        def sample_ray_differential(self, time, sample1, sample2, sample3, active=True):
            ray, weight = self.sample_ray(time, sample1, sample2, sample3, active)
            return mi.RayDifferential3f(ray), weight
        def bbox(self):
            return mi.ScalarBoundingBox3f()
        def to_string(self):
            return 'SpectralPanorama[]'
    mi.register_sensor('uv_panorama', lambda p: Panorama(p))


def build_scene(geometry, pose, width, height, sun_az, sun_el, spp, polarized=False,
                receptor_weights=None, material_lookup=None, elevation=(-20.,90.),
                polarization_max=.75, response_names=None, surface_detail=False):
    curves = receptor_spectra() if receptor_weights is None else receptor_weights
    material_lookup = material_spectrum if material_lookup is None else material_lookup
    film = dict(type='specfilm', width=width, height=height, component_format='float32',
                rfilter=dict(type='box'))
    names=response_names or ('band1_uv','band2_blue','band3_green')
    if len(names)!=len(curves): raise ValueError('Response labels and curves differ')
    for name, curve in zip(names, curves):
        film[name] = spectrum(WAVELENGTHS, curve)
    az, el = np.deg2rad([sun_az, sun_el])
    sun = [np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)]
    scene = dict(type='scene', integrator=dict(type='path', max_depth=5, rr_depth=4),
        sensor=dict(type='uv_panorama', origin=pose['position'], yaw=pose['heading'], film=film,
                    elevation_min=elevation[0], elevation_max=elevation[1],
                    sampler=dict(type='independent', sample_count=spp)),
        sky=dict(type='sunsky', sun_direction=sun, turbidity=2.5,
                 albedo=spectrum(WAVELENGTHS, material_lookup('earth'))))
    if polarized:
        scene['sky'] = dict(type='rayleigh_sunsky', nested=scene['sky'], sun_direction=sun,
                            polarization_max=polarization_max)
        scene['integrator'] = dict(type='receptor_stokes', component=0)
    meta = json.loads((geometry/'geometry.json').read_text())
    if surface_detail:
        import surface_materials
        surface_materials.register()
        if meta.get('surface_schema')!='source-informed-surfaces-v1':raise ValueError('Missing surface provenance')
    materials = {}
    for i, item in enumerate(meta['meshes']):
        data = np.load(geometry/item['file'])
        values = material_lookup(item['material'])
        materials[item['material']] = values.tolist()
        props = mi.Properties()
        props['bsdf'] = (surface_materials.bsdf(values,meta['surface_materials'][item['material']]) if surface_detail
            else mi.load_dict(dict(type='twosided', nested=dict(type='diffuse', reflectance=spectrum(WAVELENGTHS,values)))))
        mesh = mi.Mesh(item['material'], len(data['vertices']), len(data['faces']), props,has_vertex_normals=surface_detail)
        params = mi.traverse(mesh)
        params['vertex_positions'] = data['vertices'].ravel()
        params['faces'] = data['faces'].ravel()
        if surface_detail:params['vertex_normals']=data['normals'].ravel()
        params.update()
        scene[f'mesh_{i:02}'] = mesh
    return mi.load_dict(scene), materials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path, default=ROOT/'apiaviz/output/uv-mitsuba/renders')
    parser.add_argument('--width',type=int, default=960)
    parser.add_argument('--height',type=int, default=294)
    parser.add_argument('--spp',type=int, default=128)
    parser.add_argument('--only',type=int)
    args = parser.parse_args()
    mi.set_variant('llvm_ad_spectral')
    dr.set_thread_count(2)
    register_camera()
    geometry = ROOT/'apiaviz/output/uv-mitsuba/geometry'
    meta = json.loads((geometry/'geometry.json').read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    examples = [dict(name='01-route-start', pose=meta['poses'][0], sun_az=35., sun_el=38.),
                dict(name='02-route-middle', pose=meta['poses'][1], sun_az=35., sun_el=38.),
                dict(name='03-route-end', pose=meta['poses'][2], sun_az=35., sun_el=38.),
                dict(name='04-middle-sun-shift', pose=meta['poses'][1], sun_az=145., sun_el=20.)]
    for i, ex in enumerate(examples):
        if args.only is not None and i != args.only:
            continue
        print('START', ex['name'],flush=True)
        started = time.monotonic()
        scene, materials = build_scene(geometry,ex['pose'],args.width,args.height,ex['sun_az'],ex['sun_el'],args.spp)
        output = mi.render(scene,seed=20260930,spp=args.spp)
        data = np.array(output)
        assert data.shape == (args.height,args.width,3), data.shape
        assert np.isfinite(data).all() and np.min(data) >= 0
        np.save(args.output/(ex['name']+'.npy'),data)
        scene.sensors()[0].film().bitmap().write(str(args.output/(ex['name']+'.exr')))
        record = dict(**ex, wavelength_nm=WAVELENGTHS.tolist(), receptor_peaks_nm=PEAKS,
            receptor_sigma_nm=WIDTHS, receptor_weights=receptor_spectra().tolist(),
            materials=materials, mitsuba=mi.__version__, variant=mi.variant(), seed=20260930,
            spp=args.spp, shape=data.shape, scene_sha256=meta['scene_sha256'],
            azimuth='360 degrees, image left positive from heading; centre is forward',
            elevation_deg=[90,-20], material_model='Two-sided Lambertian; illustrative spectra, no measured species calibration',
            elapsed_s=time.monotonic()-started, polarization='Native sunsky is unpolarized')
        (args.output/(ex['name']+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        print('DONE',ex['name'],record['elapsed_s'],data.min(),data.max(),flush=True)


if __name__=='__main__':
    main()
