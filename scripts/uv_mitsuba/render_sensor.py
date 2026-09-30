"""Render calibrated linear bee bands and optional modelled Stokes panoramas.

Writes fresh, provenance-labelled frames for a future spectral model adapter.
Existing RGB navigation and historical calibration controls remain separate.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import render as base

sys.path.insert(0, str(base.ROOT))
from apiaviz.research.spectral_input import Calibration, CHANNELS, SCHEMA, file_sha, validate_arrays


def render(args):
    geometry = args.geometry.resolve()
    calibration = Calibration(args.calibration)
    metadata = json.loads((geometry/'geometry.json').read_text())
    poses = json.loads(args.poses.read_text()) if args.poses else metadata['poses']
    if not poses or args.width <= 0 or args.height <= 0 or args.spp <= 0 or args.threads <= 0:
        raise ValueError('Require poses and positive render dimensions, samples and threads')
    if not -90 <= args.elevation_min < args.elevation_max <= 90:
        raise ValueError('Invalid elevation range')
    if not 0 <= args.polarization_max <= 1 or not -90 <= args.sun_elevation <= 90:
        raise ValueError('Invalid sun elevation or polarization maximum')
    for pose in poses:
        if len(pose['position']) != 3 or not base.np.isfinite([*pose['position'],pose['heading']]).all():
            raise ValueError('Each pose needs finite world-space XYZ and heading in degrees')
    hashes = {'geometry.json':file_sha(geometry/'geometry.json')}
    for asset in metadata['meshes']:
        calibration.material(asset['material'])  # Fail before any rendering on unknown material.
        path = (geometry/asset['file']).resolve()
        if path.parent != geometry:
            raise ValueError('Mesh asset must be directly inside the geometry directory')
        hashes[asset['file']] = file_sha(path)
        if 'sha256' in asset and asset['sha256'] != hashes[asset['file']]:
            raise ValueError('Exported mesh checksum mismatch')
    variant = 'llvm_ad_spectral_polarized' if args.polarized else 'llvm_ad_spectral'
    base.mi.set_variant(variant)
    base.dr.set_thread_count(args.threads)
    base.register_camera()
    if args.polarized:
        from polarization import register_polarization
        register_polarization()
    scripts = [Path(__file__), Path(base.__file__), Path(__file__).with_name('polarization.py'),
               base.ROOT/'apiaviz/research/spectral_input.py']
    config = dict(schema=SCHEMA, channels=CHANNELS, linear=True,
        units='relative band-weighted radiance; unit-area photon-weighted responses; not absolute photon catch',
        wavelength_nm=calibration.wavelengths.tolist(), receptor_weights=calibration.weights.tolist(),
        calibration_sha256=calibration.sha256, geometry_sha256=hashes,
        scene_sha256=metadata['scene_sha256'],
        source_sha256={str(p.relative_to(base.ROOT)):file_sha(p) for p in scripts},
        versions=dict(mitsuba=base.mi.__version__,drjit=base.dr.__version__,numpy=base.np.__version__),
        environment_lock_sha256=file_sha(Path(__file__).with_name('pixi.lock')),
        variant=variant, spp=args.spp, seed=args.seed, threads=args.threads,
        shape=[args.height,args.width,3], elevation_deg=[args.elevation_max,args.elevation_min],
        azimuth='360 degrees; centre forward; image left positive from heading; pixel centres',
        sun=dict(azimuth_deg=args.sun_azimuth,elevation_deg=args.sun_elevation,turbidity=2.5),
        polarization=dict(enabled=args.polarized,maximum=args.polarization_max if args.polarized else None,
            calibration='provisional analytical Rayleigh boundary; wavelength-independent fraction; not empirical UV polarimetry',
            basis='Q positive along increasing-azimuth horizontal tangent; U uses cross(-view,horizontal)'),
        surfaces='measured USGS material proxies; two-sided Lambertian depolarizers',
        missing_band='300–320 nm omitted; archived calibration documents sensitivity loss',
        poses=poses)
    # No reuse of old output/cache directories, including failed partial runs.
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    (args.output/'calibration.json').write_bytes(calibration.path.read_bytes())
    for i,pose in enumerate(poses):
        started = time.monotonic()
        scene,_ = base.build_scene(geometry,pose,args.width,args.height,args.sun_azimuth,args.sun_elevation,
            args.spp,polarized=args.polarized,receptor_weights=calibration.weights,
            material_lookup=calibration.material,elevation=(args.elevation_min,args.elevation_max),
            polarization_max=args.polarization_max)
        stokes=None
        if args.polarized:
            stokes=base.np.stack([base.np.array(base.mi.render(scene,
                integrator=base.mi.load_dict(dict(type='receptor_stokes',component=c)),
                seed=args.seed,spp=args.spp)) for c in range(4)])
            intensity=stokes[0]
        else:
            intensity=base.np.array(base.mi.render(scene,seed=args.seed,spp=args.spp))
        validate_arrays(intensity,stokes)
        if intensity.shape != (args.height,args.width,3):
            raise ValueError('Unexpected spectral film shape')
        arrays={}
        for name,data in [('intensity',intensity),('stokes',stokes)]:
            if data is None: continue
            path=args.output/f'{i:05d}-{name}.npy'
            base.np.save(path,data,allow_pickle=False)
            arrays[name]=dict(file=path.name,sha256=file_sha(path))
        record=dict(config,pose=pose,arrays=arrays,elapsed_s=time.monotonic()-started)
        (args.output/f'{i:05d}.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(frame=i,elapsed_s=record['elapsed_s'],output=str(args.output))),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(frames=len(poses),protocol_sha256=file_sha(args.output/'protocol.json')))+'\n')


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=base.ROOT/'docs/uv-calibration/calibration.json')
    p.add_argument('--poses',type=Path,help='JSON list of {position:[x,y,z], heading:degrees}; default exported poses')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=294)
    p.add_argument('--spp',type=int,default=128)
    p.add_argument('--seed',type=int,default=20260930)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--sun-azimuth',type=float,default=35.)
    p.add_argument('--sun-elevation',type=float,default=38.)
    p.add_argument('--elevation-min',type=float,default=-20.)
    p.add_argument('--elevation-max',type=float,default=90.)
    p.add_argument('--polarized',action='store_true')
    p.add_argument('--polarization-max',type=float,default=.75)
    return p


if __name__ == '__main__': render(parser().parse_args())
