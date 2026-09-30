"""Explicit Rayleigh boundary approximation with Mitsuba polarized transport.

The standard sunsky supplies spectral intensity. An analytical, partially
polarized single-scattering field supplies Q/U in the implicit propagation
basis. This is not an atmospheric multiple-scattering simulation. Diffuse
surfaces depolarize. Four separate spectral-film passes retain insect bands;
Mitsuba's stock Stokes AOVs use human RGB and are unsuitable for this UV film.
"""
import argparse
import json
from pathlib import Path
import time

from render import ROOT, build_scene, register_camera, mi, dr, np


def register_polarization():
    class RayleighSky(mi.Emitter):
        def __init__(self, props):
            super().__init__(props)
            self.nested = props['nested']
            self.sun = dr.normalize(mi.Vector3f(props['sun_direction']))
            self.maximum = float(props.get('polarization_max', .75))
            if not 0 <= self.maximum <= 1:
                raise ValueError('Polarization maximum must be in [0,1]')
            self.m_flags = mi.EmitterFlags.Infinite | mi.EmitterFlags.SpatiallyVarying
        def polarize(self, spec, view):
            # view points from observer toward sky; photons propagate in -view.
            cosine = dr.clip(dr.dot(view, self.sun), -1., 1.)
            p = self.maximum*(1-cosine*cosine)/(1+cosine*cosine)
            p = dr.select((view.z > 0)&(cosine < np.cos(np.deg2rad(.5358/2))), p, 0.)
            electric = dr.cross(self.sun, view)
            electric *= dr.rsqrt(dr.maximum(dr.squared_norm(electric), 1e-20))
            axis = mi.mueller.stokes_basis(-view)
            perpendicular = dr.cross(-view, axis)
            a, b = dr.dot(electric, axis), dr.dot(electric, perpendicular)
            out = dr.zeros(mi.Spectrum)
            out[0,0] = spec[0,0]
            out[1,0] = spec[0,0]*p*(a*a-b*b)
            out[2,0] = spec[0,0]*p*(2*a*b)
            return out
        def eval(self, si, active=True):
            return self.polarize(self.nested.eval(si, active), -si.wi)
        def sample_direction(self, it, sample, active=True):
            ds, weight = self.nested.sample_direction(it, sample, active)
            # The scene must attribute this direction sample to the wrapper.
            ds.emitter = mi.EmitterPtr(self)
            return ds, self.polarize(weight, ds.d)
        def pdf_direction(self, it, ds, active=True):
            return self.nested.pdf_direction(it, ds, active)
        def eval_direction(self, it, ds, active=True):
            return self.polarize(self.nested.eval_direction(it, ds, active), ds.d)
        def set_scene(self, scene):
            self.nested.set_scene(scene)
        def bbox(self):
            return mi.ScalarBoundingBox3f()
        def to_string(self):
            return f'RayleighSky[analytical boundary, maximum DoLP={self.maximum}]'
    mi.register_emitter('rayleigh_sunsky', lambda p: RayleighSky(p))

    class ReceptorStokes(mi.SamplingIntegrator):
        def __init__(self, props):
            super().__init__(props)
            self.component = int(props.get('component',0))
            self.nested = mi.load_dict(dict(type='path',max_depth=5,rr_depth=4))
        def sample(self, scene, sampler, ray, medium=None, active=True):
            result, valid, aovs = self.nested.sample(scene,sampler,ray,medium,active)
            # Geographic horizontal tangent (increasing azimuth), with a
            # right-handed second axis about incoming photon direction.
            horizontal = dr.normalize(mi.Vector3f(-ray.d.y,ray.d.x,0.))
            rotation = mi.mueller.rotate_stokes_basis(-ray.d,
                mi.mueller.stokes_basis(-ray.d), horizontal)
            result = rotation@result
            return mi.mueller.depolarizer(result[self.component,0]), valid, []
        def to_string(self):
            return f'ReceptorStokes[component={self.component}]'
    mi.register_integrator('receptor_stokes',lambda p:ReceptorStokes(p))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=196)
    parser.add_argument('--spp',type=int,default=128)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    mi.set_variant('llvm_ad_spectral_polarized')
    dr.set_thread_count(2)
    register_camera()
    register_polarization()
    out=ROOT/'apiaviz/output/uv-mitsuba/polarization'
    out.mkdir(parents=True,exist_ok=True)
    geometry=ROOT/'apiaviz/output/uv-mitsuba/geometry'
    pose=json.loads((geometry/'geometry.json').read_text())['poses'][1]
    examples=[(35.,38.),(145.,20.)]
    if args.smoke: examples=examples[:1]
    for idx,(az,el) in enumerate(examples):
        started=time.monotonic()
        scene,_=build_scene(geometry,pose,args.width,args.height,az,el,args.spp,polarized=True)
        components=[]
        for c in range(4):
            print('START',idx,'Stokes',c,flush=True)
            integrator=mi.load_dict(dict(type='receptor_stokes',component=c))
            data=np.array(mi.render(scene,integrator=integrator,seed=20260930,spp=args.spp))
            assert np.isfinite(data).all()
            components.append(data)
        stokes=np.stack(components)
        intensity=stokes[0]
        assert np.min(intensity)>=0
        degree=np.sqrt((stokes[1:]**2).sum(0))/np.maximum(intensity,1e-15)
        assert degree.max() < 1.0001, degree.max()
        assert np.abs(stokes[3]).max() < 1e-9
        np.savez_compressed(out/f'stokes-{idx}.npz',stokes=stokes)
        (out/f'stokes-{idx}.json').write_text(json.dumps(dict(sun_azimuth_deg=az,sun_elevation_deg=el,
            pose=pose,shape=stokes.shape,spp=args.spp,seed=20260930,elapsed_s=time.monotonic()-started,
            max_degree_polarization=float(degree.max()),variant=mi.variant(),
            spectral_intensity='Mitsuba sunsky, 320-700 nm, three approximate receptor bands',
            polarization='Rayleigh angular boundary approximation, Pmax=0.75, wavelength-independent fraction; direct solar disk unpolarized; no atmospheric multiple scattering',
            basis='Q positive for increasing-azimuth horizontal tangent; U uses cross(-view,horizontal)',
            surfaces='Two-sided Lambertian depolarizers; no leaf transmission or surface gloss model'),indent=2)+'\n')
        print('DONE',idx,time.monotonic()-started,'max DoLP',degree.max(),flush=True)


if __name__=='__main__':main()
