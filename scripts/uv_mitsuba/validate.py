"""Numerical checks for UV sampling, band order and polarized-sky transport."""
import json
from pathlib import Path

from render import ROOT, register_camera, receptor_spectra, spectrum, WAVELENGTHS, mi, dr, np


def main():
    out=ROOT/'apiaviz/output/uv-mitsuba'
    mi.set_variant('llvm_ad_spectral')
    dr.set_thread_count(2)
    register_camera()
    film=dict(type='specfilm',width=16,height=8,rfilter=dict(type='box'))
    for name,curve in zip(('band1_uv','band2_blue','band3_green'),receptor_spectra()):
        film[name]=spectrum(WAVELENGTHS,curve)
    outputs=[]
    for blocked in (False,True):
        values=np.ones(len(WAVELENGTHS))
        if blocked: values[WAVELENGTHS<400]=0
        scene=mi.load_dict(dict(type='scene',integrator=dict(type='path'),
            sensor=dict(type='uv_panorama',film=film,sampler=dict(type='independent',sample_count=2048)),
            emitter=dict(type='constant',radiance=spectrum(WAVELENGTHS,values))))
        result=np.array(mi.render(scene,seed=42,spp=2048))
        outputs.append(result.mean((0,1)))
    unblocked,blocked=outputs
    np.testing.assert_allclose(unblocked,1.,atol=.015,rtol=0)
    assert blocked[0]/unblocked[0]<.02
    assert blocked[1]/unblocked[1]>.85
    assert blocked[2]/unblocked[2]>.99
    result=dict(constant_radiance_response=unblocked.tolist(),
        response_after_blocking_below_400nm=blocked.tolist(),
        checks=['Equal normalized bands reproduce a unit spectral radiance',
                'Blocking UV suppresses the UV channel while retaining blue and green'])
    # A native polarized scene must not invent skylight polarization.
    mi.set_variant('llvm_ad_spectral_polarized')
    from polarization import register_polarization
    register_camera()
    register_polarization()
    native=mi.load_dict(dict(type='sunsky',sun_direction=[1,0,1]))
    props=mi.Properties('rayleigh_sunsky')
    props['nested']=native
    props['sun_direction']=mi.ScalarVector3f(1,0,1)
    wrapper=mi.load_dict(dict(type='rayleigh_sunsky',nested=native,sun_direction=[1,0,1]))
    directions=np.array([[1,0,1],[-1,0,1],[0,1,1],[1,1,.2]],dtype=float)
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    n=mi.Vector3f(directions.T)
    si=dr.zeros(mi.SurfaceInteraction3f,len(directions))
    si.wi=-n
    si.wavelengths=mi.UnpolarizedSpectrum(344.,436.,544.,620.)
    spec=native.eval(si)
    assert np.max(np.abs(np.array(spec[1,0])))==0
    polarized=wrapper.eval(si)
    np.testing.assert_allclose(np.array(polarized[0,0]),np.array(spec[0,0]),atol=0,rtol=0)
    s=np.stack(np.broadcast_arrays(*[np.array(polarized[i,0]) for i in range(4)]))
    degree=np.sqrt((s[1:]**2).sum(0))/np.maximum(s[0],1e-15)
    np.testing.assert_allclose(degree[:,0],0,atol=1e-6)
    np.testing.assert_allclose(degree[:,1],.75,atol=1e-6)
    assert np.max(degree)<=.75001
    result.update(polarization_at_sun=degree[:,0].tolist(),
                  polarization_at_90deg=degree[:,1].tolist())
    result['checks']+=['Native spectral sunsky has zero Q/U',
        'Rayleigh extension preserves spectral intensity exactly',
        'Polarization vanishes toward the sun and peaks at 90 degrees',
        'Stokes vectors obey their physical magnitude bound']
    # Independently check the reference-frame convention in the rendered files.
    rendered=[]
    for idx in range(2):
        path=out/'polarization'/f'stokes-{idx}.npz'
        if not path.exists(): continue
        saved=np.load(path)['stokes']
        meta=json.loads(path.with_suffix('.json').read_text())
        h,w=saved.shape[1:3]
        az=np.deg2rad(meta['pose']['heading'])+(.5-(np.arange(w)+.5)/w)*2*np.pi
        el=np.deg2rad(90-110*(np.arange(h)+.5)/h)
        aa,ee=np.meshgrid(az,el)
        view=np.stack([np.cos(ee)*np.cos(aa),np.cos(ee)*np.sin(aa),np.sin(ee)],-1)
        saz,sel=np.deg2rad([meta['sun_azimuth_deg'],meta['sun_elevation_deg']])
        sun=np.array([np.cos(saz)*np.cos(sel),np.sin(saz)*np.cos(sel),np.sin(sel)])
        dot=view@sun
        degree=.75*(1-dot**2)/(1+dot**2)
        electric=np.cross(sun,view)
        electric/=np.linalg.norm(electric,axis=-1,keepdims=True)
        horizontal=np.stack([-np.sin(aa),np.cos(aa),np.zeros_like(aa)],-1)
        vertical=np.cross(-view,horizontal)
        x=(electric*horizontal).sum(-1);y=(electric*vertical).sum(-1)
        mask=(ee>np.deg2rad(25))&(ee<np.deg2rad(85))&(dot<.99)
        errors=[float(np.quantile(np.abs(saved[i,...,0]/saved[0,...,0]-expected)[mask],.99))
                for i,expected in [(1,degree*(x*x-y*y)),(2,degree*2*x*y)]]
        assert max(errors)<.01,errors
        rendered.append(dict(sun=idx,q_u_99percent_error=errors))
    result['rendered_sky_basis_checks']=rendered
    if rendered:
        result['checks'].append('Rendered UV Q/I and U/I agree with independent Rayleigh basis calculation within 0.01 at the 99th percentile away from horizon/sun/zenith')
    result['passed']=True
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
