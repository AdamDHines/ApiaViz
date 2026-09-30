"""Check spectral coverage, source integrity and calibrated film responses."""
import json
import numpy as np
import render as base
from calibrate import OUT, SOURCES, GRID, sha, checked_interpolation


def main():
    c = json.loads((OUT/'calibration.json').read_text())
    for name,record in c['sources'].items():
        assert sha(SOURCES/name) == record['sha256']
    curves = np.array(c['receptor_weights'])
    np.testing.assert_allclose(np.trapz(curves,GRID,axis=1),1.,atol=1e-12)
    assert curves.min()>=0 and np.isfinite(curves).all()
    for reflectance in c['material_reflectance'].values():
        a = np.array(reflectance)
        assert np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    # Deliberately bad inputs must fail, rather than become invented UV data.
    for x,y in [(np.array([350.,700.]),np.array([.1,.2])),
                (np.array([320.,340.,700.]),np.array([.1,np.nan,.2]))]:
        try:
            checked_interpolation(x,y)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid UV coverage was accepted')
    mi, dr = base.mi, base.dr
    mi.set_variant('llvm_ad_spectral')
    dr.set_thread_count(2)
    base.register_camera()
    film=dict(type='specfilm',width=32,height=16,rfilter=dict(type='box'))
    for name,curve in zip(('band1_uv','band2_blue','band3_green'),curves):
        film[name]=base.spectrum(GRID,curve)
    checks=[]
    for kind in ('flat','uv_blocked'):
        illuminant=np.ones(len(GRID))
        if kind=='uv_blocked': illuminant[GRID<400]=0
        scene=mi.load_dict(dict(type='scene',integrator=dict(type='path'),
            sensor=dict(type='uv_panorama',film=film,sampler=dict(type='independent',sample_count=2048)),
            emitter=dict(type='constant',radiance=base.spectrum(GRID,illuminant))))
        output=np.array(mi.render(scene,seed=42,spp=2048)).mean((0,1))
        fine=np.linspace(320,700,38001)
        expected=np.array([np.trapz(np.interp(fine,GRID,r)*np.interp(fine,GRID,illuminant),fine) for r in curves])
        np.testing.assert_allclose(output,expected,atol=.008,rtol=0)
        checks.append(dict(input=kind,expected=expected.tolist(),rendered=output.tolist()))
    # Independent seed is a Monte Carlo stability check, not an empirical CI.
    first=json.loads((OUT/'renders/middle-r1-m1.json').read_text())
    repeat=json.loads((OUT/'renders/middle-repeat.json').read_text())
    relative={k:(np.array(repeat['mean_band_radiance'][k])/first['mean_band_radiance'][k]-1).tolist() for k in ('sky','terrain')}
    assert max(abs(v) for values in relative.values() for v in values)<.02
    original=np.load(base.ROOT/'apiaviz/output/uv-mitsuba/renders/02-route-middle.npy')
    control=np.load(OUT/'renders/middle-r0-m0.npy')
    np.testing.assert_allclose(control,original,atol=1e-6,rtol=1e-5)
    result=dict(passed=True,source_hashes_checked=len(c['sources']),coverage_rejections_passed=True,
        unit_and_uv_block_tests=checks,independent_seed_relative_mean_changes=relative,
        original_render_max_abs_difference=float(np.max(abs(control-original))),
        primary_ascii_crosschecks=c['primary_ascii_crosschecks'],
        interpretation='Numerical and provenance validation only; not an empirical validation of scene realism or UV sky polarization.')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
