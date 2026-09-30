"""Small numerical check of the calibrated renderer without research artifacts."""
import json
import sys

import render as base

sys.path.insert(0,str(base.ROOT))
from apiaviz.research.spectral_input import Calibration


def main():
    c=Calibration(base.ROOT/'docs/uv-calibration/calibration.json')
    mi,np=base.mi,base.np
    mi.set_variant('llvm_ad_spectral')
    base.dr.set_thread_count(2)
    base.register_camera()
    film=dict(type='specfilm',width=16,height=8,component_format='float32',rfilter=dict(type='box'))
    for name,weights in zip(('band1_uv','band2_blue','band3_green'),c.weights):
        film[name]=base.spectrum(c.wavelengths,weights)
    checks=[]
    for blocked in (False,True):
        values=np.ones(len(c.wavelengths))
        if blocked: values[c.wavelengths<400]=0
        scene=mi.load_dict(dict(type='scene',integrator=dict(type='path'),
            sensor=dict(type='uv_panorama',film=film,sampler=dict(type='independent',sample_count=2048)),
            emitter=dict(type='constant',radiance=base.spectrum(c.wavelengths,values))))
        observed=np.array(mi.render(scene,seed=42,spp=2048)).mean((0,1))
        fine=np.linspace(320,700,38001)
        expected=np.array([np.trapz(np.interp(fine,c.wavelengths,w)*np.interp(fine,c.wavelengths,values),fine) for w in c.weights])
        np.testing.assert_allclose(observed,expected,atol=.008,rtol=0)
        checks.append(dict(input='UV blocked' if blocked else 'flat',expected=expected.tolist(),observed=observed.tolist()))
    print(json.dumps(dict(passed=True,checks=checks,calibration_sha256=c.sha256),indent=2))


if __name__ == '__main__': main()
