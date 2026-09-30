import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from apiaviz.research.spectral_input import Calibration, CHANNELS, SCHEMA, file_sha, load_frame, validate_arrays


class SpectralInputTests(unittest.TestCase):
    def test_archived_calibration_has_normalized_measured_bands(self):
        c=Calibration('docs/uv-calibration/calibration.json')
        np.testing.assert_allclose(np.trapz(c.weights,c.wavelengths,axis=1),1.)
        self.assertEqual(c.material('earth').shape,(77,))
        with self.assertRaises(ValueError): c.material('invented UV surface')

    def test_linear_highlights_and_signed_polarization_survive_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            intensity=np.full((2,4,3),12.,dtype=np.float32)
            stokes=np.stack([intensity,-intensity*.4,intensity*.1,intensity*0])
            arrays={}
            for name,data in [('intensity',intensity),('stokes',stokes)]:
                path=d/f'{name}.npy'
                np.save(path,data)
                arrays[name]=dict(file=path.name,sha256=file_sha(path))
            record=dict(schema=SCHEMA,channels=CHANNELS,linear=True,shape=list(intensity.shape),arrays=arrays)
            path=d/'frame.json'
            path.write_text(json.dumps(record))
            a,s,_=load_frame(path)
            np.testing.assert_array_equal(a,intensity)
            np.testing.assert_array_equal(s,stokes)
            np.save(d/'intensity.npy',intensity*.5)
            with self.assertRaisesRegex(ValueError,'checksum'): load_frame(path)

    def test_rejects_display_rgb_and_unphysical_stokes(self):
        with self.assertRaises(ValueError): validate_arrays(np.zeros((2,4,3),dtype=np.uint8))
        intensity=np.ones((2,4,3))
        with self.assertRaises(ValueError): validate_arrays(intensity,np.stack([intensity,intensity*2,intensity*0,intensity*0]))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'display.json'
            p.write_text(json.dumps(dict(schema=SCHEMA,channels=['red','green','blue'],linear=False)))
            with self.assertRaises(ValueError): load_frame(p)


if __name__ == '__main__': unittest.main()
