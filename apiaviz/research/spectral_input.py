"""Linear UV/blue/green render contract, separate from RGB model preprocessing.

No display gamma, clipping, per-channel scaling or RGB conversion occurs here.
Polarization is returned separately; it is not implicitly a model input.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'apiaviz-spectral-frame-v1'
CHANNELS = ['uv', 'blue', 'green']


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Calibration:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = file_sha(path)
        self.record = json.loads(self.path.read_text())
        self.wavelengths = np.asarray(self.record['wavelength_nm'], dtype=float)
        self.weights = np.asarray(self.record['receptor_weights'], dtype=float)
        if not np.array_equal(self.wavelengths, np.arange(320.,701.,5.)):
            raise ValueError('Require calibrated 320–700 nm, 5 nm grid; no extrapolation')
        if self.weights.shape != (3,len(self.wavelengths)) or not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError('Invalid receptor weights')
        if not np.allclose(np.trapz(self.weights,self.wavelengths,axis=1),1.,atol=1e-8,rtol=0):
            raise ValueError('Receptor weights must have unit wavelength integral')
        for name in self.record['scene_material_mapping']:
            self.material(name)

    def material(self, name):
        if name == 'earth': name = 'Grassland earth'
        try:
            proxy = self.record['scene_material_mapping'][name]
            values = np.asarray(self.record['material_reflectance'][proxy], dtype=float)
        except KeyError as exc:
            raise ValueError(f'No calibrated material proxy for {name!r}') from exc
        if values.shape != self.wavelengths.shape or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
            raise ValueError(f'Invalid reflectance for {name!r}')
        return values


def validate_arrays(intensity, stokes=None):
    a = np.asarray(intensity)
    if a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 1 or a.dtype.kind != 'f' or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Require finite nonnegative floating H×W×3 linear responses')
    if stokes is not None:
        s = np.asarray(stokes)
        if s.shape != (4,*a.shape) or s.dtype.kind != 'f' or not np.isfinite(s).all():
            raise ValueError('Require finite floating I/Q/U/V×H×W×3 Stokes array')
        if not np.array_equal(s[0],a):
            raise ValueError('Stokes I differs from intensity')
        if np.any(np.linalg.norm(s[1:],axis=0) > a*(1+1e-4)+1e-7):
            raise ValueError('Stokes polarization magnitude exceeds intensity')


def load_frame(sidecar):
    """Verified scientific data only. PNGs and unlabelled three-channel arrays fail."""
    sidecar = Path(sidecar)
    record = json.loads(sidecar.read_text())
    if record.get('schema') != SCHEMA or record.get('channels') != CHANNELS or record.get('linear') is not True:
        raise ValueError('Not a labelled linear UV/blue/green frame')
    def read_asset(name):
        asset = record['arrays'][name]
        filename = Path(asset['file'])
        if filename.name != str(filename) or filename.suffix != '.npy':
            raise ValueError('Expected adjacent raw .npy array')
        path = sidecar.parent/filename
        if file_sha(path) != asset['sha256']:
            raise ValueError('Spectral array checksum mismatch')
        return np.load(path, allow_pickle=False)
    intensity = read_asset('intensity')
    stokes = read_asset('stokes') if 'stokes' in record['arrays'] else None
    validate_arrays(intensity, stokes)
    if list(intensity.shape) != record['shape']:
        raise ValueError('Spectral array shape differs from metadata')
    return intensity, stokes, record
