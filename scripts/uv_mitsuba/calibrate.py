"""Build a traceable library-proxy calibration, without fitting navigation results."""
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'apiaviz/output/uv-calibration'
SOURCES = OUT / 'sources'
GRID = np.arange(320., 701., 5.)
SAMPLES = {
    'green_grass': 'Lawn_Grass_GDS91_green_BECKa',
    'dry_grass': 'Cheatgrass_ANP92-11A_BECKa',
    'dead_plant': 'Tumbleweed_ANP92-2C_Dry_BECKa',
    'soil': 'Stonewall_Playa_CU93-52A_a11_BECKa',
    'carbonate': 'Calcite_CO2004_BECKb',
}
# These assignments are ecological approximations, not measurements of this site.
MAPPING = {
    'Grass 0': 'green_grass', 'Grass 2': 'green_grass',
    'Grass 1': 'dry_grass', 'Grass 3': 'dead_plant',
    'Grassland earth': 'soil', 'Limestone': 'carbonate',
    **{f'Litter {i}': 'dead_plant' for i in range(4)},
    'Scrub stems': 'dry_grass', 'Weathered twigs': 'dead_plant',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_interpolation(x, y, target=GRID, max_gap=8.):
    """Reject extrapolation and large missing bands; do not silently clip data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert x.shape == y.shape and np.all(np.diff(x) > 0)
    good = np.isfinite(y) & (y >= 0) & (y <= 1)
    gx, gy = x[good], y[good]
    assert gx[0] <= target[0] and gx[-1] >= target[-1], 'Missing spectral coverage'
    edges = (gx[:-1] < target[-1]) & (gx[1:] > target[0])
    gap = float(np.diff(gx)[edges].max())
    assert gap <= max_gap, f'Unacceptable spectral gap: {gap} nm'
    missing = x[(~good) & (x >= target[0]) & (x <= target[-1])]
    return np.interp(target, gx, gy), dict(max_interpolation_gap_nm=gap,
        invalid_points_in_band_nm=missing.tolist(), extrapolated=False)


def read_legacy_ascii(path):
    # USGS v5 uses three fixed-width columns, with stars for deleted values.
    rows = []
    for line in path.read_text().splitlines()[16:]:
        if not line.strip():
            continue
        rows.append([float(s) if '*' not in s else np.nan
                     for s in (line[:15], line[15:30], line[30:45])])
    return np.array(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checks = {}
    for line in (SOURCES/'checksums.txt').read_text().splitlines():
        digest, name = line.split()
        if name in ('usgs_splib07.parquet', 'wavelengths.parquet'):
            assert sha(SOURCES/name) == digest, name
            checks[name] = digest
    assert len(checks) == 2
    provenance = {}
    for record in sorted(SOURCES.glob('*.provenance.json')):
        p = json.loads(record.read_text())
        source = SOURCES/record.name.removesuffix('.provenance.json')
        assert sha(source) == p['sha256'], source
        provenance[source.name] = p
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        receptors = rdata.read_rda(SOURCES/'pavo-sysdata.rda')['vissyst']
    wl = receptors.wl.to_numpy(float)
    raw = receptors[['apis.s', 'apis.m', 'apis.l']].to_numpy(float).T
    assert np.isfinite(raw).all() and raw.min() >= 0
    response = np.array([np.interp(GRID, wl, r)*GRID for r in raw])
    response /= np.trapz(response, GRID, axis=1)[:, None]
    receptor_record = dict(species='Apis mellifera', columns=['apis.s','apis.m','apis.l'],
        empirical_peaks_nm=wl[np.argmax(raw, axis=1)].tolist(),
        below_320_fraction_of_300_700_photon_weighted_sensitivity=[
            float(np.trapz(r[wl<=320]*wl[wl<=320], wl[wl<=320])/np.trapz(r*wl, wl)) for r in raw],
        source='Peitsch et al. (1992), as distributed by pavo; see pinned NEWS.md',
        note='Below-320 fractions describe sensitivity under flat spectral energy, not missing natural-sky photon catches.',
        units='Band-normalized energy weighting: lambda*S(lambda)/integral(lambda*S(lambda)) on 320–700 nm; absolute gains unknown')
    pd.DataFrame(dict(wavelength_nm=wl, UV=raw[0], blue=raw[1], green=raw[2])).to_csv(OUT/'receptors-source.csv', index=False)
    pd.DataFrame(dict(wavelength_nm=GRID, UV=response[0], blue=response[1], green=response[2])).to_csv(OUT/'receptors-render.csv', index=False)
    data = pd.read_parquet(SOURCES/'usgs_splib07.parquet')
    waves = pd.read_parquet(SOURCES/'wavelengths.parquet').set_index('grid_id')
    materials, records, comparisons = {}, {}, {}
    for key, token in SAMPLES.items():
        rows = data[data['source.original_id'] == f'splib07a_{token}_AREF']
        assert len(rows) == 1, token
        row = rows.iloc[0]
        grid = waves.loc[row['spectral_data.wavelength_grid_id']]
        assert row['spectral_data.wavelength_unit'] == grid.wavelength_unit == 'um'
        assert row['spectral_data.reflectance_scale'] == 'unit'
        x, y = np.asarray(grid.wavelengths)*1000., np.asarray(row['spectral_data.values'])
        materials[key], coverage = checked_interpolation(x, y)
        pd.DataFrame(dict(wavelength_nm=x, reflectance=y)).to_csv(OUT/f'{key}-source.csv', index=False)
        records[key] = dict(sample_id=row['source.original_id'], instrument=row['measurement.instrument'],
            measurement_type=row['spectral_data.type'], coverage=coverage,
            provenance='USGS v7 through OpenSpecLib v0.0.6; original instrument samples, linear interpolation to 5 nm',
            caveat='Library proxy, not a measurement of the rendered species or terrain')
        legacy = {'dry_grass':'cheatgrass', 'dead_plant':'tumbleweed', 'carbonate':'calcite'}.get(key)
        if legacy:
            old = read_legacy_ascii(SOURCES/f'{legacy}-v5.asc')
            assert len(old) == len(x)
            np.testing.assert_allclose(old[:,0]*1000, x, atol=.001, rtol=0)
            mask = (x >= 320) & (x <= 700) & np.isfinite(y) & (y>=0) & np.isfinite(old[:,1])
            error = float(np.max(np.abs(y[mask]-old[mask,1])))
            assert error < 2e-6, (key, error)
            comparisons[key] = dict(primary_file=f'{legacy}-v5.asc', n=int(mask.sum()), max_absolute_difference=error)
    pd.DataFrame(dict(wavelength_nm=GRID, **materials)).to_csv(OUT/'materials-render.csv', index=False)
    manifest = dict(schema=1, purpose='Empirical library-proxy calibration; no navigation outcome fitting',
        wavelength_nm=GRID.tolist(), receptor_weights=response.tolist(), receptors=receptor_record,
        material_reflectance={k:v.tolist() for k,v in materials.items()}, samples=records,
        scene_material_mapping=MAPPING, sources=provenance, release_checksums=checks,
        primary_ascii_crosschecks=comparisons,
        excluded=['Dry_Long_Grass AV87-2: removed by USGS for poor SNR',
            'AMX27–32: mathematical mixtures containing that dry-grass spectrum',
            'ASD spectra starting at 350 nm: no UV extrapolation',
            'Artificial shifted/offset LawnGrass spectra: not natural material measurements'],
        unresolved=['Site/species matching; reflectance angular dependence and leaf transmission',
            'Absolute receptor gains, ocular transmission and adaptation',
            'Sky spectral radiance and UV polarization: not empirically calibrated',
            'Spectrum below 320 nm excluded by sunsky support'])
    (OUT/'calibration.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(receptors=receptor_record, crosschecks=comparisons, coverage={k:v['coverage'] for k,v in records.items()}),indent=2))


if __name__ == '__main__':
    main()
