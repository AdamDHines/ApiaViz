# Sources and provenance

This record accompanies the 30 September 2026 calibration. Exact download URLs, timestamps, byte counts and SHA-256 hashes are in [sources.lock.json](sources.lock.json). The source bytes and their individual `.provenance.json` files are retained in `apiaviz/output/uv-calibration/sources/`.

## Honeybee receptors

Peitsch, D. et al. (1992), *The spectral input systems of hymenopteran insects and their receptor-based colour vision*, Journal of Comparative Physiology A. [DOI: 10.1007/BF00190398](https://doi.org/10.1007/BF00190398).

The numerical route into this project is [pavo](https://github.com/rmaia/pavo), revision `f25bdb2f5d0ed7fe3386e9946b72b6b15b4f2bb2`, `R/sysdata.rda`, `vissyst` columns `apis.s`, `apis.m`, `apis.l`. Its [pinned release notes](https://github.com/rmaia/pavo/blob/f25bdb2f5d0ed7fe3386e9946b72b6b15b4f2bb2/NEWS.md) identify the empirical replacement under version 2.1.0. We also archive the data documentation, preparation script and GPL version 2 licence supplied by the repository. Retain that attribution and licence with the derived receptor tables. We do not claim to have re-digitized the original paper or recovered independent animals' raw records.

Extraction uses rdata 1.1.0. The 300–700 nm tabulation is saved unchanged as `receptors-source.csv`; `receptors-render.csv` applies the wavelength weighting, support restriction and unit-area normalization described in the main note. No peak fitting, Gaussian approximation or navigation-based adjustment is performed.

## Material reflectances

Kokaly, R. F. et al. (2017), *USGS Spectral Library Version 7*, USGS Data Series 1035. [Report and DOI](https://pubs.usgs.gov/publication/ds1035); [data release](https://doi.org/10.5066/F7RR1WDJ). The archived report describes instruments, quality control and removed spectra. USGS data are public domain; the redistributed records identify the same status.

The direct ScienceBase metadata request returned HTTP 403 during this session. Data were therefore obtained from the [OpenSpecLib v0.0.6 release](https://github.com/null-jones/openspeclib/releases/tag/v0.0.6), which repackages the USGS files. Both `usgs_splib07.parquet` and `wavelengths.parquet` match that release's `checksums.txt`. The mirror is a transport source, not an independent scientific measurement.

Selected exact records:

```
splib07a_Lawn_Grass_GDS91_green_BECKa_AREF
splib07a_Cheatgrass_ANP92-11A_BECKa_AREF
splib07a_Tumbleweed_ANP92-2C_Dry_BECKa_AREF
splib07a_Stonewall_Playa_CU93-52A_a11_BECKa_AREF
splib07a_Calcite_CO2004_BECKb_AREF
```

They are Beckman reflectance spectra. Instrument coverage alone is insufficient: the script checks each sample's actual valid wavelength support, invalid points and interpolation gaps. Wavelengths are converted from micrometres to nanometres; reflectances remain unit fractions. Source values, including invalid bands, are preserved in the full-spectrum CSV files. Only valid values enter interpolation.

Primary descriptions were checked for [green lawn grass](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/V/lawn_grass_gds91.html), [dry cheatgrass](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/V/cheatgrass_anp92-11a_veg.html), [dried tumbleweed](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/V/tumbleweed.anp92-2c.veg.html) and [calcite](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/M/calcite_co2004.html). These are legacy descriptions of the identified samples. Three corresponding primary ASCII spectra were also downloaded and numerically cross-checked; results are in `calibration.json`. Stonewall uses the v7 record metadata; no separate original sample description was recovered here, which is a provenance limitation.

Rejected candidates and reasons:

- **Dry_Long_Grass AV87-2 brown:** the v7 release notes report removal for poor signal-to-noise. We do not restore it from older libraries.
- **Grass dry/green AMX27–32:** the [primary description](https://pubs.usgs.gov/of/2003/ofr-03-395/DESCRIPT/V/grass_dry%2Bgreen_series.html) identifies mathematical mixtures containing the same rejected dry-grass sample. They are not independent measured mixtures.
- **ASD golden dry grass, limestone and pine wood records:** useful visible spectra, but their approximately 350 nm lower limit requires unsupported values across the bee UV peak. They were not extended into UV.
- **Shifted/offset LawnGrass records:** artificial spectral transformations, excluded from natural-material assignments.
- **Blackbrush:** inspected and archived as a possible vegetation alternative. Its description includes green leaves and flowers; not used in the final grass mapping.

The measured samples do not constitute an uncertainty distribution for our environment. Differences between plant species, moisture conditions, whole rock surfaces and mineral samples may exceed instrument noise. We therefore report source quality and proxy choices rather than manufacture confidence intervals for scene realism.

## Sky and polarization

The intensity renders use Mitsuba's native spectral sunsky emitter. [Mitsuba documentation](https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_emitters.html) and [polarization documentation](https://mitsuba.readthedocs.io/en/latest/src/key_topics/polarization.html) describe the renderer's capabilities. The previous analytical polarization extension is described in [the prototype record](../uv-mitsuba/README.md). Neither the 0.75 maximum polarization nor the atmospheric turbidity was fitted to measurements in this calibration.

Poughon, L., Aubry, V., Monnoyer, J., Viollet, S. and Serres, J. (2024), [*A 2 month-long annotated skylight polarization images database*](https://doi.org/10.57745/9L2YUB), with [associated processing code](https://github.com/mol-1/A-2-month-long-annotated-skylight-polarization-images-database---associated-code). Its metadata and README are archived. The README identifies an IMX250MYR colour-polarization sensor, Bayer R/G/B channels, geometric calibration, weather labels and timing correction. This is valuable for future visible-channel checks, but it has no calibrated UV measurement channel. We did not download the multi-gigabyte daily image arrays or fit polarization parameters to them. The README states CC-BY-SA; preserve its original attribution if redistributing the documentation or data.

The [Prague sky model](https://cgg.mff.cuni.cz/publications/prague-sky-model/) was investigated as a more detailed atmospheric/polarization model. Its coefficient data derive from simulation. Substituting it would not by itself turn the sky into an empirically calibrated input, so no such substitution was made in this pass.
