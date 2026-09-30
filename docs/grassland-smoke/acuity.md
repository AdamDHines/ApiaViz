# Is the current input too coarse for honeybee vision?

The current 74 × 18 panorama samples a 296° × 75° field. Because the renderer
includes both endpoints, adjacent ray centres are **4.055° horizontally and
4.412° vertically**. This is a legacy navigation setting, not a calibrated
honeybee retina. The smoke test demonstrates navigation at that setting; it
does not establish its biological adequacy.

Rigosi, Warrant and O'Carroll (2021) measured interommatidial spacing below 2°
over a broad frontal region of worker honeybee eyes, with a minimum of 1.3°.
This spacing is the angular separation between neighbouring visual axes.
[Primary study](https://www.nature.com/articles/s41598-021-00407-2).

Rigosi, Wiederman and O'Carroll (2017) separately measured photoreceptor angular
acceptance widths: mean horizontal/vertical values around 2.2°/2.3°, an estimated
frontal value of 1.9°, and individual cells down to 1.6°. These describe optical
blur, not sampling intervals. Their detection of 0.6° targets in single-cell
recordings should not be interpreted as an eye resolving a 0.6° pixel grid.
[Primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5382694/).

Colour behaviour also has different spatial limits. In a particular coloured-disc
detection task, Giurfa et al. (1996) found thresholds of 5° when green-receptor
contrast was available and 15° for chromatic contrast without green contrast.
These task-dependent thresholds are not ommatidial spacing estimates and do not
justify sampling every visual pathway at 4°.
[Primary study](https://link.springer.com/article/10.1007/BF00227381).

## Controlled follow-up

The [39-run resolution comparison](../grassland-resolution/README.md) has now
tested these grids with both unchanged pixel filters and angular-offset controls.
The discussion below records the design rationale. A calibrated retinal-optics
model remains separate from that experiment.

| Width × height | Horizontal spacing | Vertical spacing | Purpose |
|---|---:|---:|---|
| 74 × 18 | 4.055° | 4.412° | Existing reference |
| 149 × 39 | 2.000° | 1.974° | Practical intermediate sampling |
| 199 × 51 | 1.495° | 1.500° | Finer, frontal-acuity-inspired sampling |

Uniform rectangular sampling is an approximation: bee resolution varies across
the eye, and an actual ommatidial lattice is not rectangular. A calibrated eye
model should represent both sampling positions and angular acceptance functions.

Use the same saved world, route, wiring seeds, teaching positions and memory /
spike / controller budgets at each input resolution. Rebuild route memories from
the new inputs, identically for each method. Run preprocessing before the existing
common pooling, so finer input does not simply give one method more mushroom-body
cells or a larger stored representation. Keep the published Ardin-style 10 × 36
reduction explicit; an alternative resolution-scaled downsampler is a separate
control, not silently the same baseline.

Spatial filter widths need particular care. Current adaptation, centre-surround
and Sobel kernels are specified in pixels. A resolution increase with unchanged
kernels also changes their angular receptive fields. Distinguish an input-only
sensitivity run (unchanged code, acknowledged angular change) from an experiment
that holds receptive-field widths fixed in degrees. Neither should be described
as an isolated test of the other factor. Optical blur and any colour-specific
pooling need their own explicit definitions rather than just sharper images.

The saved 720 × 152 panoramas already sample at 0.5° in both axes and retain much
more detail than the 74 × 18 input. They can supply these finer grids at cached
positions. Newly visited positions during free navigation will still need rendering.
The original 74 × 18 near-tie did not determine which method would benefit from
finer sampling; the completed comparison reports the observed changes.
