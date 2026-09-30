# Mechanistic navigation study

The subsequent [implementation deep dive](../frontend-deep-dive/report.md)
tests five deterministic frontend refinements in 280 additional development
trials. It retains this study's results and exact reference behaviour.

**Completed:** [three-page illustrated report](mechanism-report.pdf),
[full interpretation](report.md), [literature review](literature.md).
The pilot advantage did not replicate: ApiaViz and Sobel each completed 29/40
trials, with mean deviation 55.2 and 50.4 cm. No baseline comparison was
significant after correction. Component tests identify effects on visual
representation without establishing a general navigation advantage.

This study tests the four-route pilot's accuracy result on eight additional ant
routes and five new wiring seeds. Its fixed matrix contains 440 navigation
trials: six preprocessing baselines and five input-only interventions, with 40
trials per condition. There is no new parameter fitting or baseline calibration.

- `protocol.json` fixes cases, endpoints, comparisons and stopping rules.
- `literature.md` develops the hypotheses and links the primary research.
- `runs.json` identifies the exact source archives, checkpoints and trials.
- `renderer-validation.json` records pixel and navigation reproduction checks.
- `results.csv`, `statistics.json`, `results-table.md` and `figures/` contain
  the completed analysis of all 440 trials.
- `audit.json` records verification of saved stimuli, decisions and trajectories.
- `fixed-count-*.json` contains an additional offline diagnostic with exactly
  200 active cells per stream. This ranks continuous drives and also removes
  finite timing/binning; it is not another navigation model or primary test.

The eight route identities are the primary statistical units. Five wiring seeds
are paired within each route. Individual steps and probe images are not treated
as independent replicates. The primary tests condition on the chosen seeds;
crossed bootstrap intervals additionally describe route-by-seed uncertainty.
All comparisons remain conditional on one world geometry and colour assignment.

Run from the repository root:

```sh
.pixi/envs/default/bin/python -m apiaviz.research.validate_fast_render
.pixi/envs/default/bin/python -m apiaviz.research.mechanisms --ants 4 6 7 9 10 12 13 15
```

The recorded study splits the eight ants into four processes, each using one
Torch thread. Runs are listed explicitly in `runs.json`; failed initial launches
are listed separately and contribute no data. They failed before saving any
result row because of a JSON boolean conversion error. The identical matrix
was rerun after correcting serialization.

After completion:

```sh
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/mechanism-study/analyse.py
.pixi/envs/default/bin/python docs/mechanism-study/audit.py
.pixi/envs/default/bin/python docs/mechanism-study/fixed_count.py --ants 4 6 7 9
.pixi/envs/default/bin/python docs/mechanism-study/fixed_count.py --ants 10 12 13 15
.pixi/envs/default/bin/python docs/mechanism-study/feature_geometry.py
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/mechanism-study/analyse_controls.py
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/mechanism-study/diagnostics.py
.pixi/envs/default/bin/python docs/mechanism-study/validate_study_images.py
```

The faster renderer uses the original geometry, triangle ordering and float32
pixel calculations, with sparse candidate-pixel selection. It reproduced 120
sampled original images exactly, plus all four prior ApiaViz paths and every
saved steering score. New trajectories use the same 13-heading scan, movement
step and 200-step limit as the pilot.

The completed decision audit reconstructed 61,091 navigation decisions and
35,200 probe decisions without discrepancies. A further original-renderer
check at the largest-excursion query of every new trial gave 440 exact image
matches. These checks verify stored decisions and sampled renderings; they
are not a complete independent rerun of every neural computation.

The `no_dog` condition removes the entire form filter bank, including its
Gaussian low-pass channel. Interpret it as a filter-bank intervention, not as
an isolated removal of centre–surround inhibition. Pathway swaps preserve the
number of maps and both KC populations. Actual KC activity may change under
any intervention because the inhibition settings are held fixed.

Build the three-page note from this directory:

```sh
mkdir -p build
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build mechanism-report.tex
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build mechanism-report.tex
cp build/mechanism-report.pdf mechanism-report.pdf
```
