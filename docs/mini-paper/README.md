# Three-page ApiaViz research note

**Historical pilot:** the [expanded 440-trial study](../mechanism-study/report.md)
did not reproduce this four-route accuracy advantage. No baseline comparison
was significant after correction in that study. Read the
[updated three-page report](../mechanism-study/mechanism-report.pdf) for the
current assessment; the original pilot remains unchanged for provenance.

The main document is `apiaviz-mini-paper.pdf`. The editable manuscript is
`apiaviz-mini-paper.tex`, with the result-dependent text in `findings.tex`.
`results.csv` contains every paired trial used in the paper.

Across four routes, completion was 3/4 for ApiaViz, 3/4 for grayscale,
2/4 for raw colour, 2/4 for simple opponency, 4/4 for Sobel plus colour,
and 2/4 for Ardin-style input. ApiaViz had the lowest mean route deviation
(11.5 cm; Sobel 18.5 cm), including failed trials. This pilot does not establish
an overall navigation advantage over simple filtering.

All 24 trajectories passed the saved-artifact audit in `audit.json`.
The figure builder also verified identical acquisition images and settings,
and exact reproduction of the four earlier ApiaViz trajectories.

All baselines are tested in the same navigation loop with the same learning
images, frozen connections, spike parameters, memory rule and controller.
Only input processing changes. There is no per-baseline recalibration.
The Ardin condition uses the published preprocessing sequence (inversion,
local histogram equalisation, downsampling) through Python implementations.
It does not claim to reproduce the full original neural network or MATLAB
pixel values exactly. Sobel uses a 5x5 Gaussian (sigma 1 pixel), then horizontal
and vertical derivatives and their magnitude, with simple opponent colour.

The shared input interface contains three form planes and two auxiliary planes.
Grayscale and Ardin-style intensities are repeated into all five planes, keeping
both 4,000-KC pools available. Raw colour uses [L,G,B] and [G,B]; simple
opponency uses [L,L,L] and rectified G-B/B-G; Sobel uses [dx,dy,magnitude]
and the same opponent pair. All conditions then share 8x64 pooling,
within-view standardisation and the exact same fixed projection matrices.

Run the comparison from the repository root:

```sh
.pixi/envs/default/bin/python -m apiaviz.research.paper_baselines --ants 5 8 11 14
```

The recorded paper experiment split the four ants into separate processes;
`runs.json` identifies the exact four runs included in the manuscript.
Its source archives, checkpoints, training-image hashes, trajectories and
scan scores are saved under `apiaviz/output/paper-baselines/`.

Build figures from the completed recorded runs:

```sh
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/mini-paper/build_figures.py
.pixi/envs/default/bin/python -m apiaviz.research.audit_openloop apiaviz/output/paper-baselines/2026* --output docs/mini-paper/audit.json
```

From this directory, build the manuscript with:

```sh
mkdir -p build
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build apiaviz-mini-paper.tex
pdflatex -halt-on-error -interaction=nonstopmode -output-directory=build apiaviz-mini-paper.tex
cp build/apiaviz-mini-paper.pdf apiaviz-mini-paper.pdf
```

The figure builder checks that all conditions received identical teaching
images, poses and headings, verifies common encoder settings, and confirms
that the ApiaViz trajectories reproduce the earlier frozen runs exactly.
The experiment is descriptive: four routes, one world and one wiring seed.
