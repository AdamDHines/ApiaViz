# Separate and combined visual frontend changes

This experiment compares original ApiaViz, oriented form filters, linear colour
opponency, and their combination. Sobel plus colour and Ardin-style image
preprocessing are matched controls. The six conditions each use eight routes
and five saved wiring seeds: 240 fresh navigation trials.

The original model remains the default. Both changes are opt-in in
[`RefinementEncoder`](../../apiaviz/research/frontend_refinements.py):

```python
from apiaviz.research.frontend_refinements import RefinementEncoder

original = RefinementEncoder("apiaviz", seed=19)
oriented = RefinementEncoder("oriented_form", seed=19)
linear = RefinementEncoder("linear_colour", seed=19)
combined = RefinementEncoder("oriented_linear", seed=19)
```

The encoder accepts a batch of RGB or GB images in `[0, 1]`. The benchmark uses
18 × 74 RGB renders, keeps green and blue, constructs three form and two colour
maps, pools each map to 8 × 64, and drives two populations of 4,000 spiking cells.
The two populations compete independently. Navigation stores binary spike
patterns using the same `SpikeOverlapMemory` in every condition. A wiring seed
selects fixed weights; no visual filters are fitted.

All conditions use identical teaching/probe images, saved random weights,
pooling, normalisation, spiking parameters, learning procedure, memory readout,
steering and stopping rules. The runner checks exact reproduction of original,
Sobel and Ardin-style trajectories. The analysis also requires exact reproduction
of the previously evaluated individual variants. These reruns check reproducibility;
they are not additional independent samples.

- [Results and interpretation](report.md)
- [Protocol fixed before the combination was evaluated](protocol.json)
- [Trial data](results.csv), [summary table](results-table.md), and [paired statistics](statistics.json)
- [Independent trajectory audit](audit.json)
- [Run locations](runs.json), whose directories include source archives and manifests

From the repository root:

```sh
.pixi/envs/default/bin/python -m unittest discover -s tests -p test_frontend_refinements.py
.pixi/envs/default/bin/python -m apiaviz.research.frontend_refinements --ants 4 6 7 9 10 12 13 15 --protocol docs/frontend-factorial/protocol.json --output apiaviz/output/frontend-factorial
```

The recorded experiment was split into four processes with disjoint pairs of
routes. A fresh run creates a new timestamped directory. After completion, set
`runs.json` to the directories of that complete execution, then run:

```sh
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/frontend-factorial/analyse.py
.pixi/envs/default/bin/python docs/frontend-deep-dive/audit.py --study-dir docs/frontend-factorial
```

The runner uses the saved teaching images and checkpoints indexed by
`docs/mechanism-study/runs.json`; the previous-variant verification also uses
`docs/frontend-deep-dive/runs.json`. These local output directories are excluded
from Git. To regenerate the entire chain on a new checkout, first reproduce the
[mechanism study](../mechanism-study/README.md), then the
[individual-variant study](../frontend-deep-dive/README.md), updating their run
indexes before this experiment. The underlying ant/world data and project
environment must also be available.

Statistics use eight route clusters with five crossed wiring seeds, rather
than treating 40 runs as independent environments. All 12 prespecified
contrasts share one Holm correction; intervals are unadjusted crossed bootstrap
intervals. With eight route clusters, the smallest possible two-sided exact
sign-flip p-value is 2/256; a 12-comparison Holm family therefore cannot pass
0.05 even at that minimum. This matrix estimates effects and checks
reproducibility; it cannot establish corrected significance. More independent
route/environment replication is needed for that purpose. The routes and individual-variant results were already inspected,
so this remains development evidence. Ardin-style means adapted image
preprocessing under the shared learning system, not a reproduction of the
complete published Ardin network.
