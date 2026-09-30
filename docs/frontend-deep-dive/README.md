# ApiaViz versus Sobel: implementation deep dive

The subsequent [240-trial factorial evaluation](../frontend-factorial/report.md)
tests oriented form and linear colour both separately and together, with exact
reproduction checks for the results reported here.

This follow-up examines the input operations behind the earlier matched
navigation comparison. It includes a filter/colour audit and a fixed matrix of
280 navigation trials: two exact reference reruns plus five deterministic
frontend candidates, each on eight previously inspected routes and five wiring
seeds. These are development experiments, not a held-out validation set.

- [Interpretation and recommendations](report.md)
- [Research context](literature.md)
- [Fixed development protocol](protocol.json)
- [All trial results](results.csv) and [summary table](results-table.md)
- [Paired effects and uncertainty](statistics.json)
- [Input audit](input-audit.json) and [trajectory audit](audit.json)
- [Experimental encoder](../../apiaviz/research/frontend_refinements.py)

The original frontend defaults are unchanged. Corrections in the source
comments describe the existing colour nonlinearity accurately; they do not
alter model behaviour. All candidates use the existing saved wiring weights,
teaching images, finite spiking parameters, memory and controller. Reference
reruns must reproduce every original steering score and trajectory exactly.

Run from the repository root:

```sh
.pixi/envs/default/bin/python -m unittest discover -s tests -p test_frontend_refinements.py
.pixi/envs/default/bin/python -m apiaviz.research.frontend_refinements --ants 4 6 7 9 10 12 13 15
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/frontend-deep-dive/input_audit.py
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python docs/frontend-deep-dive/analyse.py
.pixi/envs/default/bin/python docs/frontend-deep-dive/audit.py
```

`runs.json` names the four recorded processes, each covering two ants. To analyse
a fresh execution, update that index explicitly after the complete matrix has
finished. The runner reads stimulus banks and checkpoints from the previous
[mechanism study](../mechanism-study/README.md); the saved source archives and
manifests identify exactly what was run. It does not download external models.

Negative candidate-minus-reference effects favour the candidate. Exploratory
sign-flip tests use eight route-level averages over five paired seeds, with
Holm correction across five candidates for each reference separately. Crossed
bootstrap intervals resample route and seed axes; these are unadjusted 95%
intervals and do not measure uncertainty across independent environments.
The choice of candidate family was informed by previously observed benchmark
results, so neither correction nor additional seeds makes this a confirmatory
test of general superiority.
