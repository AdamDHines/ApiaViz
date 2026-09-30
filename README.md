# ApiaViz

ApiaViz is an insect-inspired visual encoder with fixed colour/contrast processing and associative memory for navigation and flower reward choice. The visual encoder needs no training. Route and reward memories **do learn** through deterministic one-shot updates.

The repository contains the historical ANN evaluation and a controlled study runner with deterministic spiking projection neurons and Kenyon cells, adaptive firing, and finite graded feedback inhibition. The spiking implementation uses ordinary PyTorch; it does not require surrogate gradients or SNNTorch.

**Current handoff:** see [AGENTS.md](AGENTS.md) and the [workstation handoff](docs/HANDOFF.md). The expanded navigation run was cancelled at 145/378 trials pending collision-geometry fixes. Large scenes, caches and videos are local artifacts excluded from Git.

## Setup

Install [pixi](https://pixi.sh), then run `pixi install`. Existing local navigation data belongs in `apiaviz/mbant/data/antview/`; flower data belongs in `apiaviz/dataset/17flowers/<class>/*.jpg`. The study runner never downloads datasets or model weights implicitly.

```sh
pixi run test
pixi run study --help
pixi run study --list
```

## Controlled experiments

`--code-dim` is the **total** KC population, divided equally between form and colour streams. All encoders receive the same views, and representations can be crossed with the same memories and navigation protocols.

```sh
# Small end-to-end flower experiment (validation, not a publication benchmark).
pixi run study --task flowers --representations gray apiaviz --modes kwta adaptive --code-dim 256 --pool 4 16 --per-class 6 --folds 2 --fixations 1 2 --calibration-views 8 --bootstrap 50

# Navigation: calibration uses ants 1–3; evaluation uses other ants.
pixi run study --task nav --ants 4 --routes 1 --representations gray apiaviz --modes kwta adaptive --protocols offline reset free --viewpoints 1 9
```

`offline` tests heading selection at fixed positions; `reset` adds corrective replacements onto the route; `free` evaluates navigation without replacements. Both moving-agent protocols use visual feedback. The controlled reset protocol is explicit in the new runner; use the historical command below to reproduce the older reset implementation.

Each run writes a manifest, encoder configuration and fixed weights, development/evaluation splits, JSONL results, and navigation traces or per-image flower predictions. Navigation intervals resample ants; flower intervals resample original images with repeated observations kept together. Single-ant runs do not estimate between-ant uncertainty.

See [the study protocol](docs/spiking-study.md) for mechanisms, calibration, complete experiment commands, limitations, citations and deep baseline setup.

See [implementation validation](docs/validation.md) for tests and the local pilot results.

The current research priority is [spiking navigation without corrective resets](docs/openloop-spiking.md). The [expanded 440-trial study](docs/mechanism-study/report.md) compares six preprocessing baselines and five component interventions on eight routes and five wiring seeds, with identical learning, neural settings and control. ApiaViz and Sobel plus colour each completed 29/40 trials; mean deviation was 55.2 cm and 50.4 cm, respectively. The earlier pilot's accuracy advantage did not replicate, and no baseline comparison was significant after multiple-comparison correction. A [three-page illustrated report](docs/mechanism-study/mechanism-report.pdf), [data and reproduction instructions](docs/mechanism-study/README.md) explain the representation findings and limitations. The [earlier four-route note](docs/mini-paper/apiaviz-mini-paper.pdf) is retained as a pilot. The development runner includes directional acquisition, lateralized memories, sequence-context diagnostics and a LIF latency reference for the historical feature currents.

A [280-trial implementation follow-up](docs/frontend-deep-dive/report.md) tests five fixed frontend refinements and reproduces both references exactly. Oriented form filters and linear colour processing are promising development candidates; simple channel balancing worsened performance. Neither candidate has a corrected significant advantage or independent-world validation, and the original defaults remain unchanged. [Code, data and reproduction](docs/frontend-deep-dive/README.md) accompany the analysis.

A [240-trial factorial evaluation](docs/frontend-factorial/report.md) compares oriented form and linear colour separately and together against the unchanged original, Sobel-plus-colour and Ardin-style preprocessing. Linear colour gives the lowest mean deviation (29.9 cm), oriented form the most nest arrivals (32/40), and their combination gives 44.6 cm and 28/40 arrivals. All variants are opt-in; [reproduction commands](docs/frontend-factorial/README.md) and paired uncertainty accompany these development results.

[Navigation videos](docs/navigation-videos/index.html) show three full linear-colour runs with the ant's panoramic view, actual model input, visual features, firing cells, heading scans and route progress. The [video notes](docs/navigation-videos/README.md) include direct MP4 links, reproduction commands and the distinction between the display panorama and the 74 × 18 model input.

[Terrain concepts](docs/terrain-concepts/README.md) explore four procedural environments with natural material palettes, solid landmarks and directional sunlight. Ant-height PNGs, matching 296° panoramas and editable Blender scenes are provided for feedback; these worlds have not been used in navigation evaluations.

[Grassland smoke test](docs/grassland-smoke/README.md) evaluates a generated grassland at the original world's scale. All nine runs reached the nest: mean deviation was 3.13 cm for linear-colour ApiaViz, 3.16 cm for Sobel + colour and 4.73 cm for Ardin-style preprocessing. The scene and panorama cache are saved for reuse; all trajectories and scan scores reproduced exactly without rendering again.

[Input-resolution comparison](docs/grassland-resolution/README.md) tests 74 × 18, 149 × 39 and 199 × 51 inputs while retaining the same teaching poses, projection wiring, spiking circuit, memory and controller. All 39 runs reached the nest. ApiaViz at 199 × 51 with filter offsets fixed in degrees gave 2.13 cm mean deviation, versus 3.13 cm at the original resolution; increasing resolution with unchanged pixel filters did not give that improvement. This is a one-route development result, with three paired wiring seeds and explicit filter-scale controls.

The [training and movement audit](docs/navigation-regime/README.md) qualifies these arrival results: under the original collision-free evaluator, an agent that simply maintains its starting heading also reaches the endpoint, with 26.80 cm mean deviation. The smaller visual route deviations remain informative; arrival alone does not establish robust navigation on that route.

[Teaching, recovery and active-sensing experiments](docs/navigation-experiments/INTERPRETATION.md) follow that audit with 387 trials. A single taught traversal sufficed for the original route, but positional recovery was weak across all three preprocessing methods. ApiaViz did not show a consistent advantage across the three scene–route pairs. The tested active controller substantially reduced observations but also reduced arrival rates; it remains an experimental alternative, not a replacement for the reference controller.

[Familiarity-guided controller development](docs/familiarity-controller/README.md) adds matched-view comparisons before and after movement, sideways casting, and physical reorientation scans. It includes controlled-field tests, a separate development runner and recorded-view replays. The original navigation controllers remain intact.

## Using the spiking encoder

```python
import torch
from apiaviz.research.encoders import VisualEncoder, EncoderConfig
from apiaviz.research.circuit import CircuitConfig

encoder = VisualEncoder(
    EncoderConfig(mode="adaptive", code_dim=4000, sparsity=0.05, seed=7),
    CircuitConfig(duration_ms=50, dt_ms=1),
)
# Inputs: RGB or G/B, [batch, channels, height, width], float in [0, 1].
# Calibrate only on separate development images; this uses no task labels.
encoder.calibrate(development_images)
codes = encoder(test_images)  # [batch, 4000], binary KC activity by default
trace = encoder.diagnostics(test_images, record=True)
# trace["streams"] contains PN/KC rasters, spike counts, first-spike times,
# feedback traces and explicit continuation state.
```

Independent calls reset all neural state. Connectivity is fixed and seeded; noise and stochastic spike sampling are absent. Reproducibility is tested on CPU, not promised bit-for-bit across different devices or library versions. Default circuit settings are modelling assumptions; measured activity can differ from the calibration target.

## Historical evaluations

```sh
pixi run flowers
pixi run nav --ant 4 --routes 1
pixi run nav --ant 4 --routes 1 --without-resets
```

These preserve the earlier representation/readout pairings. In historical navigation, `--code_dim 4000` means **4,000 KCs per stream**, or 8,000 total. CLAHE with cosine template memory is a preprocessing/template baseline, not the complete Ardin spiking network. The older Izhikevich simulator remains under `apiaviz/mbant/`; its default membrane noise is nonzero.

The earlier pretrained `VisionModel`/`SNNVisionModel` instructions described removed code and no longer apply.

## License

See [LICENSE](LICENSE).
