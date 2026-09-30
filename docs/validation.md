# Implementation validation — 24 September 2026

The deterministic circuit and experiment runner were exercised on the local CPU
with PyTorch 2.7.1 / Torchvision 0.22.0. These are implementation checks and pilot
measurements, not the complete paper study.

## Automated checks

`python -m unittest discover -s tests -v`: **17 tests passed**.

The suite checks repeated spike rasters, batch and order independence, fixed
wiring without global RNG consumption, spike-count conservation under temporal
perturbation, inhibition/adaptation interventions, explicit continuation state,
tonic-rate timestep refinement, calibration, unchanged encoder weights during
memory acquisition, total k-WTA budgets with a silent colour stream, equivalence
of the legacy projection adapter, no-evidence handling, blank/uniform images,
corruption reproducibility and clipping, navigation metrics/resets, disjoint
development images, and cluster/paired bootstrap behaviour.

Compilation and `git diff --check` passed. The historical flower classification
command completed, and the historical Izhikevich simulator imports and constructs
as a package after fixing its absolute imports.

## Flower pilot: 4,000 total KCs

340 images (20 per class), five-fold CV, one split/wiring/reward seed, clean
one-fixation memory acquisition. Threshold calibration uses 24 separate images
sampled across the development set. This is a new controlled comparison, not a
reproduction of the earlier colour-versus-CLAHE task configuration.

| Representation | Mechanism | AUC, 1 fixation | AUC, 8 fixations |
| --- | --- | ---: | ---: |
| Grayscale | k-WTA | 0.572 | 0.554 |
| Grayscale | Adaptive spiking | 0.524 | 0.526 |
| ApiaViz | k-WTA | 0.787 | 0.793 |
| ApiaViz | Adaptive spiking | 0.794 | 0.786 |

The paired spiking-minus-k-WTA AUC difference for ApiaViz was +0.007 at one
fixation (image-cluster bootstrap 95% interval −0.018 to +0.036), and −0.007 at
eight fixations (−0.028 to +0.014). This pilot does **not** demonstrate a spiking
advantage or statistical equivalence. It demonstrates that the implemented
spiking encoder can retain useful flower reward information without encoder
weight learning.

The adaptive ApiaViz development active fraction was approximately 4.92% against
a 5% target. Test KC events averaged approximately 223 per view at one fixation.
Activity and events are distinct measurements; they are not energy estimates.

Artifacts: [run manifest](../apiaviz/output/studies/20260924T060745.574627Z/manifest.json),
[results](../apiaviz/output/studies/20260924T060745.574627Z/results.jsonl),
[paired differences](../apiaviz/output/studies/20260924T060745.574627Z/paired-comparisons.json),
[plot](../apiaviz/output/studies/20260924T060745.574627Z/flower-pilot.png).

## Pretrained adapters

Official MobileNetV3-Small and DINOv2 ViT-S/14 checkpoints were downloaded and
tested with both GB-controlled and native RGB inputs, and global/spatial pooling.
All eight variants completed a separate 102-image, two-fold, one-fixation flower
validation run. Their readouts used graded associative memory, with no encoder
fine-tuning.

| Encoder | GB global AUC | GB spatial AUC | RGB global AUC | RGB spatial AUC |
| --- | ---: | ---: | ---: | ---: |
| MobileNetV3-Small | 0.791 | 0.823 | 0.858 | 0.885 |
| DINOv2 ViT-S/14 | 0.955 | 0.958 | 0.967 | 0.967 |

These scores are from a smaller sample than the 340-image KC pilot and must not
be used as a direct paired comparison to that table. They provide no evidence
that ApiaViz outperforms the pretrained models on flower choice. The full
comparison should place all encoders in the same run with identical splits.

Artifacts: [manifest](../apiaviz/output/studies/20260924T060251.485597Z/manifest.json),
[results](../apiaviz/output/studies/20260924T060251.485597Z/results.jsonl).
Checkpoint hashes and individual encoder settings are alongside these files.

## Navigation pilot: centreline memory

Route 1 of each evaluation ant (4–15; 12 routes), environment seed 99, wiring
seed 7, 4,000 total KCs at 5% activity, MBON population readout (S = 16),
single centreline memory view, 200-step budget. Calibration used ants 1–3.
Intervals resample ants. Each encoder ran as a separate process with identical
settings; the rows were merged for paired comparison.

| Representation | Mechanism | Offline heading error | Reached nest, reset | Resets per route | Reached nest, free |
| --- | --- | ---: | ---: | ---: | ---: |
| Grayscale | k-WTA | 29.3° (25.5–32.8) | 8/12 | 18.6 (12.6–25.8) | 0/12 |
| Grayscale | Adaptive spiking | 33.9° (32.4–35.5) | 9/12 | 24.1 (16.8–32.8) | 1/12 |
| ApiaViz | k-WTA | 20.0° (18.2–21.8) | 12/12 | 6.6 (5.3–7.9) | 0/12 |
| ApiaViz | Adaptive spiking | 21.1° (19.1–23.0) | 12/12 | 8.3 (6.8–9.6) | 0/12 |

Paired differences, candidate minus baseline (ant-cluster 95% interval):

| Comparison | Offline heading error | Resets per route |
| --- | ---: | ---: |
| ApiaViz − grayscale, k-WTA | −9.3° (−13.0 to −5.5) | −12.0 (−19.3 to −6.2) |
| ApiaViz − grayscale, adaptive | −12.8° (−15.0 to −10.4) | −15.8 (−24.3 to −8.5) |
| Adaptive − k-WTA, ApiaViz | +1.1° (−0.7 to +2.8) | +1.7 (+0.3 to +3.3) |
| Adaptive − k-WTA, grayscale | +4.6° (+0.5 to +8.6) | +5.5 (−5.3 to +15.8) |

In this pilot the representation effect is large and consistent: ApiaViz
reduced offline heading error and corrective resets relative to grayscale under
both mechanisms. The spiking circuit did not improve on k-WTA; for ApiaViz it
needed slightly more resets, and for grayscale it increased offline heading
error. No silent scans occurred. Adaptive memory-bank activity was 4.9%
(ApiaViz) and 5.1% (grayscale); test views produced approximately 210 and
238 KC events per view, respectively. Memory activity and test event counts
are different measurements.

Free navigation was unreliable: grayscale adaptive reached the nest on 1/12
routes and the other encoders on 0/12. The historical controller also failed
without corrective resets using centreline memory (Ant 1, progress 0.575) and
succeeded with a nine-view ±0.2 m corridor and S = 80. That comparison changes
both memory coverage and readout segmentation; it does not isolate a cause of
failure. One route per ant, one environment and one wiring seed limit these
conclusions. The colour and usable-capacity confounds described in the
[research audit](research-audit.md) also limit attribution to the representation.

Artifacts: [merged results and run index](../apiaviz/output/studies/navpilot-centreline-merged/runs.json).

## Navigation pilot: corridor memory, without resets

Same ants, routes, seeds and encoders, with the historical free-navigation
memory: nine viewpoints across a ±0.2 m corridor, MBON population S = 80,
200-step budget.

| Representation | Mechanism | Reached nest | Final nest distance | Mean route deviation |
| --- | --- | ---: | ---: | ---: |
| Grayscale | k-WTA | 1/12 | 5.77 m (4.18–7.16) | 1.59 m (1.16–2.05) |
| Grayscale | Adaptive spiking | 0/12 | 6.54 m (5.65–7.39) | 1.52 m (1.16–1.85) |
| ApiaViz | k-WTA | 4/12 | 3.45 m (1.99–4.80) | 0.69 m (0.42–0.95) |
| ApiaViz | Adaptive spiking | 2/12 | 5.11 m (3.45–6.77) | 1.32 m (0.82–1.88) |

Paired differences, candidate minus baseline (ant-cluster 95% interval):

| Comparison | Reached nest | Final nest distance | Mean route deviation |
| --- | ---: | ---: | ---: |
| ApiaViz − grayscale, k-WTA | +0.25 (−0.08 to +0.58) | −2.32 m (−4.61 to −0.13) | −0.90 m (−1.43 to −0.33) |
| ApiaViz − grayscale, adaptive | +0.17 (0.00 to +0.42) | −1.43 m (−3.45 to +0.55) | −0.20 m (−0.80 to +0.43) |
| Adaptive − k-WTA, ApiaViz | −0.17 (−0.50 to +0.17) | +1.66 m (−0.56 to +4.03) | +0.63 m (+0.02 to +1.33) |
| Adaptive − k-WTA, grayscale | −0.08 (−0.25 to 0.00) | +0.77 m (−0.80 to +2.32) | −0.07 m (−0.54 to +0.36) |

The corridor/S = 80 configuration permits autonomous homing on some routes, but it
remains unreliable: the best encoder reached the nest on a third of routes.
ApiaViz k-WTA stayed closer to the route and finished nearer the nest than
grayscale k-WTA. The adaptive spiking circuit did not help, and deviated
further from the route than k-WTA with ApiaViz inputs. The historical single-route
success (Ant 1, a calibration ant here) should not be read as typical. Twelve
routes are too few to rank success rates; the step budget, corridor width and
S were not swept.

Artifacts: [merged results and run index](../apiaviz/output/studies/navpilot-corridor-merged/runs.json).

## End-to-end coverage and remaining study work

- A 72-row navigation smoke matrix completed: two representations × two encoder
  modes × two corridor sizes × three memories × three protocols. It used Ant 4,
  Route 1, 256 KCs and only three movement steps. It validates execution and
  artifacts, **not** route success. This run preceded the final increase of the
  numerical tie tolerance to 1e-6; its heading scores are not paper results.
  [Artifacts](../apiaviz/output/studies/20260924T060746.784848Z/manifest.json).
- A 40-row flower matrix completed across four circuit interventions, five
  sensory conditions and two fixation counts, using fixed-reference thresholds
  and one acquisition example per class.
  [Artifacts](../apiaviz/output/studies/20260924T060745.574625Z/manifest.json).
- Classification with persistent fixation state and 0.5 ms timesteps completed.
  [Artifacts](../apiaviz/output/studies/20260924T060854.117079Z/manifest.json).

The full multi-ant, multi-environment, multi-seed navigation experiments,
behavioural timestep convergence, broader sparsity/presentation sweeps and
matched-split deep comparisons remain to be run using the commands in
[the study protocol](spiking-study.md). No claim of improved autonomous navigation,
spiking superiority or hardware energy savings has been established by these
validation runs. Output directories and downloaded checkpoints are ignored by
Git; this report preserves the scope and headline observations in tracked text.
