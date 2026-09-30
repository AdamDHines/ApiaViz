# Research audit — 29 September 2026

This records the earlier pilot audit. Its source-match statements refer to
the files at that audit, before the navigation follow-up. The current priority
is [robust spiking navigation without corrective resets](openloop-spiking.md),
with spiking treated as an architectural constraint. That brief supersedes
the earlier experiment prioritisation below; the audited pilot results remain
unchanged.

The saved navigation results are internally consistent. They do not establish
a performance benefit from the current spiking implementation. There is a
promising representation difference, but the present controls cannot attribute
it specifically to ApiaViz's computations or to sparsity mechanisms.

## Verification

- All 17 research unit tests pass.
- Independently recomputed metrics from all 192 saved navigation result rows:
  offline heading choices/errors and silent flags; trajectory-derived nest
  distance, homing success, route deviation, travel distance and reset counts.
  No discrepancies were found.
- All eight saved encoder checkpoints load strictly and match recorded encoder
  fingerprints. World and route file hashes match the manifests; development
  ants 1–3 are disjoint from evaluation ants 4–15.
- Both merged result sets match their constituent runs, and the paired
  comparisons reproduced the reported confidence intervals.
- The circuit, encoder, navigation and visual-module source hashes match the
  current files for all eight runs. The first two centreline runs precede the
  output-directory PID suffix change in the study runner.

This is an artifact and implementation audit, not an independent rerun of the
eight full simulations. Merged result trace paths are relative to their
original run directories; use each merged directory's `runs.json` to resolve
them. Existing simulation artifacts were not altered.

## What the results support

| Measure | Grayscale k-WTA | ApiaViz k-WTA | ApiaViz adaptive spiking |
| --- | ---: | ---: | ---: |
| Centreline offline heading error | 29.3° | 20.0° | 21.1° |
| Centreline reset-assisted homing | 8/12 | 12/12 | 12/12 |
| Centreline resets per route | 18.6 | 6.6 | 8.3 |
| Corridor/S = 80 autonomous homing | 1/12 | 4/12 | 2/12 |

ApiaViz k-WTA gives better heading estimates than the tested grayscale code.
Autonomous homing remains unreliable. Its corridor success advantage over
grayscale is 25 percentage points, with an ant-bootstrap 95% interval of
−8 to +58 points. Adaptive spiking versus ApiaViz k-WTA is −17 points
(−50 to +17). These success estimates are too uncertain to rank reliably;
route-deviation results also give no encouragement for the spiking variant.

## Important limitations of the implemented comparison

1. **Usable KC capacity is not matched.** The original study harness allocates
   half of the KCs to each of two streams. Grayscale sets the colour stream to
   zero, leaving half its KCs permanently silent. At a global 5% target, 200
   active KCs occupy 10% of its available 2,000-cell pool, versus 5% of the
   nominal 4,000-cell population. Zero feature slots also change the effective
   fan-in statistics. This is a limitation of our original harness, not an
   error in the subsequent runs.
2. **Colour access differs.** The world contains synthetic chromatic landmarks.
   A grayscale control cannot exploit the same cues. Raw colour and simple
   opponent controls are available but were not included in these pilots.
   The observed advantage may partly reflect available sensory information.
3. **The memory intervention changes two things.** Centreline versus corridor
   changes both viewpoint coverage and MBON segmentation (16 to 80). It cannot
   isolate the benefit of corridor exposure.
4. **Generality remains untested.** Twelve routes use one world and one wiring
   seed. These are paired route-level results, not independent environment
   replications. Further tuning on these ants would make them development data.
5. **Historical models are not interchangeable controls.** The new binary,
   positive sparse projection differs from the historical signed, graded
   projection in several respects. This grayscale condition is not a complete
   reproduction of Ardin's model. New pilot failures do not by themselves
   invalidate previous results obtained with another encoder/controller.
6. **The current task/readout gives timing limited opportunity to help.** Views
   reset circuit state and the default memory consumes binary spike occupancy.
   Timing can influence spike generation, but the memory does not directly
   exploit precise spike times. These results test this conversion, not the
   usefulness of spiking computation in general.

The memory-bank lifetime usage is also very different: across centreline runs,
approximately 92% of grayscale k-WTA KCs are unused versus 64% for ApiaViz;
with corridor exposure, approximately 90% versus 39%. This makes cell reuse and
code overlap interesting measurements, but the silent-half confound must be
removed before treating this as evidence for a mechanism.

## Research decision

Do not lead with “spiking improves ApiaViz” or “beats deep learning.” Neither
is supported. The flower pilots also show no reliable spiking gain; the small
deep-feature pilot is encouraging for the pretrained features, although its
different sample prevents a formal paired comparison.

Spiking mushroom-body navigation is already established prior work, including
[Ardin et al. (2016)](https://doi.org/10.1371/journal.pcbi.1004683) and a
[2024 spiking robot navigation study](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2024.1379977/full).
A conversion to spikes alone is insufficient novelty. A useful neural
computation contribution would need a specific mechanism and causal tests.

The strongest remaining question is whether equally sparse codes preserve
different amounts of navigation-relevant information. Before investing in a
large sweep, run a bounded comparison of grayscale, raw colour, simple
opponency and ApiaViz with matched usable KC capacity, effective connectivity,
activity and memory settings. Include achromatic and chromatic worlds, and
reserve fresh worlds/routes for evaluation. Measure heading error, autonomous
homing, lifetime usage and near/distant code overlap together.

If raw colour or simple opponency explains the advantage, drop the claim that
ApiaViz's extra processing provides the benefit. If the advantage survives,
test which computation causes it. Keep the current spiking circuit as a
secondary implementation until a task requiring temporal processing gives a
clear, pre-specified reason to revisit it.
