# Spiking navigation without corrective resets

The research target is reliable autonomous navigation with a deterministic
spiking visual encoder, followed by a fair comparison of visual preprocessing.
Spiking is an architectural constraint, not a claim of superiority to an ANN.
The visual encoder remains fixed; route acquisition changes memory synapses.

**Implementation follow-up:** a [280-trial deep dive](frontend-deep-dive/report.md)
found promising development variants using oriented form filters or linear
colour differences, while simple form-channel balancing worsened performance.
The variants retain the same spiking circuit, teaching and controller. Neither
has a corrected significant advantage or independent-world validation; both
remain explicit experimental options. The original defaults and earlier results
below are unchanged.

**Latest matched study (29 September 2026):** the
[expanded 440-trial experiment](mechanism-study/report.md) did not reproduce the
four-route pilot's accuracy advantage. Across eight routes and five wiring
seeds, ApiaViz and Sobel plus colour each completed 29/40 trials, with mean
deviation of 55.2 and 50.4 cm. No baseline or component-removal comparison was
significant after its specified multiple-comparison correction. Local adaptation
made places more distinguishable, but both models showed weak restoring steering.
See the [three-page report](mechanism-study/mechanism-report.pdf) and
[reproducible data](mechanism-study/README.md). The development results below
remain useful history, not evidence of a replicated accuracy advantage.

**Development assessment:** retain this research direction, but do not claim robust
open-loop spiking navigation yet. A functional LIF latency encoder with full
templates completed 3/3 nominal development routes and 3/4 frozen new-route
cases. A finite-timing model using spike identities also completed 3/3 development
routes and 3/4 further frozen cases, without precise latency decoding for
retrieval. Release recovery and compressed MBON memory remain unreliable.
The two evaluation sets differ, so their equal success fractions are not a
paired timing comparison. These results do not establish an advantage over
Sobel or other matched filtering controls.

The subsequent [matched preprocessing comparison](mini-paper/apiaviz-mini-paper.pdf)
tested all six inputs with identical teaching images, frozen weights, spike
parameters, memory construction and steering. ApiaViz completed 3/4 routes,
grayscale 3/4, raw colour 2/4, simple opponency 2/4, Sobel plus colour 4/4,
and Ardin-style input 2/4. ApiaViz had the lowest mean route deviation
(11.5 cm versus Sobel's 18.5 cm, including failures). This suggested a tracking
precision advantage in the pilot, which the expanded study above did not reproduce.
See the [paired data and audit](mini-paper/README.md) for all 24 trials.

Here “open-loop” follows the project's usage: autonomous motion without
ground-truth corrective resets. The policy still receives visual feedback.
Use “autonomous visual navigation without corrective resets” in a paper to
distinguish this from both offline heading scans and open-loop control in the
control-theory sense.

## What the Nature Communications paper contributes

[Gattaux et al. (2025)](https://www.nature.com/articles/s41467-025-62327-3.pdf)
uses green-channel smoothing/downsampling and Sobel edges, followed by a fixed
sparse PN–KC projection and 1% k-WTA activity. Sobel does not itself enforce the
KC sparsity. Route memories use 15,000 KCs and four inputs per KC. Rotated
teaching views populate left/right memories; their difference steers the robot,
with familiarity modulating speed. Its binary “action potentials” are not an
explicit LIF time simulation. The useful architectural lesson is directional
memory acquisition and control, alongside the sparse visual code.

Our implementations below are adaptations, not reproductions of the paper.

Sparse single/few-spike KC responses have experimental precedent in
[moth olfactory recordings](https://pmc.ncbi.nlm.nih.gov/articles/PMC3124899/),
and [KC timing can affect downstream representations](https://pmc.ncbi.nlm.nih.gov/articles/PMC4189991/).
These motivate a first-spike reference but do not establish its physiological
parameters or transfer those olfactory findings directly to visual KCs.
[Localized APL inhibition](https://pubmed.ncbi.nlm.nih.gov/32955437/) also cautions
against treating our global inhibition switch as a literal anatomical model.

## Mechanisms under investigation

The earlier adaptive deterministic encoder has a graded colour/contrast front end,
LIF PNs and KCs, finite feedback inhibition and adaptation. Initial experiments
retained this encoder to isolate navigation failures. The experiments change
acquisition or retrieval, without fitting visual features:

1. **Parallel versus convergent teaching views.** Historical corridor views
   retain the centreline tangent at every lateral offset. Recognition can then
   be accurate without producing a turn back towards the route. Convergent
   acquisition renders each view looking towards a point 0.5 m farther along
   the taught route. This also smooths/anticipates bends, so it is not a pure
   manipulation of lateral correction. Coordinates are used only in teaching
   and scoring; memory stores neural codes/synapses, not waypoints for steering.
   A later `--convergence tangent` variant instead targets a point 0.5 m along
   the local tangent. Its correction is `-atan2(lateral_offset, 0.5)` and it
   preserves the original centreline heading exactly, separating inward
   correction from bend anticipation. The default `future` mode preserves the
   initial experiments for reproducibility.
2. **Compressed memory versus templates.** Compare the same spiking code in
   MBON population memory and a cosine template bank. The latter diagnoses
   information loss during storage and is not the desired neural architecture.
3. **Left/right associative memory.** Rotated teaching views depress two
   direction-labelled synaptic populations. Opponent synaptic currents command
   continuous turns from one current image, with no online scan. The prototype
   supports segmented memories; this differs from the paper's single route
   pair. Output currents are currently graded readouts of spiking KC codes;
   MBONs are not themselves simulated LIF neurons.
4. **Acquisition-order context.** A local gate restricts accessible MBON
   segments to a neighbourhood of the last visually selected segment. Only
   the chosen visual match advances context. This tests whether interference
   between distant route memories causes loops. It is currently a discrete
   algorithmic gate, not a demonstrated spiking recurrent circuit. It assumes
   release at the taught start and is not a general kidnapping solution.
5. **Additive prototype memory.** Average normalized KC codes within each
   segment and use normalized synaptic currents for retrieval. This preserves
   graded pattern structure with the same number of output units as the
   multiplicative-depression population. It tests the memory update rule;
   the readout remains graded and is not a simulated spiking MBON.

If a context gate proves essential, a neural account of its update and recovery
from mistaken state transitions is required. A successful algorithmic diagnostic
alone would not establish a fully spiking controller.

### A functional latency reference

The earlier PN/KC implementation changed several things relative to historical
ApiaViz: projection signs, input normalization, population size and the graded
readout. Its navigation performance therefore cannot be read as the effect of
converting that earlier computation to spikes.

`LatencyEncoder` provides an explicit reference conversion: retain the
historical fixed feature currents, drive LIF KCs, and inhibit the population
after the chosen spike budget. For nonnegative feature drive `d`, set `I=1+g*d`
and solve the first threshold crossing as `t = tau * log(1 + 1/(g*d))`. Zero drive
never spikes. Decode emitted latencies using `d = 1 / (g*expm1(t/tau))` before the
existing graded memory. This preserves the historical graded code in the ideal
limit, except for tied spikes and any spikes outside the presentation window.
The current gain `g` defaults to 1; changing it alters physical timing without
altering the ideal decoded pattern, provided the same cells fire within the window.

This is a deliberately idealized event-based reference: it assumes identical
initial states, instantaneous inhibition and precise latency decoding. It has
graded visual currents, spiking KCs and graded memory readouts, not spiking PNs
or MBONs. Each KC can emit at most one spike per presentation, with state reset
between views; sustained firing and across-view adaptation are not simulated.
Signed weights belong to effective feature-current computation and
are not claimed to be negative excitatory PN synapses. There is no trained
encoder. It tests whether information-preserving spike conversion is viable;
it does not by itself establish a physiologically realistic circuit.

`--inhibition-delay` exposes one approximation: positive delay allows extra
cells to fire. The implementation does not truncate those extra spikes back to
the nominal budget. Simultaneous ties also fire together. Finite delay, timing
precision and perturbation tolerance must be tested before promoting this
reference to the final architecture.

`--time-bin` rounds each candidate spike time upwards before population
inhibition and decoding. Thus finite resolution can change both the number of
emitted spikes and their decoded amplitude. A 1 ms bin is not a full 1 ms
Euler simulation: membrane crossings are still solved analytically.

For this encoder `--sparsity` specifies the inhibition trigger budget, not a
guarantee of actual activity with ties/delay. A fixed activity-only calibration
on 24 development images (eight each from Ants 1–3) found that triggering after
29 spikes per 4,000-cell stream (`--sparsity 0.00725`) gives mean activity
0.04891 with 1 ms bins and 1 ms inhibition delay. The two streams were 0.04052
and 0.05729. No trajectory success was used for that selection. The coarse
5% trigger gives approximately 9.6% actual activity instead.
[Calibration table](../apiaviz/output/openloop/finite-timing-activity-calibration.json).

## Reproducible development experiments

The runner saves settings, encoder/data/source fingerprints, result rows and
traces. It accepts spiking checkpoints or `--encoder latency`. Default development routes are
Ants 1–3; other ants require an explicit `--role evaluation`. This is a label,
not proof that a route has never previously been inspected. The earlier
Ants 4–15 pilot has already been examined and should not be described as a
pristine final test set after further tuning.

```bash
.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --checkpoint apiaviz/output/studies/20260924T080428.816984Z-56959/world-99/encoder-9184266bd368.pt \
  --ants 1 2 3 --memories population cosine --lookaheads 0 0.5

.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --checkpoint apiaviz/output/studies/20260924T080428.816984Z-56959/world-99/encoder-9184266bd368.pt \
  --ants 1 2 3 --memories sequential --lookaheads 0.5

.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --checkpoint apiaviz/output/studies/20260924T080428.816984Z-56959/world-99/encoder-9184266bd368.pt \
  --ants 1 --memories lateralized --viewpoints 1 --lookaheads 0 \
  --segments 80 --turn-gain 900

.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --encoder latency --ants 1 2 3 --memories population --memory-code binary --lookaheads 0

.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --encoder latency --ants 1 2 3 --memories population --memory-code binary \
  --lookaheads 0 --time-bin 1 --inhibition-delay 1

# Finite-timing spike-overlap candidate; activity-only calibration targets ~5%.
.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --encoder latency --ants 1 2 3 --memories spike_overlap --lookaheads 0 \
  --current-gain 0.1 --time-bin 1 --inhibition-delay 1 --sparsity 0.02825

# Recovery check with the same encoder and inward-facing teaching views.
.pixi/envs/default/bin/python -m apiaviz.research.openloop \
  --encoder latency --ants 1 --memories spike_overlap \
  --lookaheads 0.5 --convergence tangent --start-lateral 0.2 --start-heading 20 \
  --current-gain 0.1 --time-bin 1 --inhibition-delay 1 --sparsity 0.02825
```

`--start-lateral` and `--start-heading` impose release errors. The evaluator
uses the nest radius to terminate and score a trial; nest detection is not yet
an autonomous learned behaviour. This must remain explicit in reports.
The scan controller renders 13 candidate headings per movement step. The
left/right prototype instead uses one image and a continuous turn; its sensory
and action budgets therefore differ. No energy or real-time advantage has
been established from these simulations.

## Development observations, 29 September 2026

The adaptive-encoder experiments use a frozen 4,000-KC model with approximately
5% activity. The latency-reference experiments use 8,000 KCs. All use environment
seed 99, wiring seed 7 and a 200-step budget. These are development observations,
not a robustness claim. [All recorded outcomes](openloop-results.csv) retain
failures and explicitly distinguish development from evaluation.

On Ant 1, parallel corridor teaching gave **0/16 inward turns** at lateral
probes. Both MBON population and cosine retrieval failed to reach the nest.
Convergent teaching with the MBON population gave **15/16 inward turns** and
reduced mean route deviation from **1.20 m to 0.48 m**, but still failed.
The convergent trajectory repeatedly lost forward progress and only reached
the vicinity of route point 36 of 80. This motivates the context experiment;
it does not prove that memory aliasing is the sole cause.

The lateral probes are at acquisition corridor positions and test the learned
steering field, not generalization to unseen lateral positions. Their inward
turn fraction is a directional diagnostic, not a stability proof.

The initial segmented left/right prototype (16 segments, gain 180, quantized
10-degree turns) failed. A revision using continuous turns, 80 segments and
gain 900 also failed on Ant 1. Both are retained as negative observations;
several settings changed, so their difference is not a causal gain comparison.

The sequence-gated adaptive model completed **1/3 development routes** (Ant 2:
82 steps, mean deviation 0.071 m). Ants 1 and 3 failed. This particular sequence
gate is not a robust solution.

The functional latency reference restored autonomous homing on Ant 1 using
parallel corridor teaching:

| Readout | Homed | Steps | Mean route deviation |
| --- | --- | ---: | ---: |
| Binary MBON population | Yes | 110 | 0.166 m |
| Graded MBON population | No | 200 | 3.340 m |
| Cosine template diagnostic | Yes | 80 | 0.037 m |

The latency reference uses **8,000 total KCs**, historical signed/standardized
feature currents, and ideal first-spike competition; the earlier adaptive
encoder uses 4,000 KCs and different projection/input normalization. These are
architecture-development results, not a matched comparison of spiking
mechanisms. The binary-versus-graded MBON comparison within the latency
reference keeps the encoder and teaching exposure fixed.

Ant 1 success establishes feasibility on that route, not robustness. Subsequent
checks below separate nominal navigation, finite timing and release errors.

Subsequent development results show that the initial successes are insufficient:

- Ideal latency, binary MBON, parallel teaching: **2/3** nominal routes homed
  (Ants 1 and 3; Ant 2 failed).
- The same with 1 ms bins / 1 ms inhibition delay: **2/3** nominal routes homed,
  again Ants 1 and 3. Actual activity was approximately 9.6%.
- Ideal latency, binary MBON, future-point teaching: **1/3** nominal routes
  homed (Ant 3). Ant 1 did home when released at +0.2 m / +20°; parallel
  teaching failed from that same perturbed release. This is a local positive
  recovery result, not uniform robustness.
- Finite timing, future-point teaching, +0.2 m / +20° release: **2/3** homed
  (Ants 1 and 2). Ant 3 failed.
- The left/right prototype with the latency encoder, 80 segments and gain 180
  failed on all three development routes. It is not a reproduction or a
  refutation of the cited paper's complete controller, which also controls speed.
- The additive prototype memory failed on Ant 1. That branch was stopped;
  its manifest records the early stop, rather than implying a completed
  three-route study.

Preserving the centreline heading during convergent acquisition did not fix
Ant 1. The ideal-timing tangent variant failed there; Ant 2 eventually homed
after 183 steps, with mean deviation 1.067 m. Ant 3 homed after 135 steps,
giving 2/3 nominal successes. The activity-calibrated finite variant with a
perturbed release failed on all three ants. A high inward-turn
fraction at teaching locations therefore does not establish recovery or stable
route following. Ants 1–3 have been used for architecture selection; their
success rates are development outcomes only.

### Frozen reference evaluation

The ideal latency encoder with full cosine templates completed **3/3 nominal
development routes**. Settings were then frozen for Ants 4, 7, 10 and 13 on
Route 2, selected before observing their outcomes. These particular route cases
had not been used in this investigation. All use the same world geometry,
landmark-colour seed 99 and wiring seed 7; route-specific memories are acquired
normally. This is a small new-route check, not independent-world validation.

| Evaluation case | Homed | Steps | Mean route deviation | Maximum deviation |
| --- | --- | ---: | ---: | ---: |
| Ant 4, Route 2 | Yes | 112 | 0.184 m | 0.897 m |
| Ant 7, Route 2 | No | 200 | 2.408 m | 3.915 m |
| Ant 10, Route 2 | Yes | 81 | 0.027 m | 0.063 m |
| Ant 13, Route 2 | Yes | 86 | 0.071 m | 0.124 m |

**3/4 succeeded without corrective resets.** This is evidence of feasibility,
but it misses the proposed 90% nominal engineering target. The fourth route's
failure is retained, and these evaluation outcomes have not been used to tune
the reference. Success also permits substantial excursions, as Ant 4 shows;
completion and route fidelity must both be reported.

The memory stores 738–783 separate 8,000-cell patterns for these cases, versus
80 output weight vectors in the compressed MBON experiments. Its larger
storage budget and graded cosine readout are material limitations. Spiking KCs
alone do not make it a fully spiking navigation circuit.

Artifacts:

- [Ants 4 and 7](../apiaviz/output/openloop/20260929T003912.660663Z-60823/)
- [Ants 10 and 13](../apiaviz/output/openloop/20260929T003912.660664Z-60824/)
- [Evaluation trajectories](../apiaviz/output/openloop/frozen-evaluation.svg)

### Checks beyond ideal nominal navigation

Additional development checks retain the full template bank while separately
testing finite timing at approximately 5% actual activity and a +0.2 m / +20°
release. `SpikeOverlapMemory` adds a useful control: it stores and compares only
spike identities, discarding decoded latency amplitudes. Its normalized overlap
is still a graded readout, but it does not require precise amplitude recovery
from spike times. These checks are not tuning on the four evaluation cases.

A paired acquisition check then combined the stronger ideal template readout
with local-tangent convergent teaching. On Ant 1, from the same +0.2 m / +20°
release, parallel teaching failed whereas convergent teaching **homed in 83
steps**, with mean deviation **0.068 m** and maximum deviation 0.404 m. The
inward-turn probe fraction changed from 0/16 to 16/16. This isolates acquisition
in that case and provides a useful recovery result, but it is not a multi-route
robustness claim. The same acquisition is also tested with finite timing.

The initial finite-timing implementation had an important scale problem. On
eight Ant 1 development views, median first/budget-boundary spike times were
approximately 1.8/4.8 ms. The selected population occupied only about four
1 ms bins. Finite-timing templates completed **1/3 nominal development routes**
(Ant 3). The ideal template also completed only **1/3** from the +0.2 m / +20°
release (Ant 3). Thus nominal success with precise timing does not transfer
reliably to either condition. The finite spike-overlap control also completed
only **1/3** (Ant 3), at the original current gain.

A separate development experiment reduces current gain to 0.1, spreading
latencies while keeping tau at 10 ms and the presentation at 50 ms. This is an
engineering timing choice, not a fitted physiological estimate. On the same
24-image activity calibration, a trigger of 113 spikes per stream (0.02825)
gives actual activity 0.04970, with stream fractions 0.04640 and 0.05300.
No navigation outcomes were used to select that trigger. Both graded-template
and spike-overlap readouts are tested with these finite-timing settings.
The graded-template test still failed on Ant 1 (200 steps, mean deviation
0.881 m). Spreading the spikes over more bins fixes an encoding-scale problem
but is not sufficient with that readout. **Spike-overlap retrieval did home
on Ant 1** with the same scaled currents, 1 ms bins and 1 ms inhibition delay
(108 steps, mean deviation 0.153 m, maximum deviation 0.757 m). This is a
positive finite-timing result that requires no latency-to-amplitude decoding
for memory retrieval. It subsequently completed Ants 2 and 3 using unchanged
settings, giving **3/3 nominal development successes**:

| Ant | Steps | Mean deviation | Maximum deviation |
| --- | ---: | ---: | ---: |
| 1 | 108 | 0.153 m | 0.757 m |
| 2 | 103 | 0.085 m | 0.371 m |
| 3 | 116 | 0.348 m | 1.295 m |

It remains a template memory with graded normalized overlap, not simulated
spiking MBONs. The excursion on Ant 3 shows that reliable completion and
accurate route following are distinct targets. The model was then frozen for
four further Route 2 cases, Ants 5, 8, 11 and 14, selected before their outcomes
were observed. No tuning is performed on those evaluation outcomes.

| Frozen finite-timing case | Homed | Steps | Mean deviation | Maximum deviation |
| --- | --- | ---: | ---: | ---: |
| Ant 5, Route 2 | Yes | 82 | 0.041 m | 0.170 m |
| Ant 8, Route 2 | No | 200 | 0.374 m | 1.160 m |
| Ant 11, Route 2 | Yes | 76 | 0.034 m | 0.094 m |
| Ant 14, Route 2 | Yes | 80 | 0.013 m | 0.035 m |

The result is **3/4 nominal completions within the 200-step budget**. The model
is a viable finite-timing navigation candidate, not a demonstrated 90% robust
solution. These cases again share one physical world and wiring seed. Ant 8's
failure is retained; additional tuning would require a fresh evaluation set.

- [Ants 5 and 8 artifacts](../apiaviz/output/openloop/20260929T011500.321299Z-62148/)
- [Ants 11 and 14 artifacts](../apiaviz/output/openloop/20260929T011500.321299Z-62149/)
- [Finite-timing evaluation trajectories](../apiaviz/output/openloop/frozen-finite-evaluation.svg)

Local-tangent convergent teaching did not rescue the finite **graded-template**
readout on the perturbed Ant 1 case (200 steps, mean deviation 0.763 m), despite
16/16 inward-turn probes. The finite spike-overlap readout is a separate test.
It also failed from that perturbed release with convergent teaching (200 steps,
mean deviation 3.620 m), despite 16/16 inward-turn probes. The successful ideal
recovery result therefore **has not transferred to the finite-timing model**.
[Scaled-current calibration](../apiaviz/output/openloop/scaled-current-activity-calibration.json).
[KC event raster](../apiaviz/output/openloop/kc-spike-timing.svg) shows the actual
spikes for one development view; its per-view activity is not forced to equal
the calibration-set mean.

The artifact auditor recomputes every saved trajectory's nest outcome,
deviations, path length, movement kinematics and absence of resets, and checks
dataset/checkpoint fingerprints plus source archives where available. The
auditor also reconstructs scan choices from saved familiarity scores where
those records exist. Older missing score records and the distinct lateralized
policy are excluded from that controller-choice check. The
earliest runs record source hashes without an archived source copy; their
archival verification is explicitly marked unavailable. This audit does not
independently reimplement the renderer or visual encoder. Deliberately altered
nest outcomes and steering scores in temporary artifact copies were rejected.

Final verification for this round: **31 unit tests passed; 61 saved trajectories
and 5,600 recorded scan decisions passed the artifact audit with no discrepancies**.
All 28 run checkpoints/configurations and dataset fingerprints were checked.
Source archives were available and verified for 18 runs; the other 10 predate
source archiving. There are 27 completed runs and one explicitly stopped
prototype branch, with no experiments left running.
[Audit output](../apiaviz/output/openloop/audit.json).

```bash
.pixi/envs/default/bin/python -m apiaviz.research.audit_openloop \
  apiaviz/output/openloop/2026*
```

## Decision criteria and next stages

Carry forward the finite-timing phasic KC encoder and spike-overlap template
readout as the working diagnostic architecture. The encoder has fixed colour
and form currents, one spike at most per KC per presentation, and population
inhibition. The memory consumes spike identities. This demonstrates that exact
latency-to-amplitude recovery is not necessary for useful nominal navigation;
it does not establish that spike identities outperform other codes in general.

The next bottleneck is recovery and stable forward progress. The inward-turn
probes are necessary diagnostics but have repeatedly failed to predict stable
autonomous recovery. Test confidence-sensitive motion or route-context memory
as explicit controller interventions, with the encoder fixed and acquisition
held constant. Neither is an established fix, and the earlier adaptive-model
sequence gate already failed. Avoid further broad circuit sweeps before a
specific failure mechanism is isolated.

Keep compressed MBON memory as a separate target. Success with hundreds of
stored patterns does not validate the 80-output circuit. Any replacement must
preserve the useful visual similarities and be tested at a stated synaptic
storage budget; a fully spiking MBON readout is still unimplemented.

Then freeze settings and test across new routes, independent wiring seeds,
worlds and controlled release/visual perturbations. A working engineering
target is at least 90% nominal route completion, with recovery tested
separately; this target does not replace uncertainty estimates or guarantee
publication readiness. Changing landmark colours within the same geometry
does not count as a new physical world.

Only after a working spiking system is established, compare raw colour,
opponency, Sobel and ApiaViz using the same controller, teaching exposure,
usable KC capacity and activity budgets. Include colour-matched Sobel as well
as a green-channel implementation. A strong filtering baseline should receive
the same navigation improvements. The desired result is a large, reproducible
advantage in navigation-relevant robustness, not a win against a controller
that fails independently of its visual representation.
