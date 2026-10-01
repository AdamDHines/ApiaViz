# Workstation handoff — 30 September 2026

**Latest: user-completed bounded nine-trial rerun, and measured regression diagnosis.**
`graded-navigation-smoke-v2-bounded` is now complete: **1/3 arrivals for each model**.
The agent supplied the rerun command while four behavioural tests were still
failing; it should have repaired and validated the controller first. See the
[bounded-run diagnosis](scan-regression-review-v1/README.md#completed-bounded-rerun-diagnosis)
and [reproducible nine-trial audit](bounded-scan-failure-audit-v1/results.json).

All nine encoder, memory and teaching-image fingerprints equal graded smoke v1.
Archived executable changes are confined to scan/controller configuration and
withdrawal of the panoramic class. At bend decision 22 all models stop at ±20°
and select 0°/5°, overlooking a stronger 45° match. At hairpin decision 32 ApiaViz
selects −15° and Sobel 0° instead of 45°. Same-pose recorded-score controls that
complete **only ±60°** recover the supported 45° direction in all five cases.
These controls establish a first-decision regression, not subsequent arrival.
ApiaViz bend/hairpin terminate without any blocked proposals. Later ambiguity,
edge-of-scan rejection and weak contrast are separate from the first wrong turn.

Bounded scans do not bound avoidance/return turns: Sobel bend has 179 actual turn
commands of at least 90°, including 180° turns; 118/120 blocks are field boundaries.
Recovery repeatedly relinquishes its evasive direction and resumes the bad route
heading. Do not attribute these actual loops solely to video aliasing or rocks.
The targeted familiarity suite was rerun: 10/13 pass, the same three behavioural
failures remain (the fourth known full-suite failure is in route avoidance).
This investigation changed no controller, encoder, physics, protocol or trial;
no new navigation/rendering run was launched. Full-circle scanning stays withdrawn.

**Full-circle scanning withdrawn at the user's explicit request (1 October):**
The [rotation investigation](scan-regression-review-v1/README.md) audits 18 traces
and 161 saved full-scan score tables. Mandatory whole-circle searches, introduced
in graded smoke v1, caused excessive physical scanning; the 8 fps/12-second movie
compression additionally aliases small yaw increments into large apparent jumps.
ApiaViz meander accumulates 22 equivalent revolutions despite no turn command
above 20°. Sobel meander increases from 6.3 to 26.4 equivalent revolutions versus
the preceding run. The old encoder/memory is unchanged for that Sobel comparison.

Active defaults are now bounded ±20° then ±60°, settings above 60° are rejected,
and `PanoramicConfirmationController` is withdrawn with an explicit error.
`graded_smoke` targets `graded-navigation-smoke-v2-bounded`, now user-run as above.
No vision/memory/physics changes accompanied withdrawal. Old results and archives
remain unchanged. Full-circle scans must not be restored to make tests pass.

**Current test status: 195/199 pass, four existing synthetic navigation failures.**
The bounded scan exposes a confidence-design defect: peak-minus-median contrast
depends on scan extent. For a broad useful cosine heading cue, ±60° gives 0.134,
below the unchanged 0.15 threshold, so the navigator incorrectly stops as
uninformative. All scan-removal/bounds and new timing-replay tests pass. The four
original behavioural assertions remain intact; no thresholds were quietly lowered.
Next agent should address confidence/search integration, then assess natural
movement and navigation in a fresh protocol. A 1×, 30 fps replay of unchanged v1
data demonstrates real scanning without the accelerated-movie jumps. No new
navigation trials/renderers/coordinators were launched. See the handoff below.

**Immediate independent-review request:** the user flags unnatural 180° turns
in the latest videos and asks for a handoff before further investigation.
Read [HANDOFF_NEXT_AGENT.md](HANDOFF_NEXT_AGENT.md) first. The latest switch from
progressive scans to mandatory full-circle 5° scans is an explicit regression
candidate introduced by the agent, not evidence of a UV defect. Natural movement
and video temporal sampling were not validated by the previous arrival audits.

**1 October completed nine-trial graded navigation smoke test:**
[Graded navigation smoke](graded-navigation-smoke-v1/README.md) is complete at
`apiaviz/output/graded-navigation-smoke-v1/`: **2/3 ApiaViz + UV, 3/3 Sobel +
colour, 1/3 Ardin-style arrivals**, one aligned positive-phase trial per model
per world, seed 19. All nine have verified camera/overhead movies. There are
zero blocked rock/boundary commands and zero corrective/motor resets; the screen
does not stress actual blocked-contact recovery or external displacement.

The opt-in `graded-angular-combined-v1` encoder preserves continuous amplitudes
through angular features, combined UV opponents, cosine template memory and
teaching-only calibration. Slow adaptation is off. Both sequential and parallel
trial entry points support it. Graded response activity is recorded separately
from spikes (`active_spikes: null`). Baseline encoders, memories and calibration
reconstruct exactly to their parent checkpoints. All models share full-circle
5° reorientation, confirmed exploration, coherent visual avoidance and unchanged
360-second/2,600-observation/200-movement budgets. This is a joint engineering
screen, not an isolated UV-effect estimate. Historical defaults remain available.

ApiaViz fails the hairpin via `uninformative_views`, not contact or near-arrival
scoring. At decision 42, 80° and 140° almost tie (0.971265/0.970820), matching
adjacent taught stations 40/41 around the turn. The competing-peak rule rejects
both, casts around the prior 45° reference and later stops. A same-camera replay
shows Sobel discriminates these alternatives more strongly and supports 80°.
The measured failure is a representation/ambiguity-policy interaction, not
proof of absent UV or of a controller defect independent of representation.
Retaining competing directional hypotheses for sequential visual disambiguation
is proposed, not implemented. Ardin fails meander on ambiguity and hairpin on
the observation budget. No exhaustive run should start on this evidence.

All **195 standard tests** pass. Every trial passes motion/provenance accounting;
all nine checkpoints reconstruct and videos decode with matching frame counts.
All jobs finished successfully and no workers remain active. No cancelled or
paused exhaustive study was restarted. User authorization covered these nine
trials and their videos, with review before any exhaustive rerun.

**1 October completed graded adaptation and opponency controls:**
[Graded UV controls](graded-uv-controls-v1/README.md) test a new opt-in temporal
response model, two opponent representations and matched graded/spike readouts
using 84 fixed image positions, three wiring seeds, three illumination gains and
both scan directions. These are offline image controls, not arrival trials.
With fixed response and equal stream weights, normal-light mean heading error is
**3.88° graded versus 7.63° spike identities** for pairwise opponency and
**3.68° versus 7.51°** for combined opponency. Removing UV from the graded code
raises error to **4.78°**. Effects differ by world; meander favours spikes.

The tested slow adaptation worsens normal/bright retrieval and makes scan
direction matter. A separately frozen follow-up retains fast integration while
replacing slow gain history with instantaneous gain: pairwise spike error falls
**11.45→5.54°**, and scan-direction disagreement **13.02→1.11°**. This supports a
history mismatch mechanism, not a rejection of biological adaptation. The gain
law and time constants are engineering approximations, not calibrated physiology.
The stronger lead is preserving response strength or richer spike information.

All **49,140 correlated records** across the main and follow-up controls are
complete, with source/input/model/memory checks passing. All **191 standard tests**
pass. Figures, compact results, limitations and reproduction commands are in the
report. Production defaults, physics, arrival conditions, Sobel and Ardin remain
unchanged. No paused or cancelled navigation study was restarted; no control job
remains running. Species/pathway calibration and full-route validation remain open.

**User clarification and biological interpretation:** precise route retracing is
not the objective; useful direction and destination arrival take priority.
Inward-projection fraction is only diagnostic. The
[biological follow-up](visual-response-controls-v1/README.md#biological-interpretation-and-the-users-navigation-objective)
explains the angular correction, existing graded front end versus absent temporal
photoreceptor adaptation, and physiological evidence for UV/opponent processing.
The UV smoothed filter's L2 normalization gives a kernel sum of 3.48057; feature
energy is therefore not an information fraction or a biologically calibrated
channel weight. Species/pathway-specific calibration remains open; the subsequent
graded-versus-spike controls are now completed above. No arrival threshold, model,
physics or trial result was changed by this clarification.

**1 October matched visual-response diagnosis:**
[Visual-response controls](visual-response-controls-v1/README.md) identify an
omitted pre-UV improvement: the older navigation and cancelled route-full
protocols used `AngularEncoder` (`angles` mode), but UV trial dispatch reverted
to pixel filters. Across 84 fixed, physically valid poses in all three worlds,
three wiring seeds and 20 explicitly separated conditions (5,040 records),
restoring angular visible filtering while retaining current UV reduces mean
taught-position heading error on the 10-degree grid from **10.20 to 1.76 degrees**.
This is an offline representation effect, not a new arrival result.

Current UV ApiaViz's same-view overlap loses 0.299 after a 5-degree turn versus
0.202 for matched 8k Sobel; the sharp response already exists before spikes.
UV feature energy is about 85% smoothed UV and 6% opponent planes, but removing
the dominant plane does not repair inward steering. With angular visible filters,
adding current UV improves displaced tangent error 10.86→9.27 degrees while
reducing inward 10 cm projections **62/126→48/126**. Orientation and route recovery
must be evaluated separately. Source/input/wiring/placement audits pass; all
184 standard tests pass. Four figures and compact numeric results are in the
new document directory. All control jobs finished; trial defaults are unchanged.
The latest full-route outcomes remain 1/6, 3/6, 2/6. No larger study was restarted.

**1 October implementation of the three fidelity priorities:**
[Surface, retina and directional-control validation](navigation-fidelity-v1/README.md)
records new opt-in source-informed spectral surfaces, linear solid-angle retinal
integration before response compression, and checked casting through both trial
entry points. The world builder had also disabled bump and leaf transmission;
the new path restores source settings and exported smooth normals, using explicitly
approximate spectral shaders. Geometry is exactly unchanged in all three worlds.
Movies use a fixed display transfer and annotate the configured sun direction.
Only ApiaViz receives UV; avoidance, Sobel and Ardin remain visible-only.

The fresh 18-case aligned matrix at `apiaviz/output/navigation-fidelity-v1/` is
complete: **1/6 ApiaViz, 3/6 Sobel, 2/6 Ardin**, versus 2/6, 4/6, 1/6 in v3.
All readiness gates fail. This is not a demonstrated navigation improvement;
do not promote it or start a larger study. Motion/accounting/provenance audits
pass all 18 traces. Rock rejections are 54/4/241 respectively; separate boundary
rejections are 215/246/135. All failures exhaust the normal time budget. There
are 16 selected report movies, all decoded and hash-verified. No trial workers
or coordinator remain active.
The standard suite passes 181 tests. Historical protocols and defaults remain
available and unchanged.

Diagnosis now distinguishes missed angular peaks, local scans stopping too early,
weak inward route recovery and frontal-flow blindness. ApiaViz hairpin passes
within 23.34 cm of the nest (20 cm arrival radius) without a blocked movement,
then departs. At a taught 45-degree bend, its score is 0.984 at 45 degrees but
only 0.665/0.722 at sampled 40/50 degrees, below its straight-ahead match.
The UV-off control shows the same ordering. Surface detail does not fix the saved
false-clear frontal rock approach. A separate `PanoramicConfirmationController`
requires the full heading circle during reorientation; tests and 24 recorded-image
scan replays pass, but its full-route effect is **unvalidated**. It is not part of
the 18-case matrix and is not enabled by default. Angular resolution, attraction
back toward the route, and robust visual go-around remain unresolved.

**1 October sun and comparison follow-up:** [same-pose audit](sun-matched-audit/README.md)
checks the user's concern about missing sun and lost earlier functionality.
Raw arrays contain the sun at the configured direction; small retinal sampling
and the dark display obscure it. Visible response/interpolation order produces
heading-dependent solar peaks (0.485–0.808 versus 0.997–1.000 for linear-first).
The spectral exporter genuinely omits Blender procedural texture, shader bump,
roughness and leaf translucency; navigation consequences remain unvalidated.
At 248 distinct cached positions from both models' v3 paths, ApiaViz + UV and
Sobel retrieve within 20° of the tangent at 107/107 near-route poses each and
103/141 versus 98/141 off-route poses. UV adds no net gain over ApiaViz visible
streams on this diagnostic. First steering disagreements reproduce from identical
poses; bend −1 decision 17 ignores a supported 0° direction while casting and
reverses on declining familiarity. A sign disagreement persists with matched RGB
and 8k wiring. This is evidence for a controller/feature interaction, not proof
of a UV advantage or a complete explanation of every failed route. No new trials
or renders were launched; all 278 diagnostic input hashes and 168 tests pass.
Production defaults and historical results are unchanged; the full study remains
paused. Raw analysis: `apiaviz/output/sun-matched-audit-v2/`.

**1 October deeper pipeline audit:** [UV pipeline v3](uv-pipeline-v3/README.md)
traces acquisition through learned memory and steering. Independent images show
working UV directional retrieval; recorded near-route views implicate controller
decisions rather than a missing UV input. A separate numerical control fixes
spurious form/UV spikes on neutral fields, with no changed spike bits on 144
textured acquisition views. A separate camera-order control aligns response and
interpolation order; neither control silently replaces old checkpoints.
The 18-case matched aligned-route controller validation completed at
`apiaviz/output/coherent-validation-v3/`, using unchanged v2 encoders and new
direction-confirmation/coherent-steering controls. Arrivals are 2/6 ApiaViz,
4/6 Sobel and 1/6 Ardin-style. ApiaViz rock-contact rejections fall from 798 to
24; the longest blocked-command sequence across the matrix falls from 742 to
four. All motion/provenance audits pass, but the overall readiness gate fails.
The ten report movies are decoded and verified; no workers remain active.
A separate two-phase meander mechanism check completed at
`apiaviz/output/confirmed-exploration-v4/`; it replaces scheduled blind casting
with a charged directional check, retaining the same v3 avoidance and v2 encoder.
Both phases arrive in 62 movements / 6.14 m, with zero blocked proposals. Traces,
unchanged encoder/memory/teaching fingerprints and the movie are verified. This
is a selected mechanism check, not a second comparative matrix or a passed
general-readiness gate. The standard suite now has 168 passing tests.
Do not merge its two successes into the v3 matrix or restart a full study.
UV still feeds route memory only; the common avoidance controller uses visible
RGB and can miss smooth frontal surfaces. Robust go-around and displacement
recovery with scheduled direction confirmation remain unvalidated.

**1 October correction/validation:** [UV v2 validation](uv-v2-validation/README.md)
adds a bounded receptor response before opponent processing, canonical camera
poses and independent teaching/recall Monte Carlo seeds. The bounded validation
completed all 36 primary trials plus 12 matched UV-off controls. Arrivals are
4/12 ApiaViz + UV, 5/12 Sobel, 2/12 Ardin-style and 4/12 ApiaViz UV off. All
conditions fail the prespecified aligned-route readiness threshold. Do not
launch another 378-trial study on this evidence. Physics, placement, accounting
and provenance audits pass for all 48 trials; reports and movies are verified.
Outputs are `apiaviz/output/uv-validation-v2/` and the separate
`uv-validation-v2-uv-off/`; an explicitly labelled UV-off presentation is at
`uv-validation-v2-uv-off-labelled/report/`. A separately tested
camera-only stall-recovery control escapes a saved blocked pose in both phases;
it is not enabled in the input-validation matrix. All 155 tests and 11,952
actual-world placement checks pass. Historical v1 outputs remain unchanged.

**1 October result:** `uv-trials-v1` is complete (378/378). ApiaViz + UV arrived
in 3/126 trials, Sobel in 63/126 and Ardin-style in 27/126. The
[failure diagnosis](uv-trials/diagnosis-2026-10-01/README.md) identifies extreme
bright-source concentration in the raw spectral opponent features and
same-pose rendering variability that triggers early casting. These are measured
integration problems. Corrected development results are recorded above;
reliable navigation is still not established.
Do not relaunch the full matrix as a substitute for complete-route smoke checks.

**Studio implementation update:** [mesh contact and UV rendering](collision-uv-integration/README.md)
documents the new opt-in mesh evaluator, nonterminal blocked contact, calibrated
spectral render interface and small integration checks. Two
[synthetic camera/overhead videos](navigation-videos-mesh-v1/README.md) are now
available. The subsequent [navigation safeguards](navigation-safety-v1/README.md)
prevent releases/kicks entering or crossing rocks, block field exits, fix arrival
accounting and record actual perturbation doses. Historical scene transfer and
a corrected smoke test in those original worlds remain outstanding. The
cancellation and historical provenance below remain authoritative.

**UV model update:** [ApiaViz UV integration](uv-model-v1/README.md) adds calibrated
UV receptor input, UV spatial/opponent processing and a third spiking stream.
Rendered view-memory and UV-only diagnostic smoke tests pass. Sobel/Ardin remain
RGB-only. The subsequent [two-command UV study](uv-trials/README.md) wires this
into closed-loop navigation: `pixi run environments` prepares fresh scenes and
`pixi run trials` runs the user-confirmed 378-trial matrix with collision
safeguards, progress bars, statistics, figures and movies. The user subsequently
started `uv-trials-v1` and requested parallel execution. It now uses the separate
`uv_parallel --workers 6` coordinator, preserving the frozen scientific protocol
and completed results. See the study notes for monitoring, stopping and resuming;
check `apiaviz/output/uv-trials-v1/progress.json` for live state. Large checkpoints
and renders remain local under `apiaviz/output/`.

The user will select the new branch, commit and push. No agent commit or push was
made during this handoff. Large artifacts remain local and ignored; this is a
source/documentation handoff, not a complete data migration.

## Run cancellation

The 378-trial continuous-navigation suite was cancelled at **145 completed trials**:
61 meander, 40 bend and 44 hairpin. The coordinator, all three workers and all three
associated background Blender renderers have stopped. One in-progress trial per
world was interrupted and has no completed result row. The exact record is in
[CANCELLATION.json](route-continuous-full/CANCELLATION.json).

The local `apiaviz/output/route-continuous-full/cancellation.json` and `progress.json`
record cancellation. Frozen protocols/manifests are preserved as originally
prepared; their old `running` status does not supersede the cancellation record.
Do not resume this run automatically. The earlier 181-trial controller study
remains a separate paused experiment.

Compact copies of the frozen protocol, source-hash manifest and completed summary
rows are under [route-continuous-full/archive](route-continuous-full/archive/README.md).
They are historical, incomplete conservative-disc results. Full per-step traces
remain in the ignored local output directory.

## First issue to fix

The camera-only reflex is active, but evaluation uses rock discs larger than the
visible rocks and aborts on entering a disc. An inspected ApiaViz trial labelled
`rock_collision` was still 16.7 cm from the rock's projected outline; its next
2 cm step would retain at least 14.7 cm clearance. This is an evaluator mismatch,
not evidence that the ant struck that rock.

Read the [collision investigation](rock-collision-audit/README.md) and its saved
numerical geometry audit. Align collision geometry with rendered surfaces and an
explicit body footprint without providing geometry to the visual policy. Add
logging of blocked proposals. Separately investigate passing-side oscillation
and missed visual obstacles. The user has also indicated other fixes are needed;
resolve their scope before launching another exhaustive suite.

Changed physics requires separately labelled reruns; do not silently combine new
outcomes with the partial run. No controller or collision fix was implemented as
part of the cancellation/handoff.

## Other recent work

- [Route/avoidance integration](route-detour/README.md): continuous route guidance
  with motor feedback, preserving familiarity comparisons through evasive movement.
- [Empirical UV calibration](uv-calibration/README.md): honeybee response tables,
  USGS material proxies, source checksums, controlled renders and numerical checks.
  UV sky polarization remains provisional and is disabled in the new UV trials.
- [Training and sensing experiments](navigation-experiments/INTERPRETATION.md):
  earlier evidence about easy acquisition conditions and poor displaced recovery.
- [Mechanism study](mechanism-study/report.md): the initial ApiaViz advantage did
  not consistently replicate; retain the matched Sobel + colour control.

## What Git includes

Source, tests, documentation, compact JSON/CSV summaries, selected PNG figures
and PDF reports remain eligible for version control. The ignore rules exclude
Blender `.blend` scenes and backups, video files, raw EXR/HDR images, array bundles,
checkpoints, pickles, large archives, datasets and all of `apiaviz/output/`.

Existing video links in older reports refer to local or separately transferred
artifacts. They will not resolve in a source-only clone until those media are
restored. Ignore rules retain local files; nothing was deleted. `.gitignore` does
not remove already tracked files, but the tracked-file audit found no large
Blender scenes, videos or checkpoints in the current index.

## Artifacts to copy separately or regenerate

Do not upload these directories with `git add -f`. Preserve the old workstation
until any desired data transfer has been verified.

| Local path | What it contains | Migration choice |
|---|---|---|
| `apiaviz/output/route-continuous-full/` | Cancelled trials, full traces, protocols, hashes and logs | Copy for full forensic analysis; compact summaries are archived in docs |
| `apiaviz/output/grassland-smoke/` | Meander Blender scene, world metadata, route, three seed checkpoints, teaching images and ~3.2 GB panorama cache | Copy scene/checkpoints/metadata for exact historical reproduction; cache is optional for new rendering |
| `apiaviz/output/navigation-regime/bend/` and `hairpin/` | Other worlds and several GB of cached panoramas | Copy to preserve exact scene geometry; otherwise regenerate and record new scene hashes |
| `apiaviz/output/uv-mitsuba/geometry/` | Exported spectral scene geometry | Copy or rerun `export_scene.py` against the restored Blender scene |
| `apiaviz/output/uv-calibration/sources/` | Downloaded calibration data and provenance | Copy or restore using the committed `sources.lock.json` |
| `apiaviz/output/uv-mitsuba/renders/`, `polarization/`, and `uv-calibration/renders/` | Linear spectral and Stokes arrays | Copy for exact historical pixels or regenerate using the documented commands |
| Other `apiaviz/output/` subdirectories | Historical studies, checkpoints and cached data | Copy if exact previous results need re-analysis |
| `docs/` video and Blender files | Demonstration movies and terrain previews | Optional separate transfer; scripts/notes remain in Git |
| `apiaviz/models/`, `apiaviz/mbant/data/`, external datasets | Historical model/data dependencies | Transfer or reacquire as required by the chosen experiment |

Do **not** copy live renderer queue state as work to execute. Old `queue/`,
`ready.json`, stop markers and runtime ports/PIDs are workstation state; inspect
and recreate renderer communication directories before starting a new job.

## New-machine startup

1. Check out the user's new branch and read root `AGENTS.md`.
2. Recreate the Pixi environment from `pixi.toml`/`pixi.lock`, rather than copying
   `.pixi/`. The declared CUDA environment is Linux-only; select it only on an
   appropriate machine. Verify the actual GPU and Blender backend separately.
3. Run targeted checks or `pixi run test`. Restore only the artifacts needed for
   the next diagnostic, verifying their stored hashes where available.
4. Fix and test collision geometry and resolve the remaining controller changes
   before generating a fresh evaluation protocol.
5. Use a small smoke experiment before allocating another exhaustive suite.

The previous renderer reports Blender 5.2.2 LTS. Spectral work used Python 3.12.11,
Mitsuba 3.9.1, Dr.Jit 1.5.0, NumPy 1.26.4, pandas 2.2.3, rdata 1.1.0 and xarray
2024.11.0. Preserve these as provenance, not a claim that every platform supports
the same binary environment. UV scripts default to a macOS LLVM library path;
set a valid `DRJIT_LIBLLVM_PATH` when needed on the new machine. Do not assume
choosing a CUDA environment changes a script's hard-coded Mitsuba LLVM variant.

Frozen protocols contain paths beginning `/Users/adam/repo/apiaviz/`. Do not
rewrite those records in place and expect their hashes to remain valid. Preserve
them for interpretation and create a new explicitly relocated protocol for new
work. The source lock for spectral calibration supports reacquisition:

```sh
python scripts/uv_mitsuba/fetch_calibration.py --manifest docs/uv-calibration/sources.lock.json
```

See the individual experiment notes for rendering, training and evaluation entry
points. This handoff deliberately does not start, commit, publish or upload them.
