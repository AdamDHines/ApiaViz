# Independent review handoff: unnatural rotations and UV navigation regression

Prepared first at the user's request, 1 October 2026, before the new rotation
investigation. Repository: `/Users/hinesa/repo/apiaviz` on the Mac Studio.
Read `AGENTS.md` and `docs/HANDOFF.md` as well. A later investigation addendum
will be linked below; do not mistake initial hypotheses here for measured results.

**Current update after the initial handoff:** the user explicitly required that
full-circle scanning be removed. It is now withdrawn from active code: default
scan extents are ±20°/±60°, extents above 60° are rejected, and the old panoramic
constructor fails explicitly. The graded factory is now versioned
`graded-navigation-smoke-v2-bounded` with a fresh default output path; it has
**now been run by the user: 1/3 arrivals per model**. Read the
[bounded-run diagnosis](scan-regression-review-v1/README.md#completed-bounded-rerun-diagnosis)
before proposing further trials. It proves premature ±20° acceptance at five
corner decisions; completing only ±60° selects the stronger 45° match in all five.
UV, memory and teaching fingerprints are unchanged. Actual avoidance/return
reversals also persist. The agent's rerun recommendation preceded repair of the
known failing behavioural tests; no repair should be inferred from that command.
Read the [completed rotation investigation](scan-regression-review-v1/README.md).

**Do not report the current suite as fully passing:** 199 tests ran after this
removal, 195 passed and four existing synthetic navigation tests failed. Their
failure exposes the confidence rule's dependence on wide angular coverage;
useful broad peaks are rejected by bounded scans. All removal/bounds checks pass.
The four assertions remain intact; thresholds were not silently retuned.
Resolve confidence/search integration without reinstating full-circle scanning.

## User's concern and requested priority

The latest nine videos look unnatural, with apparent 180° turns. The user says
the behaviour before the recent UV/controller work was much more insect-like
and suspects a deep regression. They want independent review, not more claims
based only on arrival percentages. The current agent was asked to write this
handoff **first**, then investigate excessive scanning. The user's weekly usage
limit is nearly exhausted; its exact remaining quota is not visible to the agent.

Natural movement, useful direction and destination arrival matter. Exact route
retracing does not. Do not tune/select runs to make ApiaViz beat Sobel. Do not
attribute behavioural failure to UV without checking acquisition, representation,
memory, confidence criteria, controller changes and presentation separately.

## Operational state and restrictions

- No trial workers/coordinator remain running. The latest nine-trial run finished.
- Do not restart exhaustive studies. The 378-case `route-continuous-full` run was
  cancelled after 145 trials; `controller-full` is separately paused. The user
  wanted review of the nine smoke trials before considering an exhaustive rerun.
- User owns Git commits/pushes/branches. The working tree has extensive prior
  tracked and untracked changes. Do not reset, stage, commit or switch branches.
- Preserve historical outputs and frozen protocols. Source snapshots are in
  study `source.zip` archives; current source hashes need not match older studies.
- Use Pixi. Standard checks: `pixi run test`. The nine-trial implementation had
  195 passing tests; current removal status is 195/199 passing as noted above.
- Policy inputs must remain camera images plus commanded self-motion. Geometry,
  route/nest coordinates, contact flags, depth and segmentation are evaluator-only.
- Read `docs/rock-collision-audit/README.md` before changing physics and
  `docs/uv-calibration/README.md` before changing spectral input.

## Last completed experiment

Output: `apiaviz/output/graded-navigation-smoke-v1/`.
Human report: `docs/graded-navigation-smoke-v1/README.md`.
All nine videos: `apiaviz/output/graded-navigation-smoke-v1/report/index.html`.
Numeric results: `docs/graded-navigation-smoke-v1/results.json`.

| World | Graded ApiaViz + UV | Sobel + colour | Ardin-style |
|---|---|---|---|
| Meander | Arrived, 204.30 s | Arrived, 230.79 s | Uninformative views, 217.23 s |
| Bend | Arrived, 138.96 s | Arrived, 154.96 s | Arrived, 198.09 s |
| Hairpin | Uninformative views, 155.58 s | Arrived, 187.26 s | Observation budget, 274.97 s |

One aligned release per model per previously inspected world, seed 19, positive
search phase: ApiaViz 2/3, Sobel 3/3, Ardin 1/3. Zero rejected rock/boundary
movements and zero corrective/motor resets in all nine. These are not independent
replicates sufficient for superiority claims or a stress test of contact recovery.

The final audit reconstructed all nine checkpoints/calibrations and decoded all
movies with matching frame counts. Baseline encoder/memory/teaching/calibration
match their parent checkpoints exactly. Those checks establish implementation
consistency, **not natural locomotion, useful scanning or biological validity**.

## Changes introduced in the latest turn — scrutinize these

The user accepted graded responses and requested three trials/model and videos.
The agent also changed the shared scan policy while incorporating accumulated
changes. This is an important confound and likely regression candidate:

```
navigation-fidelity-v1: scan_step=10°, scan_extents=[20°,60°,180°]
graded smoke:          scan_step= 5°, scan_extents=[180°]
```

The latter forces every wide reorientation to examine the full circle, even if
a small local scan would already have found an adequate heading. It was intended
to avoid missed narrow peaks/early local acceptance, but the nine-trial study
did not validate its effect on natural movement before applying it to all models.
The smaller scan step also affects local tracking checks. This was the agent's
implementation decision; do not frame it as a user request for continuous spins.

The separate `PanoramicConfirmationController` had previously passed selected
scan replays only; its full-route behaviour had been unvalidated. The latest
protocol incorporates that idea through settings rather than that class.
Arrival improvement is not evidence that this motor change is appropriate.

New files from that turn:

- `apiaviz/research/graded_navigation.py`: graded encoder adapter and amplitude-
  preserving memory/calibration. Fixed response `Q/(Q+1)`, combined opponency,
  equal stream L2 normalization; slow adaptation is off.
- `apiaviz/research/graded_smoke.py`: fresh nine-case preparation/run/assessment.
- `tests/test_graded_navigation.py`: four integration checks.
- `scripts/summarize_graded_navigation.py`: checkpoint/trace/movie audit.
- `scripts/audit_graded_smoke_ambiguity.py`: one selected hairpin replay.

Modified dispatch/integration: `uv_trials.py`, `uv_parallel.py`,
`uv_trial_report.py`, plus documentation. These files also contain earlier edits.
The adapter changes both serial and parallel entry points. Baselines stay
visible-only/binary; graded activity has `active_spikes: null` and a separate
`active_response_components` counter.

## Earlier relevant changes and evidence

1. UV integration dropped a prior angular-filter improvement by dispatching to
   pixel-scale filters. This was diagnosed in `docs/visual-response-controls-v1/`.
   Restoring angular filtering improved recorded-image heading retrieval.
2. Linear retinal integration before compression, source-informed spectral
   surfaces and sun-marked fixed-transfer displays were added in
   `docs/navigation-fidelity-v1/`. The preceding 18-case arrival result remained
   poor: 1/6 ApiaViz, 3/6 Sobel, 2/6 Ardin. These changes are not a proven cure.
3. `docs/graded-uv-controls-v1/` compares graded/spike readouts, two opponent axes,
   adaptation and illumination on fixed images. Normal-light error was 3.68°
   with combined-opponent graded UV versus 4.78° without UV and 7.51° with binary
   spike identities. This is representation evidence, not arrival validation.
4. Slow gain adaptation worsened retrieval and made scan order matter. It stays
   off. The parameters/gain law were engineering approximations, not fitted
   insect physiology. Preserving amplitudes is not a claim that biological MB
   neurons should be nonspiking. The graded readout is denser than binary codes.
5. Collision physics now blocks/logs unsafe swept mesh/body movements and allows
   navigation to continue. Continuous camera-only avoidance retains memory.
   Do not reintroduce the old conservative discs or terminal contact response.

The visual front end is fixed; downstream memory learns teaching templates.
Current memory is maximum cosine over all taught views, not a learned temporal
sequence. ApiaViz uses continuous projected responses; Sobel/Ardin use binary
identities with the same cosine retrieval rule. The comparison is consequently
a joint sensory/representation/controller engineering comparison.

## Existing specific hairpin finding

In ApiaViz decision 42, two separated directions almost tie:
80° → 0.971265, taught station 40 (heading 77.96°);
140° → 0.970820, taught station 41 (heading 135°).
Both are adjacent views around the turn. Their difference 0.000446 is below the
controller's ambiguity margin 0.004569; its >40° competitor rule rejects the
choice despite adequate contrast. It casts about the old 45° reference and later
terminates `uninformative_views`. Final nest distance is 3.82 m; no contact occurs.

The same-camera Sobel replay supports 80°, with stronger discrimination against
its competing 135° direction. Thus this is a representation/decision interaction,
not proof that UV is absent or that the controller alone explains everything.
See `ambiguity.json` and `ambiguity.png` in the graded smoke report directory.
Neither taking the top peak nor relaxing thresholds has been route-validated.

## Where to investigate excessive turning

- `familiarity_controller.py`: `_scan`, `_peak`, `step`. Wide scans, scheduled
  exploration, local ambiguity, fixed-gaze temporal comparisons and return turns.
- `confirmed_exploration.py`, `coherent_navigation.py`: direction-confirmed
  exploration and avoidance integration. Do not overlook changes shared by models.
- `active_navigation.py::Observations.rotate/observe`: yaw state and charged
  finite-rate turns. Observation yaw is the same state used for body motion;
  there is no independent head/gaze motor model.
- `avoidance_navigation.py::VisualLocomotion`: rotates then translates in
  microsteps. `motor_feedback.py` keeps route/avoidance interaction continuous.
- `uv_trial_report.py::make_movie`: currently takes the ant marker heading from
  the last observation, holds images between observations, and samples accelerated
  time at 8 fps in a 12-second clip. It does not interpolate logged turn events.
  Quantify whether this aliases smooth logged scans into apparent 180° jumps.
- Each trial JSON preserves `events` (turns, observations, avoidance views),
  `decisions`, `microtrace`, `trace`, raw-frame references and budget totals.
  Separate actual travel reversals, scanning body yaw, and display artefacts.
- Compare saved parent traces and archived source, not only current defaults.
  Find which earlier videos the user considers natural from local report history;
  the exact reference videos are not yet identified.

Priorities: quantify actual angular travel and scan frequency; attribute turns to
scan/return/reference/avoidance; replay identical cached views with progressively
changed scan settings; inspect video time sampling; establish an explicit natural
motor criterion before choosing a fix. Do not merely smooth a movie to hide real
policy spinning. Do not relax budgets or select favourable examples.

## Commands and continuation

Read-only study reconstruction and diagnostics (preserve existing report outputs;
use a fresh `--output` if rerunning a report):

```sh
pixi run test
pixi run python scripts/summarize_graded_navigation.py --output /private/tmp/graded-review-FRESH
pixi run python scripts/audit_graded_smoke_ambiguity.py --output /private/tmp/ambiguity-review-FRESH
```

Do not run `graded_smoke run`, `pixi run trials` or other study coordinators merely
to get context. Large artifacts live under ignored `apiaviz/output/`; the clone
alone will not reproduce the workstation's experiment state.

Investigation addendum: [scan-regression-review-v1/README.md](scan-regression-review-v1/README.md).
It covers 18 saved traces, 161 same-score scan controls, exact source/input hashes,
a 1× diagnostic video and the four failures exposed by removal. The agent's
mandatory-circle change is the primary identified motor regression; movie sampling
greatly exaggerates apparent per-frame turns. Older videos used different body
orientation illustrations and the original pre-UV trace tree is absent locally.

Preserved pre-removal analysis source: `apiaviz/output/scan-regression-analysis-v1/`.
Real-time diagnostic video: `apiaviz/output/scan-regression-review-v1/`.
No route experiment was rerun. No trial workers remain active. Historical current-
tree hash checks now reject old protocols as intended; reproduce them with their
archived source, never by changing their manifests.
