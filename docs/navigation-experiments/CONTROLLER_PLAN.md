# A controller that responds to changing familiarity

Status: initial controller and development runner implemented, 29 September 2026.
See the [implementation and evaluation notes](../familiarity-controller/README.md).
The held-out study and training-density experiment remain gated follow-up work.

The aim is reliable route following when familiarity changes because of movement, sparse teaching or altered appearance. The controller should use the existing visual encoding and mushroom-body memory, remain deterministic, and recover without being reset onto the route. Robustness is a hypothesis to test, rather than an assumed property of the design.

The training-density question remains open. Controller development will initially use the existing centreline memories. A separate, fixed-controller experiment will then vary teaching spacing and image resolution; improvements from changing both together would be difficult to interpret.

## What needs to change

The existing `Active` policy in `apiaviz/research/active_navigation.py` alternates turns whose amplitude depends on the current familiarity level. It scans after three sufficiently unfamiliar observations. It does not compare familiarity before and after a movement, assess whether a scan contains a clear preferred direction, or retain a successful exploratory movement. Its calibration also treats rotated teaching views as negative examples, although these are not necessarily unfamiliar.

The completed experiments found that substantial numbers of off-route observations still exceeded its scan threshold. Those observations came from different trajectories across methods, so they do not establish a ranking of the encoders. They do establish a reason to test more than an absolute threshold. See [results and limitations](INTERPRETATION.md).

Finding the most familiar orientation and finding a movement that approaches the route are different tasks. Amin and colleagues showed that visual-compass guidance can produce parallel travel, and investigated spatial casting interrupted by movements that improve familiarity. Their cast-and-surge method still uses scans; it is a useful mechanistic reference, not an existing scan-free solution. [Amin et al., 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012798).

## Proposed behaviour

Use three explicit states so decisions are easy to inspect. All state changes depend only on observations already acquired and the agent's own movement history.

| State | Behaviour | Reason to change state |
|---|---|---|
| Follow | Maintain the current travel direction, with occasional small checks to either side. Continue while familiarity is stable and directional evidence remains useful. | A sustained decline, ambiguous directional checks, or an overdue exploratory check starts a cast. |
| Cast | Try a bounded movement to one side of the last supported travel direction. Assess whether moving improved familiarity. Alternate sides after unsuccessful trials and increase the search extent gradually. | Sustained improvement preserves the successful movement direction and returns to following; repeated unsuccessful casts trigger reorientation. |
| Reorient | Pause forward travel and physically sample a wider range of headings, expanding the scan only as needed. | A repeatable directional preference starts a short movement trial. Flat or conflicting responses lead to bounded further search, then an explicit unsuccessful termination if the search budget expires. |

At an unfamiliar release, begin with reorientation rather than assuming that the supplied starting heading is useful. A flat, highly familiar response must not suppress all exploration: a periodic small check is needed even without a familiarity decline. This cannot guarantee detection of every displaced position, but it avoids relying exclusively on a low-score alarm.

The first implementation will retain the current 10 cm movement increment. Subsequent work can couple slower speed and shorter movement to uncertainty, but this is a separate locomotion change and will receive its own comparison. This order lets us establish whether the decision rule helps before changing the movement model.

## Measure movement and turning separately

Use raw familiarity `F = -novelty`; higher values mean a stronger memory match. The controller needs three pieces of evidence:

1. **Directional preference.** From physically sampled headings at one position, estimate whether there is a useful peak relative to the other sampled directions. Record peak contrast and whether later checks support it. A broad flat response is ambiguous even when its absolute value is high. Any peak claim is limited to the headings sampled.
2. **Change following translation.** Choose a reference gaze direction, acquire its familiarity, make a movement, and reacquire that same gaze direction. Compare the two values over the distance actually moved. Do not subtract observations made at different headings and call the result a spatial gradient. Reference directions expire when the travel direction changes substantially; establish a new before/after pair instead of comparing across that change.
3. **Recent trend.** Smooth several valid movement comparisons to distinguish persistent deterioration from a single poor view. Require repeated supporting observations before declaring recovery. Use different entry and exit thresholds to prevent rapid switching between states.

A cast moves the body away from the current travel direction while its before/after reference observations remain aligned. Any necessary turns to acquire those views are physically executed and charged. If the cast improves familiarity, preserve its movement direction for a short surge instead of immediately turning back to the visual-compass heading. Re-check the improvement before extending that surge.

Maintain a fixed scale established from acquisition-only data and a separate recent-history trend. Never continually rescale a long unfamiliar episode until it looks familiar. Apply the same calibration procedure to each front end and memory bank, log raw scores, and label the output as a control signal rather than a probability of being on-route. Explicitly detect degenerate calibration and fall back to bounded exploratory behaviour. Additional calibration observations must be identical in pose across methods, counted, and excluded from route memory.

Changing sun direction can alter familiarity without indicating a wrong movement. Initially test synthetic score offsets, gains and abrupt common changes to diagnose controller sensitivity. Later use separately rendered lighting conditions to test actual appearance change. A controller cannot manufacture route information if the encoder no longer distinguishes useful views.

## Implementation order

1. **Inspect the available signal before choosing thresholds.** Use cached panoramas to assemble a common set of positions, headings and short movement pairs for all three encoders. Include on-route locations, teaching gaps, both lateral offsets and bends. Measure directional contrast and movement-induced familiarity changes. Ground-truth route direction may label these diagnostic plots, but must never enter the controller. Mark these worlds as development data.
2. **Implement and test the state machine on controlled familiarity fields.** Add `apiaviz/research/familiarity_controller.py` with explicit state, history and configurable limits. Use analytic fields to isolate a useful lateral gradient, an angular peak without any lateral information, broad peaks, false local maxima, delayed improvement, score drift and completely flat responses. In uninformative fields the expected behaviour is bounded search and an honest failure, not invented recovery.
3. **Connect it to a restricted sensor interface.** Add a separate experiment runner and adapter. The policy receives observation values, timestamps and self-motion only; it cannot access the existing sensor object's true position, route coordinates, obstacle list, endpoint or displacement notification. The evaluator owns rendering, movement, collision checks and stopping. Keep the existing controller classes and archived protocols intact.
4. **Run a small development comparison.** Use the three existing worlds, all three front ends and paired wiring seeds. Compare the new policy with exhaustive scanning and the old active policy under the same 10 cm movements, observations and time limits. Fix left/right tie-breaking deterministically and counterbalance the starting search phase. Develop one shared parameter set using pooled performance rather than tuning for an ApiaViz advantage.
5. **Freeze and evaluate.** Save controller parameters, source hashes, calibration rules, world seeds and the analysis protocol before the new test worlds are run. Only after this evaluation should speed modulation or continuously coupled turning and translation be added as separate experiments.

The oscillator evidence supports coupling visual information to movement rather than compulsory scans at every step. Ant lateral oscillations and forward speed are coordinated, with visual cues modulating oscillation amplitude. Our proposed states and thresholds are engineering approximations; they should not be presented as a demonstrated neural circuit. [Clément et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36538930/).

## Fair evaluation and the training question

Keep the current ApiaViz linear-colour variant, Sobel + colour and Ardin-style input processing. Use identical teaching poses, spiking parameters, memory rules, scenario definitions and budgets across methods. Ardin-style input remains a preprocessing control, not a replication of the complete original Ardin system. No route memories are added during evaluation.

First compare controllers with the existing 10 cm centreline teaching. Test aligned starts, both lateral releases, heading errors and mid-route displacement, including bends. New worlds must be independent of the existing three development worlds. Start planning for at least ten new scene–route pairs and five paired wiring seeds; use development variability to set the final sample size for the primary comparison before test outcomes are seen. Wiring seeds on one world are not independent environments.

Then freeze the controller and run the direct teaching-density experiment: 10, 20, 50 and 100 cm spacing crossed with 74 × 18 and 199 × 51 input resolution. Retain exhaustive scanning as the reference controller at each condition. Select teaching subsets by route arc length, counterbalance their phase, keep each subset identical across methods, and report endpoint inclusion rules and actual counts. Include evaluation poses between memories and avoid allowing every start to coincide with a stored view. Resolve calibration from each teaching bank using the same fixed procedure, without evaluation labels or condition-specific controller tuning. This experiment remains identifiable as a training study in its own report.

Use the same physical time, path-distance and observation budgets for controller comparisons. Report all three resources; fewer observations alone is not a success. Obstacles remain an evaluator collision rule shared by all methods. Do not give the controller privileged rock coordinates or imply that familiarity provides obstacle avoidance. Reject physically invalid imposed releases using a rule fixed before scoring, apply exclusions across all paired methods, and retain the excluded records.

## Measurements and decisions

- **Primary:** arrival probability and sustained return to the route after valid displacement. Report recovery time and distance with non-recoveries retained as failures or censored outcomes, not dropped from the averages.
- **Accuracy:** continuous route deviation, upper-tail deviation and distance to the endpoint. Keep success and early termination visible alongside deviation so a short failed trajectory cannot look artificially accurate.
- **Cost and behaviour:** time, walked distance, observations, rotation, scan bouts, state transitions, repeated searches and the outcome of each cast. Include failures in resource summaries and separately show matched successful runs.
- **Statistics:** paired differences between controllers within each world, method, seed and scenario; confidence intervals clustered at world level. Predeclare the main arrival and recovery comparisons. Do not count frames or repeated observations as independent samples. Treat the resolution–density interactions as a distinct analysis.
- **Mechanism:** on development worlds, ablate the temporal signal, directional contrast and periodic exploration separately. Test both starting phases. These comparisons establish whether each component helps rather than attributing every gain to the complete policy.

A candidate advances only if it preserves useful aligned-route accuracy and improves recovery over the old active policy without relying on a larger resource budget. Before held-out testing, specify a practically acceptable loss relative to exhaustive scanning and choose the sample size accordingly; absence of statistical significance is not evidence of equivalence. If confidence intervals remain too wide, report the result as inconclusive.

If controlled fields work but common rendered movement pairs contain little useful lateral information, pause controller tuning and report that limitation. That outcome would motivate changes to acquisition or representation as a separate study. If a useful signal exists but the controller cannot exploit it, continue controller development on the development worlds only.

## Deliverables

The implementation should produce a reproducible runner, focused behavioural tests, per-observation decision traces and a short report with paired results. A demonstration video should show the ant's view, route progress, raw familiarity, movement-induced change and current controller state. This will make it clear when the ant follows a familiar direction, tests a side movement or searches after losing useful guidance.
