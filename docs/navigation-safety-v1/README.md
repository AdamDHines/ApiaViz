# Safe perturbations and navigation failure audit

30 September 2026. Implemented as the separately versioned
`navigation-safety-v1` evaluator, on top of the
[mesh/body contact model](../collision-uv-integration/README.md).
No historical trial was changed or restarted.

## What changed

Releases and imposed kicks are checked **before** applying them. The evaluator
sweeps the circular body along the complete requested displacement against the
exported rock mesh projections and field bounds. If obstructed, it applies only
the collision-free prefix, with a 0.01 mm numerical backoff. If there is no room,
it skips the displacement. It never searches sideways for a favorable location.
An unobstructed displacement is unchanged. Checking the entire sweep prevents
teleporting through a rock even when the requested endpoint is clear.

Every placement records its requested and applied vectors, lengths, fraction,
limiting surface and adjusted status. A skipped kick is an attempted perturbation,
not an applied displacement or successful recovery. A genuinely invalid route
anchor raises a setup error before sensing; protocol preparation checks every
release and records the results. It does not quietly count malformed setup as
a navigation failure.

The following additional changes address reproducible implementation problems:

| Problem | Implemented behavior |
|---|---|
| A command crosses the invisible field boundary and immediately ends the trial | Reject the translation, charge commanded movement and camera time, and continue. The whole circular body must stay inside the field. |
| The ant reaches the goal partway through a 10 cm macro command, then receives a timeout | Check arrival after each completed substep. Keep the final partial macro trace. Also check after an imposed displacement. |
| All visual directions look obstructed, but the least risky forward direction is labelled `clear` | Report `avoid` and `visually_clear=false`. This also prevents the detour variant from treating an obstructed forward choice as visual clearance. The risk and turn-selection formulas are unchanged. |
| Opposing commands cancel and floating-point residue invents a net heading | Record an undefined heading (`null`), while retaining the familiarity comparison and bounded search counters. |
| Invalid speed, stride or turn-grid settings cause nonsensical accounting or loops | Reject invalid configurations explicitly. |
| NaN or negative-infinite familiarity scores are silently treated as unfamiliar views | Raise a numerical error instead of recording a behavioral failure. The documented positive-infinity silent-code sentinel remains supported. |

As with mesh contact, blocked field movement gives no contact signal to either
policy. Both receive images and commanded motion only. Actual displacement is
used for evaluator scoring and logs. Route memory persists; neither controller
is reset or guided by obstacle coordinates, endpoint bearing or route location.
Blocking can still lead to a legitimate budget failure if the camera-only policy
does not find a way onward.

## Evidence

The cancelled archive contains 145 completed trials:

| Recorded outcome | Count |
|---|---:|
| Arrival | 74 |
| Field boundary | 25 |
| Rock collision | 18 |
| Uninformative views | 14 |
| Time budget | 10 |
| Displacement into rock | 4 |

These are historical conservative-disc outcomes. Their counts do not establish
which individual trials would succeed after the fixes. In particular, the
reproduced false-timeout bug does not mean all ten archived timeouts were bugs.
The original scenes/checkpoints and full traces remain unavailable here.

All **126 tests pass**. The new regression cases cover clipped and skipped kicks,
initial releases, tunnelling, body clearance at field bounds, camera acquisition
outside rocks, policy information boundaries, immediate arrival with no remaining
sensor budget, partial-move arrival, numerical errors, ambiguous net heading,
and whole-block exclusion in the secondary dose analysis. The arrival regression
explicitly demonstrates an old `time_budget` outcome while already within the
20 cm goal radius, followed by `arrival` under the new evaluator.

A separate check used the actual five-rock mesh from the
[synthetic video scene](../navigation-videos-mesh-v1/README.md): all **240 sampled
50 cm displacement requests** produced collision-free swept placements. Of
these, 91 were shortened: 29 limited by rock contact and 62 by the field boundary.
These are geometry checks, not 240 navigation trials.

One illustrative request starts at `(0.65, -0.24)` m and requests a 50 cm lateral
kick. Its endpoint is clear but its path crosses the central rock. The safeguard
applies only **1.149 cm**, stopping before contact. A clear 50 cm request elsewhere
is applied in full. The compact [validation record](validation.json) contains
these examples, source hashes and the scene/export hashes. The full sampled
placements remain in the ignored output directory.

Exact commands used, from the repository root:

```sh
pixi run --locked test
pixi run --locked python scripts/audit_navigation_safety.py \
  --environment apiaviz/output/navigation-videos-mesh-v1/environment \
  --output apiaviz/output/navigation-safety-v1
```

The audit requires a fresh output directory on subsequent invocations. It does
not launch Blender or a navigator, modify the video scene, or replay old trials.

## Protocol and comparison rules

`scripts/prepare_mesh_protocol.py` now writes `evaluation_safety` with schema
`navigation-safety-v1`, preflights releases, freezes current source hashes and
preserves the source protocol. Use a new output such as
`apiaviz/output/route-navigation-safe-v1`, supplying restored environments,
verified collision exports and checkpoints as described by `--help`.
It prepares only; it does not launch workers. The historical study cannot yet
be prepared here because its scene/checkpoint assets are missing.

The continuous `route_full` worker passes the frozen safety configuration to
the evaluator, and the movement audit reconstructs every safe placement.
Protocols without this configuration retain their historical placement,
field-boundary and macro-arrival behavior. The clearance-label, net-heading and
numeric validation fixes are current source changes; historical reproduction
requires its original frozen source, not today's files. The old demo videos
remain demonstrations of the earlier mesh-contact protocol.

All scheduled outcomes remain in the primary denominator. Reports additionally
separate full, shortened, skipped and unattempted perturbations and show applied
dose ranges. A secondary full-dose analysis excludes an entire
world/seed/scenario block across methods and phases whenever any member had an
incomplete dose. That selection depends on trajectories and is descriptive.
Neither a shortened kick nor its resulting success is evidence of recovery from
a full 50 cm kick. Do not merge these results with the cancelled disc study or
the earlier mesh-only protocol.

## Remaining behavioral failures and proposed work

The existing `uninformative_views` outcome means repeated visual scans lacked
directional support. The code already attempts bounded exploratory casts between
scans; it is not an immediate failure on one unfamiliar frame. Keep it as a
behavioral outcome. Numeric faults now raise separately. Likewise, real time,
observation and movement limits remain enforced.

The displaced video still shows hesitation and passing-side reversals. A next
controller experiment should test a persistent, visually supported passing-side
choice, released by sustained visual clearance or a bounded search limit. Use
the same preregistered settings and teaching data for all front ends, include
mirrored obstacles and trapped/corner cases, and retain every outcome. This is a
proposal for a separate matched ablation, not a claim that the present changes
solve visual-flow failures or oscillation. No steering thresholds were tuned
from these outcomes and no long experiment was started.
