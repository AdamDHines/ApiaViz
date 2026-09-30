# What the three experiments tell us

The new tests do not establish a consistent navigation advantage for ApiaViz
over Sobel + colour or Ardin-style preprocessing. They do show that repeating a
familiar route and recovering after displacement are quite different problems.
The proposed active controller also needs further development before it can
serve as a useful test of the visual backbone.

The experiments were run in order: 36 acquisition trials, 189 recovery trials,
and 162 controller trials. The [full report](README.md) gives the methods,
figures, per-run data and verification. The backbone, spiking circuit and
template-memory readout were held fixed throughout, with matched teaching
images and wiring seeds across preprocessing methods.

**One taught traversal was enough for the original task.** All 36 acquisition
trials reached the endpoint. Reducing the bank from 693 views to 77 centreline
views changed ApiaViz's mean distance to the continuous route from 1.60 to
2.03 cm. Sobel + colour changed from 2.74 to 2.69 cm, and Ardin-style input
from 4.27 to 2.73 cm. Extra lateral memories were therefore not necessary for
basic retracing, and their effects were not consistently beneficial.

At an equal budget of 77 memories, replacing centreline views with a repeating
sequence of lateral views worsened all three methods. This shows that which
views are stored matters. It does not demonstrate that lateral exploration is
intrinsically harmful: the artificial sampling pattern omits many centreline
views and is not a natural learning trajectory.

**Displacement exposed a shared weakness.** We added two independently seeded
grasslands with substantial route bends. Their endpoints cannot be reached by
maintaining the starting heading. With centreline memories and exhaustive
scanning, every method succeeded in all 27 aligned or ±40° heading-offset
trials. Those releases remain at a taught position, and the correct heading
is within the scan range.

After valid sideways releases or mid-route displacements, results were much
weaker:

| Outcome | ApiaViz | Sobel + colour | Ardin-style input |
|---|---:|---:|---:|
| Nest arrival | 10/33 | 8/33 | 9/33 |
| Returned within 10 cm of the route for three consecutive steps | 7/33 | 6/33 | 7/33 |

The full experiment contains 36 displacement trials per method. Three per
method placed the agent directly inside a rock; the table removes the same
three world/seed/scenario cases from all methods. The unfiltered results are
retained in the report. This is a post hoc validity check, not a replacement
for the original trial record. Every lateral starting position also admitted
at least one collision-free first action in the exhaustive scan range; see
the [release feasibility check](release-feasibility.json).

The overall method ranking changed between worlds. Across all 63 recovery-phase
trials per method, arrivals were 37, 35 and 36 respectively. This small difference
does not support a claim that ApiaViz consistently outperforms the controls.

**The active controller saved observations but lost reliability.** It used
roughly 11–13 views per metre, compared with 130 for exhaustive scanning.
However, arrivals fell from 14/27 to 3/27 for ApiaViz, from 13/27 to 3/27 for
Sobel + colour, and from 13/27 to 0/27 for Ardin-style input. These are matched
scenarios with common limits on time, observations and movement steps.

This controller should not replace the reference implementation. The result
does not show that insects must scan exhaustively, or that active sensing cannot
work. Our candidate combines a particular oscillation rule, a low-familiarity
scan trigger and coarser scan angles. It also retains discrete 10 cm movements
and has no obstacle-avoidance system. These choices need their own validation.

**Familiarity alone is an incomplete guide to recovery.** In a post hoc analysis,
41% of ApiaViz's current-view queries made more than 20 cm off-route remained
at or above the confidence threshold that prevents a low-confidence scan.
The corresponding proportions were 63% for Sobel + colour and 59% for
Ardin-style input. The [confidence figure](confidence.png) shows the full
distribution. These are correlated observations within trajectories, not
independent statistical samples. Each method visits different positions, so
these proportions do not rank calibration accuracy on a common set of images.

This supports one limitation of the policy: a view can appear familiar while
the agent is displaced, so an absolute familiarity threshold can miss the need
to search. It is not a complete causal explanation of the failures. Many
off-route observations also have low confidence, yet the agent still fails;
triggering a scan and choosing a useful recovery movement are separate problems.

I would retain ApiaViz as a candidate, but stop presenting the present results
as evidence of general superiority over simple preprocessing. The immediate
priority is a validated recovery controller that uses changes in familiarity
during actual movement, together with a defensible acquisition procedure.
That controller should be developed separately from the front-end comparison
and evaluated on fresh worlds. A low average deviation on a familiar route is
insufficient to validate either component.

There are only three scene–route pairs and three wiring seeds here. The study
is useful for diagnosing the task and identifying failure modes; it does not
provide a strong basis for statistical significance or a general biological
claim. The original implementation and previous results remain intact.
