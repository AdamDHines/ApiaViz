# Expanded controller evaluation

The 756-trial matrix was fixed before the new runs. The exact machine-readable
protocol is stored in `apiaviz/output/controller-full/protocol.json`.

| Factor | Conditions |
|---|---|
| World and route | Original meander, bend, hairpin |
| Wiring seed | 19, 31, 43 |
| Image processing | ApiaViz linear colour, Sobel + colour, Ardin-style input |
| Start or disturbance | Aligned, left/right 50 cm, heading ±40°, left/right 50 cm kick before movement 36 |
| Controller | Exhaustive scanning, previous active policy, new policy with either initial search phase |

All conditions use the same saved world, 199 × 51 input, centreline teaching
every 10 cm, frozen encoder checkpoints, spiking circuit, route-memory rule,
10 cm movement increments and shared physical resource limits. Each world runs
in its own process. This changes scheduling only. Seventy-two verified smoke
trials are imported with trace hashes and source provenance; 684 trials are new.

The primary outcomes are endpoint arrival and sustained route return following
an intended displacement. Return requires three consecutive positions within
10 cm of the route. A later loss requires three consecutive positions more than
20 cm away. These are evaluated separately, since regaining a route does not
guarantee reaching its endpoint.

Both phases of the new controller are averaged within each matched world,
wiring seed, preprocessing method and scenario. Comparisons are then aggregated
within worlds. The three worlds, rather than frames, phases or wiring seeds,
are treated as independent units. World bootstrap intervals are descriptive;
this study does not claim broad statistical significance from three environments.

The primary analysis retains every scheduled trial. A separate sensitivity
analysis removes complete paired world/seed/scenario blocks when an imposed
displacement places any agent inside a rock or outside the field. The exclusion
applies to every method and controller in the block. It is trajectory-dependent
and therefore cannot replace the primary result. Ordinary collisions remain
failures, including collisions on the first attempted movement.

The report includes outcomes by scenario and world, initial-phase sensitivity,
paired controller and preprocessing comparisons, return followed by renewed
loss, sensing costs and termination reasons. Low deviation or low observation
counts from a short failed trial are not interpreted as successful navigation.

Six focused analysis tests check pairing, phase averaging, world-level
aggregation, invalid-displacement exclusions and sustained-return detection.
The controller implementation and its parameters remain unchanged.

This experiment expands the controller comparison. The longitudinal teaching
spacing and image-resolution experiment remains separate and has not been
replaced by these disturbance tests.
