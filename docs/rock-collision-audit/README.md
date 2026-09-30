# Why collision terminations remain — 30 September 2026

The full suite does run the camera-only avoidance policy. At the initial 17-collision snapshot, every collision-ended trial contained executed evasive movements. This is not a run of the pre-avoidance controller.

However, `VisualLocomotion.advance` still ends the trial immediately when a proposed movement intersects an evaluator rock disc. The boundary is defined as `1.5*r`; the rendered rock is an irregular, rotated mesh generated with scale `(1.2*r, r, 0.7*r)`. Those discs can extend well beyond visible rock surfaces. The controller sees the rendered images and has a 4.5 cm visual-clearance setting, while evaluation can require substantially more clearance to a boundary it cannot see.

An independent geometry check uses the already exported meander mesh. It separates its 110 connected rock components, matches their centres to obstacle metadata, and computes the ant's distance from the rock's complete projected convex hull. Subtracting the maximum 2 cm next step gives a conservative clearance lower bound. The final observation position is used, including after an imposed displacement. The diagnostic does not change the world, trials or policy.

For `meander-19-linear_colour-right50-familiarity-1`, the last position is **16.7 cm** from the implicated rock's projected hull. A full next step would still leave **at least 14.7 cm**, yet crosses the evaluator's 41.3 cm-radius disc. This establishes that the `rock_collision` label cannot be treated as confirmed physical contact. Several other inspected meander terminations show the same discrepancy; the complete snapshot is in [meander-clearance.json](meander-clearance.json).

There are also controller weaknesses to investigate separately. The same example changes headings from 342° to 172° and back to 342° over its last three executed substeps, while consistently detecting frontal risk. The passing-side preference is a soft cost and can reverse. In other traces the last executed action is labelled clear before termination. Because the blocked action's decision and visual-point cloud were not recorded, these logs alone cannot apportion all failures between range estimation, path selection and the collision-boundary mismatch.

`displacement_into_rock` is distinct: an imposed 50 cm perturbation places the agent inside an evaluator disc. It is not a walked collision, and the disc/mesh distinction applies to it too.

The necessary correction is to align evaluator collision geometry with the rendered surface and an explicitly defined body footprint, while keeping ground-truth geometry out of the camera-only policy. Add a log of the rejected proposal and associated visual decision. Then assess actual missed obstacles and oscillatory steering under the corrected geometry. Existing trial outcomes must remain preserved and identified as conservative-disc outcomes; a corrected protocol requires separately labelled reruns.

Reproduce this read-only audit from the repository root:

```sh
apiaviz/output/uv-mitsuba/venv/bin/python scripts/audit_rock_clearance.py
```

This audit does not claim that all collision-ended trajectories would succeed if allowed to continue. It establishes an evaluator mismatch that must be resolved before those terminations can support claims about physical avoidance performance.
