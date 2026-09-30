# Working on ApiaViz

## Start here

Read `docs/HANDOFF.md` for the current state and migration instructions. Read
`docs/rock-collision-audit/README.md` before changing navigation physics, and
`docs/uv-calibration/README.md` before changing spectral inputs.

The user handles Git commits, pushes and branch selection. Do not stage, commit,
push, publish releases, or switch branches unless they explicitly delegate it
again. Preserve existing user edits. Do not add large generated artifacts with
`git add -f` or weaken ignore rules to include them.

The 378-trial `route-continuous-full` run was **cancelled at the user's request**
on 2026-09-30 after 145 completed trials. Its coordinator, three workers and
three background Blender renderers were stopped. Do not restart it automatically.
The older `controller-full` study is also separate and paused. Existing results
are historical records, not permission to launch more long experiments.

## Research objective and comparison rules

- The central question is robust visual route navigation without corrective
  resets to the taught route. Spiking is a biologically motivated implementation;
  do not frame the paper as proving that spikes outperform graded networks.
- The visual front end is fixed and untrained. Downstream mushroom-body memory
  **does learn from teaching views**. Do not describe the whole system as having
  no learning. Seeded wiring is reproducible; distinguish it from analytic filters.
- Compare ApiaViz, Sobel + colour and Ardin-style preprocessing using the same
  teaching images, acquisition regime, mushroom-body model, learning, controller,
  resource budgets, seeds and evaluation. Label departures as separate controls.
- The current preferred ApiaViz variant uses linear colour and omits oriented
  form. Keep the original and prior ablations available. Never replace historical
  baselines silently or call an approximation an exact Ardin reproduction.
- Obstacle avoidance must use **camera images only**, plus commanded self-motion.
  Do not pass obstacle coordinates, depth buffers, segmentation labels, contact
  signals, route coordinates or nest bearing to either policy. Evaluator geometry
  is for physics, scoring and terminal conditions only.
- Route following and avoidance should interact continuously. Do not silently
  reset the navigator or discard its learned memory after an evasive movement.
- Do not tune or select runs to make ApiaViz win. Sobel + colour can be competitive
  or better. Report failures, computational costs and negative results. Distinct
  seeds, phases and stations within one world are not independent environments.
- Historical usage of “open-loop” means navigation without route snap-back; these
  agents still use sensory feedback. State the operational definition clearly.

## Immediate unresolved problem

`active_navigation.segment_collision` uses conservative circular rock bounds
(`1.5*r`) while the camera sees irregular meshes. `VisualLocomotion.advance`
terminates immediately when a proposed step enters a disc. At least some reported
`rock_collision` outcomes happen well clear of the rendered rock: the audited
ApiaViz example retains at least 14.7 cm clearance after a full 2 cm proposed step.
Avoidance was active; do not diagnose this as a missing policy.

Align evaluator geometry with rendered surfaces and an explicit body footprint
before treating those outcomes as physical collisions. Keep that geometry out of
the visual policy. Log rejected proposed movements and their visual decisions;
the cancelled run did not preserve the final blocked decision. Then investigate
genuine flow failures and oscillatory passing-side choices. External 50 cm kicks
into obstacle bounds are a separate outcome from walked collisions.

Use a fresh, versioned protocol/output directory for changed physics or control.
Do not edit frozen manifests to make old results appear compatible. The partial
145-trial results are not a final comparative result and must not be merged with
corrected trials as if they used the same protocol.

## Code and data map

- `apiaviz/src/modules.py`: original analytical graded front end.
- `apiaviz/research/`: matched encoders, spiking circuits, experiments and reports.
- `visual_avoidance.py`: deterministic optical-flow avoidance and local surface memory.
- `avoidance_navigation.py`: camera acquisition, locomotion and evaluator checks.
- `motor_feedback.py`: current continuous familiarity/avoidance integration.
- `route_full.py`: cancelled three-world experiment coordinator and worker entry points.
- `grassland_world.py`, `terrain_concepts.py`: procedural Blender environments.
- `scripts/audit_rock_clearance.py`: read-only meander mesh/disc diagnostic.
- `scripts/uv_mitsuba/`: spectral export, rendering, calibration and validation.
- `tests/`: controller, vision, budget, protocol and audit checks.
- `docs/`: research record, protocols, results summaries and selected figures.
- `apiaviz/output/`, `apiaviz/models/`, datasets and render caches: local artifacts,
  excluded from Git. A clone is not a copy of the workstation's experiment state.

## Spectral model boundaries

The UV prototype is separate from the RGB navigation experiments. Current
navigation input is 199 × 51 RGB; spectral preview panoramas are 960 × 294 × 3
UV/blue/green responses. Do not substitute PNG false colours as scientific input.

The empirical calibration uses published honeybee curves through a pinned pavo
revision and measured USGS material **proxies**. Archive sources, hashes, units,
interpolation and exclusions. No UV extrapolation or unsupported absolute photon
catch claims. Rendering currently covers 320–700 nm; missing shorter-wavelength
sensitivity is documented. Respect the archived source licences and attribution.

UV sky polarization is **not empirically calibrated**. The prior Rayleigh maximum
of 0.75 is provisional. The inspected RGB sky dataset has no UV measurement
channel; a physically richer simulated sky is not empirical validation. Preserve
raw linear arrays separately from display transformations.

## Implementation and verification

- Use the project Pixi environment; run `pixi run test` for the standard unittest
  suite, or targeted tests appropriate to the change. Do not launch Blender or a
  full experiment just to validate documentation changes.
- Add meaningful checks for changed collision geometry, sensor-only information
  boundaries, route/avoidance interaction and time/view/movement accounting.
- Freeze code, teaching-image, scene, checkpoint and protocol hashes for a new
  study. Cache render inputs with sufficient provenance. Preserve cancelled and
  superseded outputs instead of overwriting them.
- Recreate environments on a new OS. Do not copy `.pixi/` or virtualenv binaries.
  Some frozen protocols contain absolute paths from the previous workstation;
  preserve those records and create relocated/new protocols explicitly.
- CUDA availability alone does not accelerate Blender automatically. Existing
  research paths also make CPU assumptions. Inspect and benchmark each component
  before changing backends, and validate numerical equivalence where relevant.
- Document substantive choices and results in `docs/`, including exact commands,
  limitations and failure modes. Write for a broad scientific audience; distinguish
  measured evidence, interpretation and hypotheses.
