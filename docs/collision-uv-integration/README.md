# Mesh contact and calibrated spectral rendering

30 September 2026. Implementation and small integration checks on the Mac Studio;
no navigation study was started. Historical disc results and frozen protocols
remain unchanged. The original scenes and checkpoints still need transfer before
testing the previously failing route.

## Collision behavior

The new `rock-mesh-projection-v1` evaluator sweeps an explicitly sized circular
body along each proposed movement against the union of projected, evaluated
Blender rock triangles. It tests segment interiors, edges, tangencies and the
body radius. It preserves concavities rather than filling them with a convex
hull, and does not use the old `1.5*r` discs.

This is a **planar, no-climbing model**: the complete mesh projection includes
overhangs and buried vertices. It does not claim to simulate height-resolved
body contact, limb motion or walking over low rocks. The body radius is required
in each new protocol; the tests' 5 mm radius is an engineering fixture value,
not an empirically calibrated ant morphology. Ground bump shaders affect shading,
not the exported physical surface. Only rocks are collision obstacles in this
version; field bounds retain their original centre-position rule.

As requested, genuine rock contact rejects that whole translation and **continues
the trial**. The evaluator logs the original position, proposed position, heading,
stride, visual decision, visual surface points and ages, and geometry provenance.
The command consumes its normal movement time and a fresh post-command camera
observation. Actual path length increases by zero for a rejected movement;
`commanded_path_m` records attempted movement separately. Macro steps finish after
10 cm of commands, so persistent blockage cannot trap the evaluator in an
unbounded movement loop. Time, view, step and controller stopping rules still
apply; continuing after contact does not guarantee arrival.

Both policies receive images and commanded self-motion only. Neither actual
movement success nor a contact flag enters the continuous navigator or reflex.
Its visual memory and matched-gaze familiarity comparisons persist. The legacy
motor-feedback field `walked_m` now explicitly identifies itself as commanded
odometry; evaluation `path_length_m` remains actual walked distance. A completely
blocked macro step cannot count toward a three-position route recovery.

An invalid release or external kick into a rock remains a distinct evaluator
outcome, not an attempted walk. Those checks also use the mesh/body geometry.

Historical protocols without `physics` keep the original discs and terminal
contact behavior. New protocols explicitly select hashed mesh assets, body
radius and `contact_response: block`. The old modules and outcomes therefore
remain interpretable, and corrected results cannot silently resume into old
manifests. `route_full.worker` and its audit use the same selected geometry.

### Export and prepare after artifact transfer

Export from each restored scene without rewriting it:

```sh
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/export_collision_geometry.py -- \
  --scene apiaviz/output/grassland-smoke/grassland.blend \
  --output apiaviz/output/collision-mesh-v1/meander.json
```

Repeat for bend and hairpin. The exporter uses named `Limestone ` objects,
evaluated world transforms and triangles, and records the scene and exporter
hashes. It rejects empty exports and differences between viewport/render modifier
visibility or subdivision levels. New output paths are mandatory. A changed
scene needs a separately prepared study; it is not historical reproduction.

After all three scenes, environment protocols and encoder checkpoints have been
restored, this command **prepares only** a new study (5 mm is an example body
radius that must be chosen explicitly):

```sh
pixi run --locked python scripts/prepare_mesh_protocol.py \
  --source docs/route-continuous-full/archive/protocol.json \
  --world meander apiaviz/output/grassland-smoke apiaviz/output/collision-mesh-v1/meander.json \
  --world bend apiaviz/output/navigation-regime/bend apiaviz/output/collision-mesh-v1/bend.json \
  --world hairpin apiaviz/output/navigation-regime/hairpin apiaviz/output/collision-mesh-v1/hairpin.json \
  --encoder-environment apiaviz/output/grassland-smoke \
  --body-radius-m 0.005 --output apiaviz/output/route-mesh-contact-v1
```

Preparation verifies original scene, environment-protocol and checkpoint hashes,
relocates paths in the new record, verifies rock counts, and freezes source and
collision hashes. Root/world manifests start as `prepared`; reports go inside
the new study directory. It launches neither workers nor renderers. Controller
oscillation and flow failure still need diagnosis in a bounded smoke navigation
test before any new full suite.

## UV rendering interface

`scripts/uv_mitsuba/render_sensor.py` is the new calibrated rendering entry point.
It accepts exported geometry, arbitrary explicit XYZ/yaw poses, camera elevation
limits, sun direction, seed and sample count. Defaults retain the 960 × 294,
360° azimuth, +90° to −20° spectral preview. It uses the archived empirical
honeybee response tables and USGS material proxies directly, with no global
callback replacement and no fallback for unknown materials.

The pinned dedicated Pixi environment contains Mitsuba 3.9.1, Dr.Jit 1.5.0,
NumPy 1.26.4 and LLVM 20. The active environment supplies LLVM; an explicit
`DRJIT_LIBLLVM_PATH` still takes precedence. This implementation selects CPU LLVM
spectral transport. It does not claim Metal acceleration or GPU equivalence.

```sh
pixi install --manifest-path scripts/uv_mitsuba/pixi.toml
blender --background --factory-startup --python-exit-code 1 \
  --python scripts/uv_mitsuba/export_scene.py -- \
  --environment apiaviz/output/grassland-smoke \
  --output apiaviz/output/uv-sensor-v1/geometry
pixi run --manifest-path scripts/uv_mitsuba/pixi.toml --locked python \
  scripts/uv_mitsuba/render_sensor.py \
  --geometry apiaviz/output/uv-sensor-v1/geometry \
  --output apiaviz/output/uv-sensor-v1/frames --polarized
```

Omit `--polarized` for intensity only. `--poses poses.json` accepts a JSON list,
for example `[{"position":[0,0,0.01],"heading":0}]`. XYZ is the camera location
in metres, including its height; do not substitute route coordinates or a
terrain-free default height for a real world. Exported route poses already have
the 1 cm camera offset above terrain. Every output directory must be fresh.

Each frame retains floating linear `intensity.npy` in UV/blue/green order.
Polarized rendering additionally retains `stokes.npy` with axes
`[I/Q/U/V, height, width, UV/blue/green]`; Q and U can be negative. Metadata pins
geometry files, source scene, calibration, renderer source, dependency lock,
camera, illumination, sampling, units, channel order and array hashes. The
calibration record is copied alongside the frozen render protocol. Completed
batches have a `complete.json` marker; interrupted outputs are not silently
reused. No PNG/display transformation is used as scientific input.

`apiaviz.research.spectral_input.load_frame(sidecar)` verifies labels, dimensions,
hashes and physical array constraints, returning intensity, optional Stokes and
metadata. It preserves values above one and signed polarization. It is the
data interface for the **next model update**; existing RGB encoders, checkpoints
and the RGB optical-flow preprocessing have not been switched to spectral input.
The biological placement of polarization-sensitive receptors, adaptation,
angular sampling and learned-memory retraining remain model decisions.

The spectral film uses explicit response functions, with linear output as
described in the [Mitsuba film documentation](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_films.html).
Outputs are relative band-weighted radiances, not absolute photon catches.
Coverage remains 320–700 nm; the missing shorter-wavelength response and material
proxy limitations in [the calibration record](../uv-calibration/README.md) still
apply. `--polarization-max` defaults to the earlier provisional 0.75; the
Rayleigh sky boundary is **not empirically calibrated UV polarization**.

## Verification and limits

All **114 unit tests passed**. Compact numerical results are saved in
[validation.json](validation.json).

The main suite checks false disc collisions, swept body clearance, concavities,
degenerate projected edges, scene/asset hash rejection, contact continuation,
sensor-only policy equivalence, time/view/movement accounting, external kicks,
continuous navigator persistence, fresh protocol preparation, and raw spectral
data integrity. The trial-matrix test now reads the committed archived protocol
rather than requiring a missing ignored output file.

```sh
pixi run --locked test
pixi run --manifest-path scripts/uv_mitsuba/pixi.toml --locked python scripts/uv_mitsuba/validate_sensor.py
```

The spectral numerical check uses constant and UV-blocked emitters. Observed
responses agree with independent wavelength integration within 0.000424 across
all bands. The empirical UV receptor's visible tail is retained.

A synthetic ground plane and transformed/subdivided rock were used for the
Blender-to-collision-to-spectral integration check. It has 480 evaluated rock
triangles; the float32 spectral export and float64 collision projection differ
by at most 3.88 × 10⁻⁸ m. Three 32 × 16 panoramas at four samples/pixel rendered
in each mode. Stokes values were finite and physically bounded; polarized I
matched intensity-only output within 1.5 × 10⁻⁸. These are implementation checks,
not image convergence, habitat calibration or navigation success evidence.

The exact bounded fixture workflow (choose a fresh output root on repetition):

```sh
blender --background --factory-startup --python-exit-code 1 --python scripts/create_sensor_smoke_scene.py -- --output apiaviz/output/studio-sensor-v2/synthetic
blender --background --factory-startup --python-exit-code 1 --python scripts/export_collision_geometry.py -- --scene apiaviz/output/studio-sensor-v2/synthetic/grassland.blend --output apiaviz/output/studio-sensor-v2/collision.json
blender --background --factory-startup --python-exit-code 1 --python scripts/uv_mitsuba/export_scene.py -- --environment apiaviz/output/studio-sensor-v2/synthetic --output apiaviz/output/studio-sensor-v2/geometry
pixi run --manifest-path scripts/uv_mitsuba/pixi.toml --locked python scripts/uv_mitsuba/render_sensor.py --geometry apiaviz/output/studio-sensor-v2/geometry --output apiaviz/output/studio-sensor-v2/uv-linear --width 32 --height 16 --spp 4
pixi run --manifest-path scripts/uv_mitsuba/pixi.toml --locked python scripts/uv_mitsuba/render_sensor.py --geometry apiaviz/output/studio-sensor-v2/geometry --output apiaviz/output/studio-sensor-v2/uv-stokes --width 32 --height 16 --spp 4 --polarized
pixi run --locked python scripts/validate_sensor_smoke.py --output apiaviz/output/studio-sensor-v2
```

Blender needed execution outside the agent sandbox for Metal initialization.
CPU Mitsuba renders completed inside it, with warnings that its external kernel
cache was disabled and Metal unavailable. The launcher was repaired to execute
the full Blender app path, because a bare symlink could not locate bundled Python.
All large fixture/render artifacts remain ignored under `apiaviz/output/`.
