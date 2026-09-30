# Workstation handoff — 30 September 2026

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
  UV sky polarization remains provisional. This has not been integrated into the
  navigation trials.
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
