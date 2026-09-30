# Synthetic closed-loop navigation videos

30 September 2026. Two short, prespecified demonstrations in a fresh scene on
the Studio. These use real camera acquisition and policy decisions, not scripted
paths. Both scheduled outcomes are included; no scene or controller tuning was
performed after viewing the results.

| Release | Outcome | Walked | Simulated time | Camera/score observations | Blocked contacts | Old-disc intersections |
|---|---|---:|---:|---:|---:|---:|
| On the taught route | Arrival | 1.70 m | 27.73 s | 128 | 0 | 3 |
| 24 cm towards the rock | Arrival; sustained route recovery | 2.00 m | 42.36 s | 185 | 0 | 39 |

Old-disc intersections count executed movement segments that would intersect the
discarded conservative discs, not distinct collision episodes. Every executed
segment passed the swept mesh/body check. Both runs had zero route resets and
zero motor-state resets. The displaced run still hesitates and reverses direction
near the rock (eight visual-steering episodes); arrival does not establish that
passing-side oscillation is solved.

The model is the existing `RefinementEncoder(method='linear_colour', seed=19)`
with fresh reproducible wiring and 8,000 output cells. The fixed visual front end
is untrained; mushroom-body template memory learns from 16 centreline teaching
views spaced 10 cm apart. Encoder and learned-memory fingerprints were verified
unchanged during both recalls. The continuous motor-feedback navigator and
camera-only optical-flow reflex use their default settings. Input remains
199 × 51 RGB. UV is not a navigation input in these videos.

The scene has a textured ground plane, five irregular rocks and sparse grass
landmarks. Blender Cycles CPU renders 720 × 152 panoramas at 32 samples/pixel,
which are sampled onto the existing angular grid. The evaluator uses
`rock-mesh-projection-v1` with a 5 mm circular body and nonterminal blocked
contact. Budgets were fixed beforehand at 40 macro commands, 60 simulated
seconds and 1,000 observations per run. The renderer was stopped afterward.

## Watch

- [On the taught route — 10 seconds](../../apiaviz/output/navigation-videos-mesh-v1/videos/clear-lane.mp4)
- [Released towards the rock — 14 seconds](../../apiaviz/output/navigation-videos-mesh-v1/videos/toward-rock.mp4)

Both videos are 1280 × 720 H.264, at 12 frames/second and 4× simulation time,
with a three-second outcome hold. Each shows the actual acquired camera input,
recorded overhead path, projected rock geometry and visual familiarity. Dashed
amber outlines show the old ignored discs. The map is display-only; neither
policy receives it. Camera frames and poses are held between logged samples.
The agent marker is enlarged for visibility.

The original historical study scenes/checkpoints are still absent. These are
illustrations from one small synthetic scene, not a reproduction of the original
failed trial or a comparative navigation result. No UV behavioral claim follows.

## Reproduce and verify

Run from the repository root, using a fresh output directory for a new run:

```sh
pixi run --locked python scripts/navigation_demo.py --output apiaviz/output/navigation-videos-mesh-v1
pixi run --locked python scripts/navigation_demo_video.py --output apiaviz/output/navigation-videos-mesh-v1 --trial clear-lane
pixi run --locked python scripts/navigation_demo_video.py --output apiaviz/output/navigation-videos-mesh-v1 --trial toward-rock
```

Blender requires execution outside the agent sandbox on this machine. The
navigation command owns its single renderer process and shuts it down in a
`finally` block. It never starts or resumes either historical study.

The output directory preserves the frozen configuration, source hashes, scene,
collision export, encoder checkpoint, teaching images, learned memory, complete
traces, camera cache and video provenance. Compact outcomes are copied to
[results.json](results.json). All large artifacts remain ignored. Both MP4 files
were decoded successfully with FFmpeg, and representative frames were inspected
for layout and alignment. The existing movement audit checked both trajectories,
including actual versus commanded distance, body/mesh clearance and final goal
distance; view/time budgets and model/memory fingerprints were also checked.
