# Navigation videos: linear colour with original form

Open [the local video gallery](index.html), or play the MP4 files directly.
All three are 1920 × 1080 H.264 videos at 30 frames/s, with embedded explanations
and no audio. Each shows the complete recorded trial, including unsuccessful
navigation.

| Video | Duration | Outcome | Mean deviation |
|---|---:|---|---:|
| [Ant 4](ant-04-linear-colour.mp4) | 1:37 | Nest reached, 82 steps | 3.35 cm |
| [Ant 7](ant-07-linear-colour.mp4) | 2:23 | Nest reached, 128 steps | 18.08 cm |
| [Ant 13](ant-13-linear-colour.mp4) | 3:35 | Step limit, 200 steps | 65.38 cm |

These cases use Route 2, wiring seed 19 and environment seed 99 from the
[matched frontend evaluation](../frontend-factorial/report.md). They were
chosen to show a close return, a less direct return and a failed return using
the same seed. They are illustrative examples, not an additional performance
sample. The original model remains available; the videos use the opt-in
`linear_colour` variant without oriented form filters.

## What the panels show

1. **The ant's view.** A 296° panoramic view at the current position and scan
   direction. This is a separate, more detailed render for human viewing.
2. **Input and processing.** The actual 74 × 18 RGB render, neighbour averaging
   of its green and blue channels, the three original form maps, and the two
   rectified linear opponent maps. The five feature maps are shown before
   pooling to 64 × 8 each.
3. **Firing cells.** The actual binary outputs of the two 4,000-cell populations
   for the displayed candidate view. Grid position identifies a cell, not a
   location in the image. The video shows which cells fired, not their individual
   millisecond spike times.
4. **Route progress.** A map with the taught route, revealed travelled path,
   scan direction, current distance from the route, distance to the nest and
   distance walked. The ant icon is illustrative and not to physical scale.
5. **Heading scan.** Familiarity scores for the 13 candidate directions. Higher
   means a stronger match to stored route images; it is not a probability.
   Scores are revealed in scan order, then the selected direction is highlighted.

The input/feature/cell panels follow the highlighted scan direction. During a
movement they hold the last evaluated input, while the map and detailed view
move smoothly between the recorded positions. The large panorama and route map
never enter the navigation model.

## Playback and display choices

Each clip has a six-second introduction and a five-second result card. The
first two decisions are slowed to three seconds each. The remaining decisions
take one second each: 13 frames showing candidate views, seven frames holding
the chosen view, and ten frames showing the 10 cm movement. This pace explains
the algorithm; the evaluation does not assign a physical duration to a walking
step. The original 50 ms neural encoding window is unchanged.

The displayed panorama is rendered at 720 × 150 over 360°, cropped to the
current 296° view and resized for the panel. Movement uses linear interpolation
between recorded positions; those intermediate display images are not evaluated
by the model. Its real sensory input remains 74 × 18, with no high-resolution
preprocessing or antialiasing added to the navigation experiment.

Raw and feature panels use nearest-neighbour enlargement so their coarse
sampling remains visible. Feature colour intensity uses a fixed per-channel
99.5th-percentile absolute scale for the entire clip, with square-root display
contrast. This affects the illustration only. Form and colour colours are
visual labels, not simulated insect colour percepts. Display caches store
feature maps in float16; model computation and score verification use the
original float32 values.

## Verification and provenance

The video builder loads the original saved wiring weights and teaching images,
reconstructs the route memory, and verifies its fingerprint. It rerenders every
low-resolution scan, recomputes all 13 scores, and requires exact equality with
the recorded scores. Selected headings, no-evidence flags and 10 cm movement
endpoints are checked against the saved trajectory. No controller parameters,
learning images or model parameters are changed for a video.

Each video has a JSON sidecar containing the source run, source/data/checkpoint
checksums, verification counts, feature display scales, output checksum and
video timing. The three clips contain 410 navigation decisions and 5,330
candidate scores in total. The [video audit](audit.json) records the media
validation after rendering.

## Reproduce

The existing evaluation output and teaching-image banks must be present.
`ffmpeg` must be on PATH. The project environment already contains the Python
dependencies.

```sh
MPLCONFIGDIR=/tmp/apiaviz-mpl .pixi/envs/default/bin/python -m apiaviz.research.navigation_video --ants 4 7 13 --seed 19
.pixi/envs/default/bin/python docs/navigation-videos/audit.py
```

Use `--preview-only` to generate still previews. `--output` selects the output
directory. Audited sensor caches are kept under
`apiaviz/output/navigation-video-cache`; data and encoder/renderer source hashes
are checked before a cache is reused. A video-style change does not require
recomputing the same sensor outputs. The gallery is a standalone HTML file and
needs no server or external assets.
