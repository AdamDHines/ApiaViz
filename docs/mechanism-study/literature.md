# Why might visual preprocessing improve route accuracy?

This review guided the component-removal and common-pose experiments. The
mechanisms below are hypotheses about this implementation, not established
properties of ApiaViz or claims about a particular insect circuit.

## Spatial averaging can stabilise views without erasing all directional information

Van Hateren's information-theoretic model predicts a balance between spatial
pooling and lateral inhibition: pooling becomes useful when noise dominates,
whereas band-pass filtering becomes useful when there is enough signal to
resolve spatial detail. His fly LMC experiments related these predictions to
measured responses. This supports testing the combination of smoothing and
contrast filtering, rather than assuming that maximal decorrelation is always
desirable. It does not establish that our hexagonal kernel is optimal.
[Theory](https://research.rug.nl/en/publications/a-theory-of-maximizing-sensory-information),
[fly physiology](https://research.rug.nl/en/publications/theoretical-predictions-of-spatiotemporal-receptive-fields-of-fly).

Gaffin and Brayfield measured how image resolution changes the spatial region
over which a stored view remains useful for navigation. Their results motivate
measuring both similarity between nearby positions and discrimination of
different directions. A representation can become more tolerant of movement
while losing precise directional information. Our neighbour/distant overlap
measure and familiarity curves address these two requirements separately.
[Gaffin & Brayfield, 2016](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0153706).

**Intervention:** bypass ApiaViz's hexagonal averaging while retaining its
other stages, and bypass its form filter bank in a separate condition. The latter
removes both ON/OFF centre–surround filters and the low-pass channel; it cannot
identify a specific contribution of the DoG surround alone.

## A useful sparse code preserves the right similarities

Dasgupta, Stevens and Navlakha linked expansion and sparsification in the fly
olfactory circuit to similarity search. The relevant principle is that related
inputs should recruit related activity patterns. Our signed input projection,
visual stimuli and spiking implementation differ from their model, so this is
a computational motivation rather than a reproduction.
[Dasgupta et al., 2017](https://pubmed.ncbi.nlm.nih.gov/29123069/).

Jesusanmi and colleagues directly compared image similarity with Kenyon-cell
activity in a spiking visual-navigation model. Their results support examining
the relationship between image changes and the recruitment of new KCs. They
also found interactions among exposure duration, connectivity and memory
learning. Their anti-Hebbian output-memory saturation mechanism should not be
transferred directly to our separate-template memory.
[Jesusanmi et al., 2024](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2024.1379977/full).

**Measurements:** per-view activity, the fraction of cells used anywhere along
the route, overlap between neighbouring versus distant teaching views, and
the participation ratio of the centred KC representation. A larger participation
ratio indicates more independent directions of variation, not automatically
better navigation. We also measure whether the correct heading beats competing
headings on identical probe images.

## Colour and form may contribute differently

Paulk and colleagues found anatomical and physiological differences among
bumblebee visual pathways responsive to colour, motion and stimulus timing.
This provides biological context for separating processing streams. It does
not show that our green–blue channels or fixed filters reproduce those neurons.
[Paulk et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18562602/).

**Intervention:** exchange only the colour pathway between ApiaViz and the
Sobel control. The four combinations are Apia form/Apia colour, Apia form/simple
colour, Sobel form/Apia colour, and Sobel form/simple colour. These reciprocal
swaps help distinguish form-pathway effects, colour-pathway effects and their
interaction. All retain the same two 4,000-KC populations and fixed connections.

## Accuracy and recovery depend on different properties of familiarity

Amin, Philippides and Graham explicitly separate route convergence from goal
arrival. Their simulations show why a visual compass can maintain a roughly
parallel path without reliably drawing an animal back towards the route.
This is directly relevant to our combination of accurate tracking and occasional
drift. A small heading error on the route is insufficient evidence of a useful
restoring response away from it.
[Amin et al., 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012798).

Gattaux and colleagues combine Sobel input with directional memories, learning
movements and speed control. Their findings support studying preprocessing as
part of a navigation system, but their complete robot cannot serve as an
input-only control. Our experiments hold learning and steering fixed and use
Sobel plus colour as a deliberately strong preprocessing control.
[Gattaux et al., 2025](https://www.nature.com/articles/s41467-025-62327-3).

Ardin and colleagues established an influential mushroom-body route-memory
model. Our Ardin-style control transfers its image operations to the shared
current model; it does not reproduce the original neurons, learning rule or
reset-assisted evaluation.
[Ardin et al., 2016](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004683).

**Measurements:** interstitial centreline probes, lateral probes at ±10 and
±30 cm, and full navigation trajectories. Interstitial views lie halfway between
teaching locations, preventing an exact stored-image match from making the
heading test trivial. We report both tracking error and completion.

## Brightness normalisation is already partly shared

This follows from the repository implementation. Each stream is centred and
divided by its within-view standard deviation before projection. For a linear,
positively homogeneous frontend, multiplying an image by a positive constant
is cancelled by this normalisation, apart from the small numerical stabiliser.
Consequently, global darkening is not a test that automatically favours ApiaViz.
Its local divisive adaptation and nonlinear compression can change relative
feature strengths even after global standardisation.

The processed opponent difference is formed *after* a `tanh` nonlinearity in
each channel. It should not be treated algebraically as an unchanged raw G−B
difference. The brightness probes and the adaptation removal test this actual
implementation. They are not evidence for general illumination invariance,
which would require more lighting conditions and real images.

## Statistical evidence must match the experimental units

Repeated steps on a trajectory are dependent observations. Kulkarni and
colleagues discuss how ignoring such structure can substantially alter
significance estimates. Here, five wiring seeds are repeated on each of eight
routes; we average seeds within route before the primary paired test. The
reported p-value therefore has eight route-level units, not thousands of steps
or forty independent animals.
[Kulkarni et al., 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010061).

Because the same wiring seeds occur on every route, route and seed are crossed
factors. We additionally resample both axes independently for a sensitivity
interval, following the rationale of the pigeonhole bootstrap. This can be
conservative, and five seed levels provide limited information about seed
variability. Neither interval measures uncertainty across world geometries:
only one geometry is used.
[Owen, 2007](https://arxiv.org/abs/0712.1111).

Colas and colleagues explain why the required seed count depends on effect
size, variability and statistical assumptions. Five seeds do not guarantee
adequate power. We fixed the matrix and stopping rule before inspecting its
outcomes, report confidence intervals and Holm-corrected two-sided tests, and
retain unfavourable results. We do not add seeds until significance appears.
[Colas et al., 2018](https://arxiv.org/abs/1806.08295).

The earlier [ApiaViz preprint](https://arxiv.org/abs/2602.06405) describes a
contrastively trained version. The present experiment concerns the repository's
subsequent fixed visual filters and route-template learning; conclusions about
one version should not be transferred to the other without testing.
