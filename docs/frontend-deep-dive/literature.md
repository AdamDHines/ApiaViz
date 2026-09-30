# Research context for the frontend comparison

This review distinguishes computational proposals from demonstrated insect
mechanisms. The accompanying experiments test this implementation, not whether
insects use a particular convolution kernel.

**Gattaux et al. (2025): filtering is one part of the successful system.**
The Nature Communications model smooths and downsamples panoramic input before
Sobel filtering. Its route-centric lateralised memories, learning scans and
familiarity-dependent speed are central to its navigation design. Our Sobel
control instead supplies signed x/y derivatives and magnitude, with added
green–blue opponency, to the same memory and steering as ApiaViz. It is therefore
a deliberately useful input control rather than a reproduction of Antcar. The
paper supports studying image processing together with learning and control;
it does not establish that Sobel alone produces route convergence.
[Article](https://www.nature.com/articles/s41467-025-62327-3).

**Amin, Philippides and Graham (2025): recognising direction and returning to a
route are distinct problems.** Their simulations explain why a view-based
compass can follow a path parallel to a learned route after lateral displacement.
They test learning behaviours and cast-and-surge strategies that improve
convergence. This motivates a separate, matched acquisition/controller experiment
after the present input-only tests. A frontend that recognises displaced views
more consistently can still fail to produce an inward turn when teaching views
all face parallel to the route.
[Article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012798).

**Heeger (1992): gain control should have a specified denominator.** The model
normalises visual responses through pooled activity rather than relying solely
on a threshold. It provides a computational precedent for considering the gain
of different feature populations. Our per-plane RMS operation is an engineering
control inspired by that general principle, not a reproduction of the cortical
model or evidence for a particular insect circuit. Shared standardisation across
all feature planes does not equalise their individual contributions.
[Author-hosted paper](https://www.cns.nyu.edu/heegerlab/content/publications/Heeger-VisNeurosci1992a.pdf).

**Maisak et al. (2013): insects compute direction-selective signals, but motion
direction is not static edge orientation.** Recordings identified distinct
directional responses in fly T4/T5 pathways, separated by ON/OFF polarity.
This makes a richer directional pathway biologically worth considering. However,
our instantaneous Sobel derivatives lack the temporal computations of those
cells. Calling the candidate a T4/T5 model would be unjustified. A faithful motion
extension would require sequences, delays and tests against moving stimuli.
[Article](https://www.nature.com/articles/nature12320).

**Zhang (2019): filter placement matters for stability under small shifts.**
The study tests anti-aliasing before downsampling in trained convolutional
networks and reports improved shift consistency. Its direct performance findings
do not transfer automatically to this fixed insect-inspired network. It motivates
measuring stability around pooling boundaries and choosing spatial filter widths
in visual-angle units. Repeated smoothing, nonlinearities and downsampling should
be assessed as a combined sampling operation, rather than assuming that any
hexagonal averaging kernel guarantees robust retinotopy.
[Paper and code](https://proceedings.mlr.press/v97/zhang19a.html).

The [previous mechanism review](../mechanism-study/literature.md) additionally
covers insect spatial filtering, colour pathways, visual similarity and KC
recruitment, the original Ardin model, and statistical treatment of repeated
routes and wiring seeds. None of these sources establishes an accuracy advantage
for the current ApiaViz implementation; that must come from matched experiments.
