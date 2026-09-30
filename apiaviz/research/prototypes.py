"""One-pass additive route memory as a control for multiplicative depression.

Normalized per-segment prototypes retain graded KC pattern structure. This is
a synaptic-current readout, not a simulated spiking MBON or a physiology claim.
"""
import torch
from torch.nn import functional as F
from apiaviz.nav.torch_route import CosineRouteMemory


class PrototypeMemory(CosineRouteMemory):
    def __init__(self, codes, segments=80):
        if codes.ndim != 2 or not len(codes) or segments < 1:
            raise ValueError("Nonempty codes and positive segment count required")
        segments = min(int(segments), len(codes))
        group = torch.arange(len(codes), device=codes.device) * segments // len(codes)
        views = F.normalize(codes.clamp_min(0), dim=1)
        prototypes = torch.stack([views[group == s].mean(0) for s in range(segments)])
        super().__init__(prototypes)
