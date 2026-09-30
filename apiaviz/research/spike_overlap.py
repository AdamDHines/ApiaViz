"""Template retrieval using only which KCs emitted spikes.

This removes the need to decode exact spike latency into feature amplitude.
Retrieval is normalized spike overlap: a graded synaptic readout, not simulated
MBON spikes. It stores a separate pattern per taught view, without compression.
"""
from apiaviz.nav.torch_route import CosineRouteMemory


class SpikeOverlapMemory(CosineRouteMemory):
    def __init__(self, codes):
        super().__init__((codes > 0).to(codes.dtype))

    def forward(self, codes):
        return super().forward((codes > 0).to(codes.dtype))
