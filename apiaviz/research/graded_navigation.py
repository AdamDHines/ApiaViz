"""Opt-in amplitude-preserving ApiaViz route representation.

Fixed response and combined opponency from the graded image controls. No slow
adaptation state. Memory and calibration must consume the same graded code.
"""
import numpy as np
import torch
from torch.nn import functional as F
from apiaviz.nav.torch_route import CosineRouteMemory
from .graded_uv import GradedUVEncoder, check_input
from .familiarity_controller import acquisition_calibration
from .spike_overlap import SpikeOverlapMemory


class GradedNavigationEncoder(GradedUVEncoder):
    response_format = 'graded'

    @torch.no_grad()
    def forward(self, receptors):
        check_input(receptors)
        pooled = self.pooled(receptors / (receptors + 1.), 'combined')
        streams = []
        for values, projection in zip(pooled, self.projections):
            flat = values.flatten(1)
            flat = (flat-flat.mean(1, keepdim=True))/(flat.std(1, keepdim=True)+1e-6)
            streams.append(F.normalize(F.relu(flat @ projection.connection), dim=1))
        return F.normalize(torch.cat(streams, 1), dim=1)


def memory_and_calibration(model, codes):
    if getattr(model, 'response_format', None) != 'graded':
        return SpikeOverlapMemory(codes), acquisition_calibration(codes.numpy())
    memory = CosineRouteMemory(codes)
    # Same teaching-only quantiles and floor as the binary baseline calibration,
    # evaluated on the actual representation used by the memory.
    x = codes.numpy().astype(np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('At least two finite nonnegative teaching codes required')
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    matrix = x @ x.T
    low, high = np.quantile(matrix[np.triu_indices(len(x), 1)], [.1, .9])
    np.fill_diagonal(matrix, -np.inf)
    calibration = dict(scale=float(max(.02, high-low)), low=float(low), high=float(high),
        familiar=float(np.median(matrix.max(axis=1))), degenerate=bool(high-low < .02),
        views=len(x), additional_observations=0,
        procedure='teaching_pairwise_q90_minus_q10_floor_0.02', representation='graded_cosine')
    return memory, calibration
