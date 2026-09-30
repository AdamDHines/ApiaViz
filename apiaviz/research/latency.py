"""Functional event-based spiking reference for the historical visual encoder.

Graded, fixed feature currents drive ideal LIF Kenyon cells. Each KC emits at
most one spike; an ideal instantaneous population-inhibition event closes the
stream after k spikes. This is a reference limit, not a finite-delay APL model.
Signed feature weights are effective preprocessing currents, not a claim that
excitatory PN synapses have negative weights. No stochastic encoding or training.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import VisualEncoder, EncoderConfig
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize


@dataclass(frozen=True)
class LatencyConfig:
    tau_ms: float = 10.
    duration_ms: float = 50.
    inhibition_delay_ms: float = 0.
    time_bin_ms: float = 0.
    current_gain: float = 1.

    def __post_init__(self):
        if any(not math.isfinite(x) for x in (self.tau_ms, self.duration_ms, self.inhibition_delay_ms, self.time_bin_ms, self.current_gain)):
            raise ValueError("Circuit times must be finite")
        if min(self.tau_ms, self.duration_ms, self.current_gain) <= 0 or min(self.inhibition_delay_ms, self.time_bin_ms) < 0:
            raise ValueError("Invalid circuit times")


def latency_race(drive, k, config=LatencyConfig()):
    """Solve first spikes exactly for dV/dt=(I-V)/tau, V(0)=0, threshold=1.

    I=1+gain*drive: zero drive approaches threshold asymptotically and never spikes.
    Latency decoding recovers the graded drive of emitted spikes. Delayed
    inhibition allows additional cells to spike and exposes a physical limit
    of the ideal reference, rather than silently enforcing k after a delay.
    """
    if drive.ndim != 2 or not torch.isfinite(drive).all() or (drive < 0).any():
        raise ValueError("drive must be finite nonnegative [views, neurons]")
    if not 1 <= k <= drive.shape[1]:
        raise ValueError("Invalid population spike budget")
    latency = config.tau_ms * torch.log1p(1 / (drive * config.current_gain).clamp_min(torch.finfo(drive.dtype).tiny))
    latency = torch.where(drive > 0, latency, float("inf"))
    if config.time_bin_ms:
        latency = torch.ceil(latency / config.time_bin_ms) * config.time_bin_ms
    order = torch.argsort(latency, dim=1, stable=True)
    stop = latency.gather(1, order[:, k - 1:k]) + config.inhibition_delay_ms
    # Simultaneous events cannot be artificially ranked by neuron index.
    # Exact ties at the inhibition boundary therefore may exceed k.
    emitted = (latency <= stop) & (latency <= config.duration_ms)
    times = torch.where(emitted, latency, float("inf"))
    decoded = torch.where(emitted, 1 / torch.expm1(times / config.tau_ms) / config.current_gain, 0.)
    return {"codes": decoded, "spike_times_ms": times, "counts": emitted.float(),
            "inhibition_time_ms": stop.clamp_max(config.duration_ms)}


class LatencyEncoder(nn.Module):
    """Historical fixed currents -> LIF first-spike competition -> latency code."""
    def __init__(self, code_dim=8000, sparsity=.05, seed=7, circuit=None):
        super().__init__()
        self.config = EncoderConfig(mode="legacy", code_dim=code_dim, sparsity=sparsity,
                                    seed=seed, readout="count")
        self.circuit_config = circuit or LatencyConfig()
        self.features = VisualEncoder(self.config)
        self.eval()

    @torch.no_grad()
    def currents(self, images):
        form, colour = self.features.maps(images)
        drives = []
        for feature, projection in zip((form, colour[:, 1:]), self.features.legacy):
            flat = adaptive_avg_pool2d_anysize(feature, projection.pool_hw).flatten(1)
            flat = (flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-6)
            drives.append(F.relu(flat @ projection.connection))
        return drives

    @torch.no_grad()
    def diagnostics(self, images):
        streams = [latency_race(drive, projection.active_units, self.circuit_config)
                   for drive, projection in zip(self.currents(images), self.features.legacy)]
        return {"codes": torch.cat([F.normalize(s["codes"], dim=1) for s in streams], dim=1),
                "streams": streams}

    def forward(self, images):
        return self.diagnostics(images)["codes"]
