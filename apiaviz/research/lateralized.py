"""Route-centric left/right mushroom-body memories for spiking KC codes.

Inspired by Gattaux et al. (2025), doi:10.1038/s41467-025-62327-3.
This is a research adaptation, not a reproduction of their robot or encoder.
Readouts are normalized synaptic currents; PN and KC spiking is supplied by
VisualEncoder. The output neurons are not yet simulated as LIF neurons.
"""
import numpy as np
import torch
from torch import nn


class LateralizedMemory(nn.Module):
    """Local one-pass depression gated by the sign of a teaching rotation.

    Each segment has left-looking and right-looking synapses. Select the most
    familiar pair using sensory activity alone, then turn away from the familiar
    error direction. There is no stored coordinate or online route-index input.
    Multiple segments are an explicit departure from the paper's single pair.
    """
    def __init__(self, codes, rotations, groups=None, depression=.5):
        super().__init__()
        if codes.ndim != 2 or not len(codes) or not torch.isfinite(codes).all():
            raise ValueError("Expected nonempty finite [views, KCs] codes")
        rotations = torch.as_tensor(rotations, device=codes.device)
        groups = torch.zeros(len(codes), dtype=torch.long, device=codes.device) if groups is None else torch.as_tensor(groups, device=codes.device)
        if rotations.shape != (len(codes),) or groups.shape != rotations.shape:
            raise ValueError("Each code needs one rotation and one segment")
        if not 0 < depression <= 1 or not torch.isfinite(rotations).all():
            raise ValueError("Invalid depression or rotations")
        activity = (codes > 0).float()
        weights = []
        for group in torch.unique(groups, sorted=True):
            pair = []
            for mask in (rotations >= 0, rotations <= 0):
                selected = activity[(groups == group) & mask]
                if not len(selected):
                    raise ValueError("Each segment needs both left- and right-looking examples")
                pair.append((1 - depression * selected).prod(0))
            weights.append(torch.stack(pair))
        self.register_buffer("weights", torch.stack(weights))

    @torch.no_grad()
    def forward(self, codes):
        activity = (codes > 0).float()
        total = activity.sum(1, keepdim=True)
        novelty = torch.einsum("bk,slk->bsl", activity, self.weights) / total.clamp_min(1)[:, :, None]
        segment = novelty.mean(2).argmin(1)
        pair = novelty[torch.arange(len(codes), device=codes.device), segment]
        signal = pair[:, 0] - pair[:, 1]
        signal = torch.where(total[:, 0] > 0, signal, 0.)
        return {"turn_signal": signal, "novelty": pair, "segment": segment,
                "no_evidence": (total[:, 0] == 0) | (signal.abs() <= 1e-6)}


def rotated_acquisition(positions, headings, rotations, segments):
    rotations = np.asarray(rotations, dtype=float)
    if not len(rotations) or segments < 1 or not np.isfinite(rotations).all():
        raise ValueError("Nonempty finite rotations and positive segment count required")
    n = len(headings)
    pos = np.repeat(positions[:n], len(rotations), axis=0)
    head = np.repeat(headings, len(rotations)) + np.tile(rotations, n)
    groups = np.repeat(np.arange(n) * min(segments, n) // n, len(rotations))
    return pos, head, np.tile(rotations, n), groups


class LateralizedScorer:
    """One current image produces a continuous turn, without online scanning."""
    def __init__(self, world, encoder, memory, encode, gain=180.):
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("Steering gain must be finite and positive")
        self.world, self.encoder, self.memory, self.encode = world, encoder, memory, encode
        self.gain = float(gain)
        self.decisions = []

    def steer(self, pos, heading):
        codes = self.encode(self.encoder, self.world.scan(pos, np.array([heading])))
        output = self.memory(codes)
        signal = float(output["turn_signal"][0])
        turn = float(np.clip(signal * self.gain, -60, 60))
        self.decisions.append({"turn_signal": signal, "command_deg": turn,
                               "novelty": output["novelty"][0].tolist(),
                               "segment": int(output["segment"][0])})
        return turn, bool(output["no_evidence"][0])

    def __call__(self, pos, headings):
        # Compatibility with the historical offline discrete-scan evaluator.
        centre = float(headings[len(headings) // 2])
        turn, silent = self.steer(pos, centre)
        if silent:
            return np.zeros(len(headings))
        return (headings - centre - turn) ** 2
