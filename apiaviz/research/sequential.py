"""Ablation of a local sequence gate over one-shot mushroom-body memories.

The gate follows the order of acquisition, not world coordinates. This discrete
gate is an algorithmic hypothesis for recurrent contextual selection, not yet
a simulated neural circuit.
"""
import torch
from torch import nn
from apiaviz.nav.retino_kc import MBONPopulation


class SequentialMemory(nn.Module):
    def __init__(self, codes, segments=80, ahead=3, behind=1):
        super().__init__()
        if ahead < 0 or behind < 0:
            raise ValueError("Sequence windows must be nonnegative")
        population = MBONPopulation(codes, segments)
        self.register_buffer("weights", torch.stack([m.weight for m in population.mbons]))
        self.ahead, self.behind = int(ahead), int(behind)
        self.reset()

    def reset(self):
        self.cursor = 0
        self.pending = None

    @torch.no_grad()
    def forward(self, codes):
        activity = (codes > 0).float()
        total = activity.sum(1)
        lower = max(0, self.cursor - self.behind)
        upper = min(len(self.weights), self.cursor + self.ahead + 1)
        novelty = activity @ self.weights[lower:upper].T / total[:, None].clamp_min(1)
        values, indices = novelty.min(1)
        self.pending = indices + lower
        return torch.where(total > 0, values, torch.ones_like(values))

    def commit(self, winner):
        if self.pending is None:
            raise RuntimeError("Score candidates before committing a decision")
        self.cursor = max(self.cursor, int(self.pending[winner]))
        self.pending = None
