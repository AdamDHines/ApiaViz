"""Retinotopic Kenyon-cell code: the generalisable sparse view code.

This is the Phase-1 core fix (see PLAN.md). It produces a sparse, high-dimensional
Kenyon-cell-like code from a single view while **preserving azimuth (heading)
structure** — the property the navigation rIDF needs and that the GeM-global
``scene_embedding`` throws away.

Design (deliberately Ardin/Webb-simple, no training of the code path):
  view -> frozen VisionBackbone front end -> early retinotopic ``contrast_features``
       -> azimuth-preserving pool (no global pooling)
       -> fixed random sparse PN->KC projection (fan-in ~10)
       -> k-WTA (~5% active) -> sparse code.

The same code is intended to drive multiple downstream tasks (navigation familiarity,
flower recognition) through cheap task-specific readouts. The encoder is a plain
``nn.Module`` with no learnable parameters in the KC path, so an SNN variant can later
reuse the identical fixed projection (Phase 3).

Validated on Ant1 routes 1-10 (n=10), heading offsets +-60 deg, default config below:
  far-offset same-place cosine ~0.27, memory margin ~0.57, selected-center fraction 1.0
  (vs current apiaviz scene_embedding: 0.998 / 0.000). Experiment:
  research/2026-06-27-retinotopic-kc/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from apiaviz.src.modules import VisionBackbone


# Default tap + pool chosen by the Phase-1 sweep. Width 64 is safe for both the
# mbant panoramic render (W=74) and 64x64 object images, so the fixed projection
# dimension is stable across tasks.
DEFAULT_TAP = "contrast_features"
DEFAULT_POOL_HW = (8, 64)
DEFAULT_CODE_DIM = 20000
DEFAULT_FAN_IN = 10
DEFAULT_SPARSITY = 0.05


def _tap_channels(tap: str) -> int:
    # contrast_features = [ON, OFF, luminance]; pathway/scene maps differ.
    return {
        "contrast_features": 3,
        "pathway": 3,
        "scene_feature_map": 32,
    }.get(tap, 3)


def _adaptive_avg_matrix(in_size: int, out_size: int, device, dtype) -> torch.Tensor:
    """Averaging matrix ``A[o, i]`` reproducing PyTorch adaptive-pool bins along one axis."""
    A = torch.zeros(out_size, in_size, device=device, dtype=dtype)
    for o in range(out_size):
        start = (o * in_size) // out_size
        end = -(-(o + 1) * in_size // out_size)  # ceil division
        A[o, start:end] = 1.0 / (end - start)
    return A


def adaptive_avg_pool2d_anysize(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """``F.adaptive_avg_pool2d`` that also works on MPS for non-divisible sizes.

    MPS only implements adaptive pooling when the input is divisible by the output
    (pytorch#96056). Adaptive average pooling is separable and linear, so for the
    general case we express it as two small fixed averaging matmuls, which every MPS
    kernel supports. Matches the native op to float32 precision; no CPU transfer.
    """
    Ho, Wo = int(out_hw[0]), int(out_hw[1])
    _, _, Hi, Wi = x.shape
    if Hi % Ho == 0 and Wi % Wo == 0:  # fast path: native kernel handles this
        return F.adaptive_avg_pool2d(x, (Ho, Wo))
    Ah = _adaptive_avg_matrix(Hi, Ho, x.device, x.dtype)
    Aw = _adaptive_avg_matrix(Wi, Wo, x.device, x.dtype)
    return torch.einsum("oi,ncij,pj->ncop", Ah, x, Aw)


class RetinotopicKCProjection(nn.Module):
    """Fixed random sparse PN->KC projection + k-WTA over an azimuth-preserving pool.

    No learnable parameters: the connection mask/weights and the WTA are fixed, so the
    code is reproducible from ``seed`` and identical between ANN and SNN variants.
    """

    def __init__(
        self,
        in_channels: int,
        pool_hw: tuple[int, int] = DEFAULT_POOL_HW,
        code_dim: int = DEFAULT_CODE_DIM,
        fan_in: int = DEFAULT_FAN_IN,
        sparsity: float = DEFAULT_SPARSITY,
        seed: int = 7,
        signed: bool = True,
        standardize: bool = True,
    ):
        super().__init__()
        self.pool_hw = (int(pool_hw[0]), int(pool_hw[1]))
        self.code_dim = int(code_dim)
        self.fan_in = int(fan_in)
        self.sparsity = float(sparsity)
        self.standardize = bool(standardize)
        pn_dim = int(in_channels) * self.pool_hw[0] * self.pool_hw[1]
        self.pn_dim = pn_dim
        if not 0 < self.fan_in <= pn_dim:
            raise ValueError(f"fan_in must be in [1, {pn_dim}], got {self.fan_in}")

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        connection = torch.zeros((pn_dim, self.code_dim), dtype=torch.float32)
        for unit in range(self.code_dim):
            idx = torch.randperm(pn_dim, generator=gen)[: self.fan_in]
            if signed:
                connection[idx, unit] = torch.randn(self.fan_in, generator=gen) / (self.fan_in ** 0.5)
            else:
                connection[idx, unit] = 1.0 / max(float(self.fan_in), 1.0)
        # Buffer (not Parameter): fixed, moves with .to(device), saved in state_dict.
        self.register_buffer("connection", connection)

    @property
    def active_units(self) -> int:
        return min(max(1, int(round(self.code_dim * self.sparsity))), self.code_dim)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.dim() != 4:
            raise ValueError(f"expected [N,C,H,W], got {tuple(feature_map.shape)}")
        pooled = adaptive_avg_pool2d_anysize(feature_map, self.pool_hw)
        flat = pooled.flatten(1)
        if flat.size(1) != self.pn_dim:
            raise ValueError(
                f"PN dim mismatch: got {flat.size(1)}, expected {self.pn_dim}. "
                "The input channel count or pool size differs from construction."
            )
        if self.standardize:
            flat = (flat - flat.mean(dim=1, keepdim=True)) / (flat.std(dim=1, keepdim=True) + 1e-6)
        drive = F.relu(flat @ self.connection)
        k = self.active_units
        values, indices = drive.topk(k, dim=1)
        codes = torch.zeros_like(drive)
        codes.scatter_(1, indices, values.clamp_min(0.0))
        return codes


class RetinotopicKCEncoder(nn.Module):
    """Full ``view -> sparse KC code`` module: frozen backbone front end + fixed projection.

    The backbone is frozen and only its early retinotopic ``tap`` map is used; the GeM
    ``scene_embedding`` is intentionally bypassed.
    """

    def __init__(
        self,
        backbone: VisionBackbone,
        tap: str = DEFAULT_TAP,
        pool_hw: tuple[int, int] = DEFAULT_POOL_HW,
        code_dim: int = DEFAULT_CODE_DIM,
        fan_in: int = DEFAULT_FAN_IN,
        sparsity: float = DEFAULT_SPARSITY,
        seed: int = 7,
        signed: bool = True,
        standardize: bool = True,
    ):
        super().__init__()
        self.tap = str(tap)
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.projection = RetinotopicKCProjection(
            in_channels=_tap_channels(self.tap),
            pool_hw=pool_hw, code_dim=code_dim, fan_in=fan_in,
            sparsity=sparsity, seed=seed, signed=signed, standardize=standardize,
        )

    def _tap_map(self, maps: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.tap == "pathway":
            return torch.cat(
                [maps["on_feature"], maps["off_feature"], maps["color_luminance_feature"]], dim=1
            )
        return maps[self.tap]

    @torch.no_grad()
    def forward(self, preprocessed_views: torch.Tensor) -> torch.Tensor:
        """preprocessed_views: [N, 2, H, W] normalised to [-1, 1] (see preprocess_apiaviz_torch)."""
        maps = self.backbone(preprocessed_views, return_maps=True)
        return self.projection(self._tap_map(maps))

    @classmethod
    def from_checkpoint(cls, backbone_path: str | Path, device: torch.device | str = "cpu", **kwargs: Any):
        checkpoint = torch.load(Path(backbone_path), map_location=device)
        state = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state = checkpoint[key]
                    break
        backbone = VisionBackbone().to(device)
        backbone.load_state_dict(state, strict=True)
        encoder = cls(backbone=backbone, **kwargs).to(device)
        encoder.eval()
        return encoder


class AntiHebbianMBON(nn.Module):
    """One-MBON familiarity readout (Ardin/Mangan/Webb 2016).

    KC->MBON synapses are depressed (anti-Hebbian LTD) for Kenyon cells active along the
    stored route. ``forward`` returns a per-view **novelty** score (lower = more familiar),
    so it is a drop-in replacement for ``CosineRouteMemory`` in the navigation scorers
    (the loop selects the heading with minimum novelty).

    Unlike a template bank, this is a single scalar readout neuron — the biologically
    faithful mushroom-body mechanism. Validated: novelty minimum sits at the correct heading
    at every route position (selected_center_fraction = 1.0). ``depression=0.5`` gives a
    smoother gradient toward the minimum than full depression; tune per navigation behaviour.

    ``graded=True`` keeps the **relative KC drive** (peak-normalized code magnitude) rather than a
    binary active mask. It was tried as a fix for the free-navigation drift but does **not** close
    the gap to the cosine readout (the limitation is structural: a single MBON collapses all route
    views into one weight vector, losing position-specificity). Default is the binary Ardin/Webb
    readout; ``CosineRouteMemory`` remains the champion nav readout (see PLAN.md D2).
    """

    def __init__(self, code_dim: int, depression: float = 0.5, w0: float = 1.0, graded: bool = False):
        super().__init__()
        if not 0.0 < depression <= 1.0:
            raise ValueError(f"depression must be in (0, 1], got {depression}")
        self.depression = float(depression)
        self.graded = bool(graded)
        self.register_buffer("weight", torch.full((int(code_dim),), float(w0)))

    def _activity(self, codes: torch.Tensor) -> torch.Tensor:
        """Per-view KC activity used by both store and forward."""
        if self.graded:
            pos = codes.clamp_min(0.0)
            return pos / pos.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return (codes > 0).float()

    @torch.no_grad()
    def store(self, route_codes: torch.Tensor) -> None:
        """One-shot anti-Hebbian pass over stored route view codes [num_views, code_dim]."""
        activity = self._activity(route_codes).to(self.weight.device)
        for view in activity:
            self.weight.mul_((1.0 - self.depression * view).clamp_min(0.0))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        activity = self._activity(codes)
        return (activity * self.weight).sum(dim=1) / activity.sum(dim=1).clamp_min(1e-6)


class RewardMBON(nn.Module):
    """Approach MBON with a reward-gated three-factor learning rule (KC activity x reward x rate).

    Unlike ``AntiHebbianMBON`` -- which measures *novelty* (have I seen this KC pattern) and only ever
    depresses -- this readout learns a *valence* boundary. A dopaminergic reward signal gates plasticity
    with a sign: appetitive (rewarded) views potentiate the approach synapses of their active KCs,
    aversive/unrewarded views depress them. Weights therefore converge to the difference of the rewarded
    and unrewarded KC-activity prototypes (a balanced linear discriminant), and ``forward`` returns the
    net **approach drive** ``a . w`` (higher = more likely to land). This is the associative mushroom-body
    rule used for foraging choice; the anti-Hebbian novelty neuron cannot separate rewarded from
    unrewarded flowers, only familiar from novel.

    One-shot / few-shot: ``store`` is a single balanced pass, so it doubles as the analytic limit of the
    reward-modulated STDP in ``mbant/network.py``. No learnable parameters (fixed once stored).
    """

    def __init__(self, code_dim: int, lr: float = 1.0, graded: bool = False):
        super().__init__()
        self.lr = float(lr)
        self.graded = bool(graded)
        self.register_buffer("weight", torch.zeros(int(code_dim)))

    def _activity(self, codes: torch.Tensor) -> torch.Tensor:
        if self.graded:
            pos = codes.clamp_min(0.0)
            return pos / pos.amax(dim=1, keepdim=True).clamp_min(1e-6)
        return (codes > 0).float()

    @torch.no_grad()
    def store(self, codes: torch.Tensor, rewards) -> None:
        """One balanced reward-gated pass: potentiate on rewarded views, depress on unrewarded ones."""
        activity = self._activity(codes).to(self.weight.device)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.weight.device) > 0.5
        zero = torch.zeros_like(self.weight)
        proto_rew = activity[r].mean(dim=0) if bool(r.any()) else zero
        proto_unrew = activity[~r].mean(dim=0) if bool((~r).any()) else zero
        self.weight.add_(self.lr * (proto_rew - proto_unrew))

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        activity = self._activity(codes)
        return (activity * self.weight).sum(dim=1) / activity.sum(dim=1).clamp_min(1e-6)


class MBONPopulation(nn.Module):
    """A small population of anti-Hebbian MBONs, each storing one route SEGMENT.

    A single ``AntiHebbianMBON`` collapses all route views into one weight vector, so it goes flat
    *between* stored nodes and drifts ~10x more than a template bank in navigation (PLAN.md D2). This
    is a readout-structure bottleneck, not an input one (colour does not fix the single MBON). The
    mushroom body has *many* MBONs (~34 in Drosophila), so a population -- each MBON responsible for a
    contiguous stretch of the route -- recovers the per-view specificity one neuron loses. Per-view
    novelty is the **most-familiar segment** (min across MBONs), the population analogue of the cosine
    bank's max-over-templates. Drop-in for ``CosineRouteMemory`` / ``AntiHebbianMBON`` (lower = familiar).

    ``n_mbons=1`` reduces to the single MBON; ``n_mbons=num_views`` approaches the cosine-bank limit.
    Validated (Ant1, front end frozen; research/2026-06-27-retinotopic-kc/run_{mbon,colour}_population.py):
    grayscale nav ``error_rate`` falls monotonically 0.12 (S=1) -> 0.036 (S=16) as the population grows;
    with the End Goal A opponent-colour code it homes 0.88 of nests under luminance corruption (n=8) --
    matching the cosine bank and beating CLAHE (0.62), where the single MBON reaches only 0.38.
    Performance saturates by ~16 MBONs; a small residual ``error_rate`` gap to the cosine dot-product is
    intrinsic to anti-Hebbian familiarity, not a capacity limit. No learnable parameters (fixed readout).
    """

    def __init__(self, route_codes: torch.Tensor, n_mbons: int, depression: float = 0.5,
                 partition: str = "contiguous", graded: bool = False):
        super().__init__()
        n_views, code_dim = route_codes.shape
        n_mbons = max(1, min(int(n_mbons), int(n_views)))
        index = torch.arange(n_views)
        if partition == "contiguous":
            segment = index * n_mbons // n_views          # 0..S-1 contiguous route segments
        elif partition == "strided":
            segment = index % n_mbons                      # interleaved views
        else:
            raise ValueError(f"partition must be 'contiguous' or 'strided', got {partition!r}")
        self.n_mbons = int(n_mbons)
        self.mbons = nn.ModuleList()
        for s in range(n_mbons):
            members = (segment == s).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                continue
            mbon = AntiHebbianMBON(code_dim=code_dim, depression=depression, graded=graded)
            mbon.store(route_codes[members])
            self.mbons.append(mbon)

    @torch.no_grad()
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        novelty = torch.stack([mbon(codes) for mbon in self.mbons], dim=0)  # [n_mbons, n_views]
        return novelty.min(dim=0).values
