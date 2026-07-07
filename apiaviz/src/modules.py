"""Core ApiaViz vision backbone modules.

The active code uses computational names for the model stages. Older backbone
checkpoints used previous attribute names; ``VisionBackbone.load_state_dict``
remaps those keys before strict loading.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_vision_backbone(device="cpu", logger=None):
    """Load a frozen ``VisionBackbone`` for inference.

    The front end is **training-free**: a random-initialised, biologically-structured backbone works as
    well as the released checkpoint (the learning ablation shows trained ~= random-init). So if
    ``untrained`` is set, or no checkpoint exists at ``model_path``, the random-initialised backbone is
    used and the whole pipeline runs with **no trained model at all**. Either way the backbone is frozen.
    """
    def _log(message):
        (logger.info if logger is not None else print)(message)

    backbone = VisionBackbone().to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone


# Ablation targets: each removes exactly one computational stage (bypassed to an identity
# that preserves the tensor contract). Threaded via the ``ablate`` attribute so every call
# site respects it without rewiring; "none" runs the full pipeline.
ABLATIONS = ("none", "hex", "adapt", "opponency", "dog")


class VisionBackbone(nn.Module):
    """Insect-inspired vision front end shared by ALL downstream tasks.

    A single pipeline feeds every task so that each computational stage is upstream of every
    readout (navigation, flower choice). Both photoreceptor channels pass through the shared
    ommatidial (hex) sampling; the streams then diverge into a luminance form pathway (light
    adaptation -> DoG center-surround) and a spectral-opponency colour pathway (a photoreceptor
    difference, deliberately not luminance-normalized so it stays invariant to isoluminant
    corruption). The two are concatenated into one retinotopic *view code*::

        input (G,B) -> hex sample ->  light adapt -> DoG contrast (form)  ┐
                                  ->  spectral opponency (G-B / B-G)      ┘ -> [contrast|chroma]

    The unified 6-channel view is what the Kenyon-cell projection taps for every task, so an
    ablation of any stage (``self.ablate``) produces a measurable effect on all of them.
    """

    def __init__(self):
        super().__init__()

        self.spatial_sampler = HexRouting2d(channels=2, learnable=True, bias=True)
        self.luminance_adapter = LocalLuminanceAdapter()                 # divisive Weber gain (form path)
        self.chromatic_adapter = LocalLuminanceAdapter(divisive=False)   # subtractive only (colour path)
        self.chromatic_encoder = ChromaticEncoder()
        self.contrast_filter = ContrastFilterBank()
        # Which stage (if any) to bypass; set once (e.g. by the eval harness) and honoured by
        # every forward call. See ``ABLATIONS``.
        self.ablate = "none"

    def _split_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return the two photoreceptor channels (G, B) as [N, 2, H, W]."""
        if x.size(1) == 2:
            return x
        if x.size(1) >= 3:
            return x[:, 1:3]
        raise ValueError(f"VisionBackbone expected 2 or 3 input channels, got {x.size(1)}")

    def freeze_frontend_layers(self) -> None:
        """Freeze the initialized low-level filters."""
        frozen_modules = [
            self.spatial_sampler,
            self.chromatic_encoder,
            self.contrast_filter,
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor, return_maps: bool = False, ablate: str | None = None):
        ablate = self.ablate if ablate is None else ablate
        if ablate not in ABLATIONS:
            raise ValueError(f"ablate must be one of {ABLATIONS}, got {ablate!r}")

        chromatic_in = self._split_input(x)                       # [N, 2, H, W] raw (G, B)
        luminance = chromatic_in.mean(dim=1, keepdim=True)        # raw luminance (pre front end)

        # --- shared front end: ommatidial (hex) sampling on BOTH photoreceptor channels ---
        sampled = chromatic_in if ablate == "hex" else self.spatial_sampler(chromatic_in)
        sampled_luminance = sampled.mean(dim=1, keepdim=True)     # hex is linear: = hex(mean)

        # --- diverge: spatial-contrast (DoG) form pathway (luminance channel, light-adapted) ---
        # Light adaptation is a luminance-channel gain control. It precedes the center-surround so an
        # "adapt" ablation still reaches every downstream task through the form sub-code.
        adapted_luminance = sampled_luminance if ablate == "adapt" else self.luminance_adapter(sampled_luminance)
        if ablate == "dog":  # bypass the center-surround: pointwise ON/OFF rectification only
            contrast_features = torch.cat(
                [F.relu(adapted_luminance), F.relu(-adapted_luminance), adapted_luminance], dim=1)
        else:
            contrast_features = self.contrast_filter(adapted_luminance)

        # --- diverge: spectral-opponency colour pathway (subtractive light adaptation) ---
        # The colour path is light-adapted too (so an "adapt" ablation reaches it), but only
        # subtractively: the opponent difference (G-B) cancels the shared local-luminance baseline and
        # so stays invariant to the isoluminant luminance corruption the navigation code must survive.
        adapted_chromatic = sampled if ablate == "adapt" else self.chromatic_adapter(sampled)
        chromatic_feature = self.chromatic_encoder(adapted_chromatic)  # [achromatic, G-B, B-G]
        if ablate == "opponency":  # remove colour opponency, keep the achromatic sum
            chromatic_feature = torch.cat(
                [chromatic_feature[:, 0:1], torch.zeros_like(chromatic_feature[:, 1:3])], dim=1)

        if return_maps:
            return {
                "luminance": luminance,
                "sampled_luminance": sampled_luminance,
                "adapted_luminance": adapted_luminance,
                "contrast_features": contrast_features,
                "chromatic_feature": chromatic_feature,
            }

        # Unified retinotopic view code [N, 6, H, W] = [ON, OFF, luminance | achromatic, G-B, B-G].
        # Every task's Kenyon-cell projection taps this, so all stages are upstream of all tasks.
        return torch.cat([contrast_features, chromatic_feature], dim=1)


class HexRouting2d(nn.Module):
    """Depthwise hexagonal (ommatidial) resampling: each channel is averaged over its six
    hex neighbours with per-channel weights (initialised to a uniform 1/6). Applied to all
    photoreceptor channels so colour and luminance share the same ommatidial front end.
    """

    def __init__(self, channels=1, learnable=False, bias=False):
        super().__init__()
        self.channels = channels

        if learnable:
            self.weight = nn.Parameter(torch.ones(channels, 6) / 6.0)
        else:
            self.register_buffer("weight", torch.ones(channels, 6, dtype=torch.float32) / 6.0)

        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None

    def forward(self, x):
        batch, channels, height, width = x.shape
        padded = F.pad(x, (1, 1, 1, 1), mode="reflect")

        left = padded[:, :, 1:height + 1, 0:width]
        right = padded[:, :, 1:height + 1, 2:width + 2]

        up_left = padded[:, :, 0:height, 0:width]
        up_mid = padded[:, :, 0:height, 1:width + 1]
        up_right = padded[:, :, 0:height, 2:width + 2]

        down_left = padded[:, :, 2:height + 2, 0:width]
        down_mid = padded[:, :, 2:height + 2, 1:width + 1]
        down_right = padded[:, :, 2:height + 2, 2:width + 2]

        row_idx = torch.arange(height, device=x.device).view(1, 1, height, 1)
        even_mask = (row_idx % 2 == 0).to(x.dtype)
        odd_mask = 1.0 - even_mask

        up_a = even_mask * up_mid + odd_mask * up_left
        up_b = even_mask * up_right + odd_mask * up_mid
        down_a = even_mask * down_mid + odd_mask * down_left
        down_b = even_mask * down_right + odd_mask * down_mid

        neighbours = torch.stack([left, right, up_a, up_b, down_a, down_b], dim=2)  # [B,C,6,H,W]
        y = torch.einsum("cn,bcnhw->bchw", self.weight, neighbours)                 # depthwise

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class LocalLuminanceAdapter(nn.Module):
    """Parameter-free local light-adaptation (luminance gain control) shared across receptor channels.

    Photoreceptor adaptation is a **shared luminance** gain: every channel is baselined and divided by
    the same local luminance (the mean across channels), not by its own channel mean. This is what gives
    colour constancy -- a per-channel divisive normalization would subtract each channel's own local DC
    and annihilate the opponent signal ``G-B`` on uniform isoluminant patches (the landmark-colour cue
    the navigation opponent-colour code depends on). With a shared denominator the opponent difference
    survives (both channels share the subtracted baseline and the gain), so the stage can sit upstream of
    both the luminance-contrast and the spectral-opponency pathways. For a single-channel (luminance)
    input the shared mean is the channel itself, so the luminance pathway is unchanged.
    """

    def __init__(self, kernel_size: int = 9, gain: float = 2.0, eps: float = 0.05, divisive: bool = True):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.gain = gain
        self.eps = eps
        # ``divisive`` = Weber-law gain control (divide by local luminance): the standard light
        # adaptation for the luminance/form pathway. The colour pathway uses ``divisive=False``
        # (subtractive only): dividing the opponent channels by the corruptible local luminance
        # would destroy their invariance to isoluminant corruption, so there we only remove the
        # shared local DC (which cancels in ``G-B`` and keeps the opponent cue corruption-invariant).
        self.divisive = bool(divisive)

    def forward(self, x):
        intensity = (x + 1.0) * 0.5
        luminance = intensity.mean(dim=1, keepdim=True)          # shared local-luminance basis
        pad = self.kernel_size // 2
        local_lum = F.avg_pool2d(
            F.pad(luminance, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=self.kernel_size,
            stride=1,
        )
        centered = intensity - local_lum
        adapted = centered / (local_lum.abs() + self.eps) if self.divisive else centered
        return torch.tanh(self.gain * adapted)


class ChromaticEncoder(nn.Module):
    """Encode achromatic sum plus rectified green/blue opponent channels."""

    def __init__(self):
        super().__init__()
        self.spectral = nn.Conv2d(2, 1, kernel_size=1, bias=True)

        with torch.no_grad():
            self.spectral.weight.zero_()
            self.spectral.bias.zero_()
            self.spectral.weight[0, 0, 0, 0] = 0.5
            self.spectral.weight[0, 1, 0, 0] = 0.5

    def forward(self, chromatic):
        achromatic = F.relu(self.spectral(chromatic))
        green = chromatic[:, 0:1]
        blue = chromatic[:, 1:2]
        green_blue = F.relu(green - blue)
        blue_green = F.relu(blue - green)
        return torch.cat([achromatic, green_blue, blue_green], dim=1)


class ContrastFilterBank(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")

        self.filters = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            bias=True,
        )
        self._init_filters(kernel_size)

    def _init_filters(self, kernel_size):
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radius_sq = xx ** 2 + yy ** 2

        sigma_center, sigma_surround = 1.0, 2.5
        center = torch.exp(-radius_sq / (2 * sigma_center ** 2))
        center = center / center.sum()
        surround = torch.exp(-radius_sq / (2 * sigma_surround ** 2))
        surround = surround / surround.sum()

        dog = center - surround
        on_kernel = dog / (dog.norm() + 1e-8)
        off_kernel = -on_kernel.clone()
        low_pass = center / (center.norm() + 1e-8)

        with torch.no_grad():
            self.filters.weight.zero_()
            self.filters.bias.zero_()
            self.filters.weight[0, 0] = on_kernel
            self.filters.weight[1, 0] = off_kernel
            self.filters.weight[2, 0] = low_pass

    def forward(self, x):
        y = self.filters(x)
        on_channel = F.relu(y[:, 0:1])
        off_channel = F.relu(y[:, 1:2])
        luminance_channel = y[:, 2:3]
        return torch.cat([on_channel, off_channel, luminance_channel], dim=1)
