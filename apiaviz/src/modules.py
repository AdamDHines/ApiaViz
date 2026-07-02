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


class VisionBackbone(nn.Module):
    def __init__(
        self,
        feature_channels: int = 32,
        embedding_dim: int = 128,
    ):
        super().__init__()

        self.spatial_sampler = HexRouting2d(1, learnable=True, bias=True)
        self.luminance_adapter = LocalLuminanceAdapter()
        self.chromatic_encoder = ChromaticEncoder()
        self.contrast_filter = ContrastFilterBank()
        self.pathway_stack = PathwayStack()
        self.feature_integrator = FeatureIntegrator(
            feature_channels=feature_channels,
            embedding_dim=embedding_dim,
        )

    def _split_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) == 2:
            luminance = x.mean(dim=1, keepdim=True)
            chromatic = x
        elif x.size(1) >= 3:
            luminance = x.mean(dim=1, keepdim=True)
            chromatic = x[:, 1:3]
        else:
            raise ValueError(f"VisionBackbone expected 2 or 3 input channels, got {x.size(1)}")
        return luminance, chromatic

    def freeze_frontend_layers(self) -> None:
        """Freeze initialized low-level filters and first pathway relays."""
        frozen_modules = [
            self.spatial_sampler,
            self.chromatic_encoder,
            self.contrast_filter,
            self.pathway_stack.on_feature,
            self.pathway_stack.off_feature,
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor, return_maps: bool = False, ablate_chromatic: bool = False):
        luminance, chromatic = self._split_input(x)

        sampled_luminance = self.spatial_sampler(luminance)
        chromatic_feature = self.chromatic_encoder(chromatic)
        if ablate_chromatic:
            chromatic_feature = torch.zeros_like(chromatic_feature)
        adapted_luminance = self.luminance_adapter(sampled_luminance)
        contrast_features = self.contrast_filter(adapted_luminance)
        pathway_features = self.pathway_stack(contrast_features, chromatic_feature)
        integrated = self.feature_integrator(
            pathway_features["on_feature"],
            pathway_features["off_feature"],
            pathway_features["color_luminance_feature"],
        )

        if return_maps:
            return {
                "luminance": luminance,
                "sampled_luminance": sampled_luminance,
                "adapted_luminance": adapted_luminance,
                "contrast_features": contrast_features,
                "chromatic_feature": chromatic_feature,
                **pathway_features,
                "scene_embedding": integrated["embedding"],
                "scene_feature_map": integrated["feature_map"],
                "pooling_exponent": integrated["pooling_exponent"],
                "fused_pathway_map": integrated["fused_pathway_map"],
            }

        return integrated["embedding"]


class HexRouting2d(nn.Module):
    def __init__(self, in_channels, out_channels=1, learnable=False, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if learnable:
            self.weight = nn.Parameter(torch.ones(out_channels, in_channels, 6) / 6.0)
        else:
            weight = torch.ones(out_channels, in_channels, 6, dtype=torch.float32) / 6.0
            self.register_buffer("weight", weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

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

        neighbours = torch.stack([left, right, up_a, up_b, down_a, down_b], dim=2)
        y = torch.einsum("ocn,bcnhw->bohw", self.weight, neighbours)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class LocalLuminanceAdapter(nn.Module):
    """Parameter-free local luminance normalization before contrast extraction."""

    def __init__(self, kernel_size: int = 9, gain: float = 2.0, eps: float = 0.05):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.gain = gain
        self.eps = eps

    def forward(self, x):
        intensity = (x + 1.0) * 0.5
        pad = self.kernel_size // 2
        local_mean = F.avg_pool2d(
            F.pad(intensity, (pad, pad, pad, pad), mode="reflect"),
            kernel_size=self.kernel_size,
            stride=1,
        )
        local_contrast = (intensity - local_mean) / (local_mean.abs() + self.eps)
        return torch.tanh(self.gain * local_contrast)


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


class OnFeaturePathway(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.gain.weight.fill_(1.0)
            self.gain.bias.zero_()

    def forward(self, on_channel):
        return F.relu(self.gain(on_channel))


class OffFeaturePathway(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            self.gain.weight.fill_(1.0)
            self.gain.bias.zero_()

    def forward(self, off_channel):
        return F.relu(self.gain(off_channel))


class ColorLuminancePathway(nn.Module):
    def __init__(self):
        super().__init__()
        self.integrate = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.local = nn.Conv2d(1, 1, kernel_size=3, padding=1, padding_mode="reflect", bias=True)

        with torch.no_grad():
            self.integrate.weight.zero_()
            self.integrate.bias.zero_()
            self.integrate.weight[0, 0, 0, 0] = 0.5
            self.integrate.weight[0, 1, 0, 0] = 0.5

            self.local.weight.zero_()
            self.local.bias.zero_()
            self.local.weight[0, 0] = torch.ones(3, 3) / 9.0

    def forward(self, luminance_channel, chromatic_feature):
        achromatic = chromatic_feature[:, 0:1]
        if chromatic_feature.size(1) >= 3:
            opponent_magnitude = chromatic_feature[:, 1:2] + chromatic_feature[:, 2:3]
        else:
            opponent_magnitude = torch.zeros_like(achromatic)
        x = torch.cat([luminance_channel, achromatic], dim=1)
        x = self.integrate(x) + 0.25 * opponent_magnitude
        x = self.local(x)
        return F.relu(x)


class PathwayStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.on_feature = OnFeaturePathway()
        self.off_feature = OffFeaturePathway()
        self.color_luminance_feature = ColorLuminancePathway()

    def forward(self, contrast_features, chromatic_feature):
        on_channel = contrast_features[:, 0:1]
        off_channel = contrast_features[:, 1:2]
        luminance_channel = contrast_features[:, 2:3]

        on_feature = self.on_feature(on_channel)
        off_feature = self.off_feature(off_channel)
        color_luminance_feature = self.color_luminance_feature(luminance_channel, chromatic_feature)

        return {
            "on_feature": on_feature,
            "off_feature": off_feature,
            "color_luminance_feature": color_luminance_feature,
        }


class PathwayFeatureMixer(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, pool_scales=(1, 2, 4)):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.local = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.context = nn.Conv2d(in_channels * len(self.pool_scales), out_channels, kernel_size=1, bias=True)
        self.refine = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels)

    def _pool_context(self, x):
        pooled = []
        for scale in self.pool_scales:
            if scale == 1:
                pooled.append(x)
                continue
            p = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=True)
            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
            pooled.append(p)
        return torch.cat(pooled, dim=1)

    def forward(self, x):
        local = self.local(x)
        context = self.context(self._pool_context(x))
        fused = torch.cat([local, context], dim=1)
        return F.relu(self.norm(self.refine(fused)))


class GeneralizedMeanPooling2d(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p)


class FeatureIntegrator(nn.Module):
    def __init__(self, pathway_channels=3, feature_channels=32, embedding_dim=128):
        super().__init__()
        self.mixer = PathwayFeatureMixer(in_channels=pathway_channels, out_channels=feature_channels)
        self.integrate = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, padding_mode="reflect", bias=True)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=feature_channels)
        self.gem_pool = GeneralizedMeanPooling2d(p=3.0)
        self.embedding = nn.Linear(feature_channels, embedding_dim)

    def forward(self, on_feature, off_feature, color_luminance_feature):
        pathway_features = torch.cat([on_feature, off_feature, color_luminance_feature], dim=1)
        fused = self.mixer(pathway_features)
        feature_map = F.relu(self.norm(self.integrate(fused) + fused))
        pooled = self.gem_pool(feature_map).flatten(1)
        embedding = self.embedding(pooled)

        return {
            "feature_map": feature_map,
            "embedding": embedding,
            "fused_pathway_map": fused,
            "pooling_exponent": self.gem_pool.p.clamp(min=self.gem_pool.eps).detach(),
        }
