"""Explicit UV v3 numerical control; historical v1/v2 encoders are unchanged.

Subtract a common reference BEFORE averaging in local light adaptation. This is
algebraically the same filter, but avoids turning floating-point DC cancellation
error into contrast and then amplifying it in stream standardization.
"""
import torch
from torch.nn import functional as F
from apiaviz.src.modules import LocalLuminanceAdapter
from .uv_encoder import ApiaVizUVEncoder, UVEncoderConfig


class StableLuminanceAdapter(LocalLuminanceAdapter):
    def forward(self, x):
        intensity = (x+1.)*.5
        luminance = intensity.mean(dim=1, keepdim=True)
        # A numerical reference, not a fitted exposure or a channel gain.
        reference = luminance[:, :, :1, :1]
        pad = self.kernel_size//2
        local_centered = F.avg_pool2d(F.pad(luminance-reference,
            (pad,pad,pad,pad), mode='reflect'), self.kernel_size, stride=1)
        centered = (intensity-reference)-local_centered
        local_lum = local_centered+reference
        adapted = centered/(local_lum.abs()+self.eps) if self.divisive else centered
        return torch.tanh(self.gain*adapted)


class UVPrecisionEncoder(ApiaVizUVEncoder):
    """Same response, filters, wiring, populations and learning; stable arithmetic."""
    def __init__(self, config=None, circuit=None):
        super().__init__(config or UVEncoderConfig(schema='apiaviz-uv-v2',response_half=1.), circuit)
        self.backbone.visible.luminance_adapter = StableLuminanceAdapter()
        self.backbone.uv_adapter = StableLuminanceAdapter()
        self.register_buffer('numerical_revision',torch.tensor(3))

    def metadata(self):
        return dict(super().metadata(), implementation='apiaviz-uv-precision-v3',
            numerical_change='Common-reference subtraction before local averaging; constant neutral fields have exactly zero contrast.')
