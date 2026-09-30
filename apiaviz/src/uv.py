"""Explicit UV/blue/green ApiaViz front end; historical RGB models are unchanged.

The added UV filters are analytical engineering choices, not fitted physiology.
Inputs are linear relative receptor radiances, never display false colours.
"""
import math

import torch
from torch import nn

from .modules import load_vision_backbone, HexRouting2d, LocalLuminanceAdapter, ContrastFilterBank


class UVVisionBackbone(nn.Module):
    channels = ('uv', 'blue', 'green')
    uv_planes = ('uv_on', 'uv_off', 'uv_lowpass', 'uv_minus_blue',
                 'blue_minus_uv', 'uv_minus_green', 'green_minus_uv')

    def __init__(self, radiance_scale=1.):
        super().__init__()
        if not math.isfinite(radiance_scale) or radiance_scale<=0:
            raise ValueError('Require a finite positive common radiance scale')
        self.register_buffer('radiance_scale',torch.tensor(float(radiance_scale)))
        self.visible=load_vision_backbone()
        self.uv_sampler=HexRouting2d(channels=1)
        self.uv_adapter=LocalLuminanceAdapter()
        self.uv_contrast=ContrastFilterBank()
        self.requires_grad_(False)
        self.eval()

    def forward(self, receptors):
        if (receptors.ndim!=4 or receptors.shape[1]!=3 or min(receptors.shape[-2:])<5
                or not receptors.is_floating_point() or not torch.isfinite(receptors).all()
                or (receptors<0).any()):
            raise ValueError('Expected finite nonnegative linear UV/blue/green [N,3,H,W], H,W >= 5')
        # One frozen scale for every channel/view. No clipping, gamma, independent
        # channel normalization or statistics fitted on test views.
        x=receptors/self.radiance_scale
        gb=x[:,[2,1]]
        visible=self.visible(gb*2-1,return_maps=True)
        sampled_gb=self.visible.spatial_sampler(gb*2-1)
        difference=.5*(sampled_gb[:,:1]-sampled_gb[:,1:])
        colour=torch.cat([difference.relu(),(-difference).relu()],1)
        sampled_uv=self.uv_sampler(x[:,:1]*2-1)
        contrast=self.uv_contrast(self.uv_adapter(sampled_uv))
        uv_blue=.5*(sampled_uv-sampled_gb[:,1:])
        uv_green=.5*(sampled_uv-sampled_gb[:,:1])
        uv=torch.cat([contrast,uv_blue.relu(),(-uv_blue).relu(),
                      uv_green.relu(),(-uv_green).relu()],1)
        return dict(form=visible['contrast_features'],colour=colour,uv=uv)
