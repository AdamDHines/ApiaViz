"""ApiaViz-only UV extension: fixed features and wiring, learned view memory.

An additive third stream preserves the existing 8,000 visible KCs and adds
2,000 UV KCs by default. This is not a resource-matched RGB/UV comparison.
No Sobel/Ardin dispatch or implicit RGB conversion is provided.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from apiaviz.src.uv import UVVisionBackbone
from apiaviz.nav.retino_kc import RetinotopicKCProjection, adaptive_avg_pool2d_anysize
from .latency import LatencyConfig, latency_race


@dataclass(frozen=True)
class UVEncoderConfig:
    schema: str = 'apiaviz-uv-v1'
    visible_code_dim: int = 8000
    uv_code_dim: int = 2000
    seed: int = 19
    pool_hw: tuple = (8,64)
    fan_in: int = 10
    sparsity: float = .02825
    radiance_scale: float = 1.
    response_half: float | None = None
    uv_enabled: bool = True

    def __post_init__(self):
        if self.schema not in ('apiaviz-uv-v1','apiaviz-uv-v2'): raise ValueError('Unknown UV encoder schema')
        if self.schema=='apiaviz-uv-v1' and self.response_half is not None:
            raise ValueError('Bounded receptor response requires v2')
        if self.schema=='apiaviz-uv-v2' and (self.response_half is None or not math.isfinite(self.response_half) or self.response_half<=0):
            raise ValueError('v2 requires a positive fixed half-response')
        if (self.visible_code_dim<2 or self.visible_code_dim%2 or self.uv_code_dim<1
                or int(self.uv_code_dim)!=self.uv_code_dim or int(self.visible_code_dim)!=self.visible_code_dim):
            raise ValueError('Require positive integer populations and even visible population')
        if len(self.pool_hw)!=2 or any(n<1 or int(n)!=n for n in self.pool_hw):
            raise ValueError('Invalid retinotopic pool')
        if not math.isfinite(self.sparsity) or not 0<self.sparsity<=1:
            raise ValueError('Invalid sparsity')
        if not math.isfinite(self.radiance_scale) or self.radiance_scale<=0:
            raise ValueError('Invalid radiance scale')

    @property
    def code_dim(self): return self.visible_code_dim+self.uv_code_dim


class ApiaVizUVEncoder(nn.Module):
    stream_names=('form','colour','uv')

    def __init__(self,config=None,circuit=None):
        super().__init__()
        self.config=config or UVEncoderConfig()
        self.circuit_config=circuit or LatencyConfig(current_gain=.1,time_bin_ms=1.,inhibition_delay_ms=1.)
        c=self.config
        self.backbone=UVVisionBackbone(c.radiance_scale,c.response_half)
        self.projections=nn.ModuleList([
            RetinotopicKCProjection(channels,pool_hw=c.pool_hw,code_dim=units,
                fan_in=c.fan_in,sparsity=c.sparsity,seed=c.seed+offset)
            for channels,units,offset in ((3,c.visible_code_dim//2,0),
                (2,c.visible_code_dim//2,16),(7,c.uv_code_dim,32))])
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def currents(self,receptors,*,uv_enabled=None):
        uv_enabled=self.config.uv_enabled if uv_enabled is None else uv_enabled
        maps=self.backbone(receptors)
        result=[]
        for name,projection in zip(self.stream_names,self.projections):
            flat=adaptive_avg_pool2d_anysize(maps[name],projection.pool_hw).flatten(1)
            flat=(flat-flat.mean(1,keepdim=True))/(flat.std(1,keepdim=True)+1e-6)
            drive=F.relu(flat@projection.connection)
            # Explicit pathway ablation. Zero UV input is a dark UV stimulus,
            # not an ablation: B-UV and G-UV would still carry visible signals.
            result.append(drive if name!='uv' or uv_enabled else torch.zeros_like(drive))
        return result

    @torch.no_grad()
    def diagnostics(self,receptors,*,uv_enabled=None):
        streams=[latency_race(drive,projection.active_units,self.circuit_config)
            for drive,projection in zip(self.currents(receptors,uv_enabled=uv_enabled),self.projections)]
        return dict(codes=torch.cat([F.normalize(s['codes'],dim=1) for s in streams],1),streams=streams)

    def forward(self,receptors,*,uv_enabled=None):
        return self.diagnostics(receptors,uv_enabled=uv_enabled)['codes']

    def metadata(self):
        return dict(encoder=asdict(self.config),circuit=asdict(self.circuit_config),
            channels=list(self.backbone.channels),streams=list(self.stream_names),
            uv_planes=list(self.backbone.uv_planes),total_kcs=self.config.code_dim,
            input='Linear relative UV/blue/green receptor radiances; no display transform',
            learning='Fixed analytical front end and seeded projection; downstream memory learns teaching views')
