"""Opt-in graded adaptation/opponency controls, not calibrated photoreceptor physiology.

Retinal mean irradiance drives a causal, receptor-class gain state. This is a
broad-field gain approximation; no route, pose, labels, or future images enter.
All time constants and opponent coefficients are declared engineering choices.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .angular_frontend import AngularEncoder, spatial_filter
from .uv_encoder import ApiaVizUVEncoder, UVEncoderConfig
from .latency import latency_race
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize


@dataclass(frozen=True)
class AdaptationConfig:
    gain_tau_s: float = 1.
    integration_tau_s: float = .02
    dark_floor: float = 1e-4

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.gain_tau_s,self.integration_tau_s,self.dark_floor)):
            raise ValueError('Finite positive adaptation constants required')


@dataclass
class ReceptorState:
    irradiance: torch.Tensor
    background: torch.Tensor


def check_input(x):
    if x.ndim!=4 or x.shape[1]!=3 or not x.is_floating_point() or not torch.isfinite(x).all() or (x<0).any():
        raise ValueError('Finite nonnegative [batch, UV/B/G, height, width] required')


def initialize(x):
    """Explicit equilibrated pre-exposure; independent state for each batch row."""
    check_input(x)
    return ReceptorState(x.clone(),x.mean((2,3),keepdim=True))


def advance(x,state,dt,config=AdaptationConfig()):
    """Exact exponential updates for two linear states under held input.

    Response is a static ratio of fast irradiance and slow background states;
    neither this ratio nor the states are claimed to be measured volts.
    """
    check_input(x)
    if not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite elapsed time required')
    if state.irradiance.shape!=x.shape or state.background.shape!=(*x.shape[:2],1,1):
        raise ValueError('State and input shapes differ')
    if state.irradiance.dtype!=x.dtype or state.irradiance.device!=x.device:
        raise ValueError('State dtype/device mismatch')
    if not torch.isfinite(state.irradiance).all() or not torch.isfinite(state.background).all() or (state.irradiance<0).any() or (state.background<0).any():
        raise ValueError('Invalid adaptation state')
    z=x+(state.irradiance-x)*math.exp(-dt/config.integration_tau_s)
    mean=x.mean((2,3),keepdim=True)
    a=mean+(state.background-mean)*math.exp(-dt/config.gain_tau_s)
    return z/(z+a.clamp_min(config.dark_floor)),ReceptorState(z,a)


def response_sequence(images,initial,dt,*,adaptive,config=AdaptationConfig()):
    """T x batch x 3 x H x W chronological observations, no state resets per view."""
    if images.ndim!=5:raise ValueError('Sequence must be T x batch x 3 x H x W')
    state=initialize(initial);responses=[]
    for x in images:
        if adaptive:y,state=advance(x,state,dt,config)
        else:
            check_input(x);y=x/(x+1.)
        responses.append(y)
    return torch.stack(responses),state


def opponent_maps(sampled,mode):
    u,b,g=sampled.split(1,1)
    if mode=='pairwise':a,c=u-b,u-g
    elif mode=='combined':a,c=u-.5*(b+g),b-g
    else:raise ValueError('Unknown opponent axes')
    return torch.cat([a.relu(),(-a).relu(),c.relu(),(-c).relu()],1)


class GradedUVEncoder(ApiaVizUVEncoder):
    """Shared angular front end; caller supplies bounded graded receptor responses.

    Same 4000/4000/2000 projection matrices and spike circuit as current UV.
    All candidates share stable DC subtraction and angular spatial footprints.
    """
    def __init__(self,seed=19):
        super().__init__(UVEncoderConfig(schema='apiaviz-uv-v2',response_half=1.,seed=seed))
        self.angular=AngularEncoder('linear_colour',seed=seed)
        for a,b in zip(self.projections,self.angular.features.legacy):
            assert torch.equal(a.connection,b.connection)

    @torch.no_grad()
    def pooled(self,response,mode):
        check_input(response)
        if (response>1).any():raise ValueError('Bounded receptor responses required')
        k=self.angular.kernels(response)
        # Sample each receptor with the same historical angular hex footprint.
        x=response*2-1
        sampled=torch.where(k['even'],*[spatial_filter(x,h[:1].repeat(3,1,1,1),groups=3) for h in k['hex']])
        bounded=(sampled+1)*.5
        def contrast(lum):
            reference=lum[:,:,:1,:1]
            local_centered=spatial_filter(lum-reference,k['local'])
            local=local_centered+reference
            adapted=torch.tanh(2*((lum-reference)-local_centered)/(local.abs()+.05))
            value=spatial_filter(adapted,k['contrast'])
            return torch.cat([value[:,:2].relu(),value[:,2:]],1)
        form=contrast(bounded[:,1:].mean(1,keepdim=True))
        diff=bounded[:,2:3]-bounded[:,1:2]
        colour=torch.cat([diff.relu(),(-diff).relu()],1)
        uv=torch.cat([contrast(bounded[:,:1]),opponent_maps(bounded,mode)],1)
        return [adaptive_avg_pool2d_anysize(v,(8,64)) for v in (form,colour,uv)]

    @torch.no_grad()
    def readouts(self,pooled):
        drives=[];bits=[]
        for x,p in zip(pooled,self.projections):
            flat=x.flatten(1)
            flat=(flat-flat.mean(1,keepdim=True))/(flat.std(1,keepdim=True)+1e-6)
            drive=F.relu(flat@p.connection)
            drives.append(F.normalize(drive,dim=1))
            bits.append((latency_race(drive,p.active_units,self.circuit_config)['codes']>0).float())
        normalized_bits=[F.normalize(b,dim=1) for b in bits]
        return dict(graded=F.normalize(torch.cat(drives,1),dim=1),
                    spikes=F.normalize(torch.cat(normalized_bits,1),dim=1),
                    production_spikes=F.normalize(torch.cat(bits,1),dim=1),
                    graded_no_uv=F.normalize(torch.cat(drives[:2],1),dim=1),
                    spikes_no_uv=F.normalize(torch.cat(normalized_bits[:2],1),dim=1),
                    production_spikes_no_uv=F.normalize(torch.cat(bits[:2],1),dim=1))
