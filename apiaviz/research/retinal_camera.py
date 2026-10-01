"""Versioned solid-angle retinal integration before any receptor response.

Piecewise-constant panorama pixels are integrated over nonoverlapping angular
receptor cells. This is a declared box acceptance model, not measured eye optics.
Both spectral and visible paths use identical weights; raw arrays stay linear.
"""
from functools import lru_cache
import numpy as np
import torch
from .dual_camera import DualCamera, visible_image
from .spectral_input import validate_arrays


@lru_cache(maxsize=64)
def _vertical(height, high, low, count):
    centres=np.linspace(60.,-15.,count);half=75/(count-1)/2
    if centres[0]+half>high or centres[-1]-half<low:
        raise ValueError('Panorama must cover complete retinal footprints')
    edges=np.linspace(high,low,height+1)
    upper=np.minimum(centres[:,None]+half,edges[None,:-1])
    lower=np.maximum(centres[:,None]-half,edges[None,1:])
    weights=np.where(upper>lower,np.sin(np.deg2rad(upper))-np.sin(np.deg2rad(lower)),0.)
    return weights/weights.sum(1,keepdims=True)


def integrate_receptors(raw,headings,*,elevation=(90.,-20.),shape=(51,199)):
    validate_arrays(raw)
    headings=np.asarray(headings,dtype=float)
    if headings.ndim!=1 or not len(headings) or not np.isfinite(headings).all():raise ValueError('Finite headings required')
    if len(shape)!=2 or any(int(n)!=n or n<5 for n in shape):raise ValueError('Invalid retinal shape')
    high,low=map(float,elevation)
    if not np.isfinite([high,low]).all() or not -90<=low<high<=90:raise ValueError('Invalid elevation bounds')
    h,w=raw.shape[:2];rh,rw=shape
    vertical=_vertical(h,high,low,rh)
    # Integrate elevation first. Small cached angular weights; no geometry.
    rows=np.einsum('rh,hwc->rwc',vertical,raw,optimize=True)
    centres=headings[:,None]+np.linspace(-148.,148.,rw)[None]
    x=(180.-centres)%360/360*w
    half=(296/(rw-1))/360*w/2
    lo=x-half;hi=x+half
    # A periodic antiderivative gives exact box integrals without dense matrices.
    prefix=np.concatenate([np.zeros((rh,1,3)),np.cumsum(rows,axis=1)],axis=1)
    def primitive(t):
        cycles=np.floor(t/w).astype(int);fraction=t-cycles*w
        i=np.floor(fraction).astype(int);alpha=fraction-i
        return (prefix[:,i,:]+alpha[None,:,:,None]*rows[:,i,:]
                +cycles[None,:,:,None]*prefix[:,-1,None,None,:])
    result=(primitive(hi)-primitive(lo))/(2*half)
    return torch.from_numpy(np.ascontiguousarray(result.transpose(1,3,0,2),dtype=np.float32))


class IntegratedReceptorCamera(DualCamera):
    schema='solid-angle-box-before-response-v1'

    def scan(self,position,headings,*,uv=False):
        raw=self.frame(position)
        linear=integrate_receptors(raw[:,:,:3] if uv else raw[:,:,3:],headings,
            elevation=self.config['elevation_deg'],shape=tuple(self.config['retina_shape']))
        return linear if uv else visible_image(linear,self.config['visible_white'])
