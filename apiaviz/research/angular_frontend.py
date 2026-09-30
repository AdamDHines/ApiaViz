"""A filter-scale control for the input-resolution experiment.

The original pixel-based encoder remains unchanged. This optional control holds
its discrete spatial tap positions/weights fixed in visual degrees, evaluating
fractional offsets by bilinear interpolation. It is not a calibrated bee retina.
"""
import math

import torch
from torch.nn import functional as F

from .frontend_refinements import RefinementEncoder
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize

REFERENCE_SHAPE=(18,74)


def scaled_kernel(kernel, scale_y, scale_x):
    """Splat each original tap to its physical offset on a finer raster.

Equivalent to bilinear sampling at each scaled tap, including at a reflected
boundary; preserve signed kernel mass, not an arbitrary image-resize gain.
"""
    kh,kw=kernel.shape[-2:]
    if scale_y<1 or scale_x<1 or kh%2!=1 or kw%2!=1:
        raise ValueError('Expected odd kernels and a grid no coarser than the reference')
    if scale_y==scale_x==1:
        return kernel
    ry,rx=math.ceil(kh//2*scale_y),math.ceil(kw//2*scale_x)
    result=kernel.new_zeros((*kernel.shape[:-2],2*ry+1,2*rx+1))
    for y in range(kh):
        for x in range(kw):
            py=(y-kh//2)*scale_y+ry; px=(x-kw//2)*scale_x+rx
            iy,ix=math.floor(py),math.floor(px)
            fy,fx=py-iy,px-ix
            for dy,wy in ((0,1-fy),(1,fy)):
                for dx,wx in ((0,1-fx),(1,fx)):
                    if wy*wx:
                        result[...,iy+dy,ix+dx]+=kernel[...,y,x]*(wy*wx)
    return result


def spatial_filter(x,kernel,bias=None,groups=1):
    ry,rx=kernel.shape[-2]//2,kernel.shape[-1]//2
    return F.conv2d(F.pad(x,(rx,rx,ry,ry),mode='reflect'),kernel,bias,groups=groups)


class AngularEncoder(RefinementEncoder):
    def __init__(self,method='linear_colour',seed=19,code_dim=8000):
        super().__init__(method=method,seed=seed,code_dim=code_dim)
        if method not in ('linear_colour','sobel_colour'):
            raise ValueError('Angular filter control is defined for ApiaViz and Sobel')
        self._angular_cache={}

    def kernels(self,images):
        h,w=images.shape[-2:]
        sy,sx=(h-1)/17,(w-1)/73
        key=(h,w,images.dtype,images.device)
        if key in self._angular_cache:
            return self._angular_cache[key]
        backbone=self.features.backbone
        # The original staggered six-neighbour hex operation has two row phases.
        # Anchor these phases to the reference angular rows, rather than toggling
        # at the finer raster's row frequency.
        hexes=[]
        for offsets in [((0,-1),(0,1),(-1,0),(-1,1),(1,0),(1,1)),
                        ((0,-1),(0,1),(-1,-1),(-1,0),(1,-1),(1,0))]:
            k=images.new_zeros((2,1,3,3))
            for index,(y,x) in enumerate(offsets):
                k[:,0,y+1,x+1]=backbone.spatial_sampler.weight[:,index]
            hexes.append(scaled_kernel(k,sy,sx))
        k=backbone.luminance_adapter.kernel_size
        local=scaled_kernel(images.new_ones((1,1,k,k))/(k*k),sy,sx)
        contrast=scaled_kernel(backbone.contrast_filter.filters.weight,sy,sx)
        axis=torch.arange(-2,3,dtype=images.dtype,device=images.device)
        gaussian=torch.exp(-axis.square()/2)
        gaussian=gaussian[:,None]*gaussian[None,:]
        gaussian=scaled_kernel((gaussian/gaussian.sum())[None,None],sy,sx)
        sobel=images.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
        sobel=scaled_kernel(torch.stack([sobel,sobel.T])[:,None],sy,sx)
        even=(torch.floor(torch.arange(h,device=images.device)/sy+.5).long()%2==0).view(1,1,h,1)
        result=dict(hex=hexes,local=local,contrast=contrast,gaussian=gaussian,sobel=sobel,even=even)
        self._angular_cache[key]=result
        return result

    @torch.no_grad()
    def pooled_features(self,images):
        if tuple(images.shape[-2:])==REFERENCE_SHAPE:
            # Preserve exact floating point operations at the reference setting.
            return super().pooled_features(images)
        if images.ndim!=4 or images.shape[1] not in (2,3) or not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
            raise ValueError('Expected finite GB/RGB images in [0,1]')
        kernels=self.kernels(images)
        gb=images[:,-2:]
        backbone=self.features.backbone
        if self.method=='linear_colour':
            x=gb*2-1
            responses=[spatial_filter(x,k,backbone.spatial_sampler.bias,groups=2) for k in kernels['hex']]
            sampled=torch.where(kernels['even'],*responses)
            luminance=(sampled.mean(1,keepdim=True)+1)*.5
            local=spatial_filter(luminance,kernels['local'])
            adapter=backbone.luminance_adapter
            adapted=torch.tanh(adapter.gain*(luminance-local)/(local.abs()+adapter.eps))
            form=spatial_filter(adapted,kernels['contrast'],backbone.contrast_filter.filters.bias)
            form=torch.cat([form[:,:2].relu(),form[:,2:]],1)
            difference=.5*(sampled[:,:1]-sampled[:,1:])
        else:
            smooth=spatial_filter(gb.mean(1,keepdim=True),kernels['gaussian'])
            gradients=spatial_filter(smooth,kernels['sobel'])
            form=torch.cat([gradients,torch.linalg.vector_norm(gradients,dim=1,keepdim=True)],1)
            difference=gb[:,:1]-gb[:,1:]
        colour=torch.cat([difference.relu(),(-difference).relu()],1)
        return [adaptive_avg_pool2d_anysize(feature,projection.pool_hw)
                for feature,projection in zip((form,colour),self.features.legacy)]
