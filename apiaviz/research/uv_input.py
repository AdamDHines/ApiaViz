"""Verified spectral panorama -> ApiaViz retinal input, without display colour.

Matches the existing navigation ray grid: columns -148..148 degrees relative to
heading, rows +60..-15 degrees. Bilinear sampling wraps azimuth, never elevation.
"""
import numpy as np
import torch

from .spectral_input import load_frame, validate_arrays

AZIMUTH='360 degrees; centre forward; image left positive from heading; pixel centres'


def sample_receptors(intensity, headings, *, panorama_heading=0., elevation=(90.,-20.), shape=(51,199)):
    validate_arrays(intensity)
    if len(shape)!=2 or any(int(n)!=n or n<5 for n in shape):
        raise ValueError('Require retinal height/width integers >= 5')
    headings=np.asarray(headings,dtype=float)
    if headings.ndim!=1 or not len(headings) or not np.isfinite(headings).all() or not np.isfinite(panorama_heading):
        raise ValueError('Require finite headings')
    high,low=elevation
    if not np.isfinite([high,low]).all() or not -90<=low<high<=90:
        raise ValueError('Invalid panorama elevation bounds')
    h,w=intensity.shape[:2]
    angles=headings[:,None]-panorama_heading+np.linspace(-148.,148.,shape[1])[None]
    x=((180.-angles)%360.)/360.*w-.5
    y=(high-np.linspace(60.,-15.,shape[0]))/(high-low)*h-.5
    if y.min() < -1e-8 or y.max()>h-1+1e-8:
        raise ValueError('Panorama pixel centres do not cover the complete retinal elevation grid')
    y=np.clip(y,0,h-1)
    x0=np.floor(x).astype(int); y0=np.floor(y).astype(int)
    y1=np.minimum(y0+1,h-1)
    dx=x-x0; dy=y-y0
    a=intensity[y0[None,:,None],(x0%w)[:,None,:]]
    b=intensity[y0[None,:,None],((x0+1)%w)[:,None,:]]
    c=intensity[y1[None,:,None],(x0%w)[:,None,:]]
    d=intensity[y1[None,:,None],((x0+1)%w)[:,None,:]]
    top=a*(1-dx[:,None,:,None])+b*dx[:,None,:,None]
    bottom=c*(1-dx[:,None,:,None])+d*dx[:,None,:,None]
    image=top*(1-dy[None,:,None,None])+bottom*dy[None,:,None,None]
    return torch.from_numpy(np.ascontiguousarray(image.transpose(0,3,1,2),dtype=np.float32))


def load_views(sidecar, headings, *, calibration_sha256, shape=(51,199)):
    intensity,_,record=load_frame(sidecar)
    if record.get('calibration_sha256')!=calibration_sha256:
        raise ValueError('Spectral calibration differs from the frozen model input protocol')
    if record.get('azimuth')!=AZIMUTH:
        raise ValueError('Unknown spectral panorama orientation')
    views=sample_receptors(intensity,headings,panorama_heading=record['pose']['heading'],
                           elevation=record['elevation_deg'],shape=shape)
    return views,record
