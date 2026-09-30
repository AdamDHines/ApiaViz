"""Display the numerical spectral renders; never alter the saved linear data."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'apiaviz/output/uv-mitsuba'
DOCS=ROOT/'docs/uv-mitsuba'
DOCS.mkdir(parents=True,exist_ok=True)
NAMES=['01-route-start','02-route-middle','03-route-end','04-middle-sun-shift']
TITLES=['Route start · sun 35° azimuth / 38° elevation',
        'Route middle · sun 35° / 38°',
        'Route end · sun 35° / 38°',
        'Same middle viewpoint · sun moved to 145° / 20°']
arrays=[np.load(RAW/'renders'/f'{name}.npy') for name in NAMES]
exposure=float(np.percentile(np.concatenate([a.ravel() for a in arrays]),98)*.45)


def display(a):
    return np.clip(a/(a+exposure),0,1)**(1/2.2)


def style(ax):
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)


fig,axes=plt.subplots(4,1,figsize=(15,18),layout='constrained',facecolor='#101e2a')
fig.get_layout_engine().set(rect=(0,.025,1,.975))
for ax,a,name,title in zip(axes,arrays,NAMES,TITLES):
    rgb=display(a)
    Image.fromarray(np.uint8(rgb*255)).save(DOCS/f'{name}.png')
    ax.imshow(rgb)
    ax.set_title(title,loc='left',color='#e3edf2',fontsize=15,pad=9)
    style(ax)
fig.suptitle('Grassland through UV, blue and green bands\nFalse colour: UV → red  |  blue → green  |  green → blue',color='white',fontsize=19)
fig.text(.5,.003,'360° panoramas · +90° to −20° elevation · common display exposure · illustrative material spectra',ha='center',color='#bdcdd5',fontsize=12)
fig.savefig(DOCS/'panoramas.png',dpi=120,facecolor=fig.get_facecolor())
plt.close(fig)

fig,axes=plt.subplots(4,1,figsize=(14,16),layout='constrained')
axes[0].imshow(display(arrays[1]))
axes[0].set_title('Route middle · UV:blue:green false colour',loc='left')
for i,name in enumerate(['UV · approximate peak 344 nm','Blue · approximate peak 436 nm','Green · approximate peak 544 nm']):
    axes[i+1].imshow(display(arrays[1][...,i]),cmap='gray',vmin=0,vmax=1)
    axes[i+1].set_title(name+' · same display exposure',loc='left')
for ax in axes:style(ax)
fig.savefig(DOCS/'channels.png',dpi=120)
plt.close(fig)

fig,axes=plt.subplots(3,2,figsize=(15,9),layout='constrained')
for idx in range(2):
    s=np.load(RAW/'polarization'/f'stokes-{idx}.npz')['stokes']
    intensity=s[0,...,0]
    dolp=np.sqrt(s[1,...,0]**2+s[2,...,0]**2)/np.maximum(intensity,1e-15)
    angle=np.mod(np.rad2deg(.5*np.arctan2(s[2,...,0],s[1,...,0])),180)
    angle=np.ma.masked_where(dolp<.02,angle)
    axes[0,idx].imshow(display(s[0]))
    axes[0,idx].set_title(['Original sun · 35° / 38°','Moved sun · 145° / 20°'][idx])
    im1=axes[1,idx].imshow(dolp,vmin=0,vmax=.75,cmap='viridis')
    im2=axes[2,idx].imshow(angle,vmin=0,vmax=180,cmap='twilight')
    for ax in axes[:,idx]:style(ax)
axes[0,0].set_ylabel('UV:blue:green intensity')
axes[1,0].set_ylabel('UV polarization fraction')
axes[2,0].set_ylabel('UV polarization angle')
fig.colorbar(im1,ax=axes[1,:],shrink=.7,label='Fraction linearly polarized')
fig.colorbar(im2,ax=axes[2,:],shrink=.7,label='Angle (degrees; modulo 180°)')
fig.suptitle('Polarized-sky prototype · same viewpoint, different sun\nSpectral sky intensity + explicit Rayleigh approximation; diffuse surfaces depolarize',fontsize=15)
fig.savefig(DOCS/'polarization.png',dpi=130)
plt.close(fig)

meta=json.loads((RAW/'renders'/'01-route-start.json').read_text())
wavelengths=np.array(meta['wavelength_nm'])
fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
for curve,label,col in zip(meta['receptor_weights'],['UV','Blue','Green'],['#a643ce','#2381c1','#3f944f']):
    axes[0].plot(wavelengths,curve,label=label,color=col)
for label in ['Grass 0','Grass 1','Grassland earth','Limestone','Weathered twigs']:
    axes[1].plot(wavelengths,meta['materials'][label],label=label)
axes[0].set(title='Approximate receptor bands',xlabel='Wavelength (nm)',ylabel='Normalized photon-weighted response')
axes[1].set(title='Illustrative material reflectance',xlabel='Wavelength (nm)',ylabel='Reflected fraction',ylim=(0,1))
for ax in axes:
    ax.axvspan(320,400,color='#d5b9eb',alpha=.2)
    ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(DOCS/'assumptions.png',dpi=140)
plt.close(fig)
(DOCS/'display.json').write_text(json.dumps(dict(mapping={'red':'UV','green':'blue','blue':'green'},
    exposure=exposure,curve='(radiance / (radiance + exposure)) ** (1/2.2)',
    normalization='One common exposure and channel gain for all four panoramas; no per-image white balance',
    raw_data=str(RAW)),indent=2)+'\n')
print('Saved panoramas, channel comparison, polarization and assumptions figures to',DOCS)
