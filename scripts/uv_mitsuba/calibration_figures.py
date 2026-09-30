"""Figures and descriptive diagnostics; one shared display transform throughout."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from calibrate import OUT, ROOT
import render as base


def main():
    dest=ROOT/'docs/uv-calibration'
    dest.mkdir(parents=True,exist_ok=True)
    names=['middle-r0-m0','middle-r1-m0','middle-r0-m1','middle-r1-m1',
           'start-calibrated','end-calibrated','sun-shift-calibrated']
    data={n:np.load(OUT/'renders'/f'{n}.npy') for n in names}
    exposure=float(np.percentile(np.concatenate([a.ravel() for a in data.values()]),98)*.45)
    def display(a): return np.clip(a/(a+exposure),0,1)**(1/2.2)
    def panel(ax,name,title):
        ax.imshow(display(data[name])); ax.set_title(title,loc='left',fontsize=12)
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
    for name,a in data.items(): Image.fromarray(np.uint8(display(a)*255)).save(dest/f'{name}.png')
    fig,axes=plt.subplots(2,2,figsize=(16,6.4),layout='constrained')
    for ax,name,title in zip(axes.ravel(),names[:4],['Original illustrative curves','Measured honeybee responses only',
                 'Measured material proxies only','Both substitutions']): panel(ax,name,title)
    fig.suptitle('What changes when we use measured spectra?\nSame viewpoint, geometry and sun · UV → red, blue → green, green → blue',fontsize=17)
    fig.savefig(dest/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(14,17),layout='constrained')
    for ax,name,title in zip(axes,['start-calibrated','middle-r1-m1','end-calibrated','sun-shift-calibrated'],
            ['Route start','Route middle','Route end','Route middle with a different sun direction']): panel(ax,name,title)
    fig.suptitle('UV:blue:green with measured receptor and material curves\nLibrary proxies for a synthetic grassland; sky remains a model',fontsize=17)
    fig.savefig(dest/'panoramas.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    for i,ax in enumerate(axes):
        ax.imshow(display(data['middle-r1-m1'][...,i]),cmap='gray',vmin=0,vmax=1)
        ax.set_title(['UV','Blue','Green'][i],loc='left'); ax.set_axis_off()
    fig.suptitle('Three receptor bands · shared display exposure',fontsize=17)
    fig.savefig(dest/'channels.png',dpi=120);plt.close(fig)
    c=json.loads((OUT/'calibration.json').read_text()); wl=np.array(c['wavelength_nm'])
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for i,(label,col) in enumerate(zip(['UV','Blue','Green'],['#983eb0','#2575b9','#318754'])):
        axes[0].plot(wl,c['receptor_weights'][i],color=col,label=label+' measured table')
        axes[0].plot(wl,base.receptor_spectra()[i],color=col,linestyle='--',alpha=.6)
    for key,values in c['material_reflectance'].items(): axes[1].plot(wl,values,label=key.replace('_',' '))
    axes[0].set(title='Receptor weighting: empirical (solid), initial (dashed)',ylabel='Normalized energy weighting (nm⁻¹)')
    axes[1].set(title='Measured material proxies',ylabel='Reflectance',ylim=(0,1))
    for ax in axes:
        ax.set_xlabel('Wavelength (nm)');ax.axvspan(320,400,color='#e5d2ef',alpha=.4)
        ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.savefig(dest/'spectra.png',dpi=150);plt.close(fig)
    rows=[]
    for name in names:
        record=json.loads((OUT/'renders'/f'{name}.json').read_text())
        rows.append(dict(render=name,**{f'{ch}_sky_terrain_ratio':v for ch,v in zip(['UV','blue','green'],record['sky_terrain_ratio'])}))
    pd.DataFrame(rows).to_csv(dest/'diagnostics.csv',index=False)
    (dest/'display.json').write_text(json.dumps(dict(exposure=exposure,
        transform='(x/(x+exposure))**(1/2.2)',mapping='UV->red, blue->green, green->blue',
        note='Identical exposure and channel gains for every displayed image; linear data kept separately.'),indent=2)+'\n')
    (dest/'sources.lock.json').write_text(json.dumps(c['sources'],indent=2)+'\n')
    for name in ['receptors-source.csv','receptors-render.csv','materials-render.csv','calibration.json','validation.json']:
        (dest/name).write_bytes((OUT/name).read_bytes())
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__': main()
