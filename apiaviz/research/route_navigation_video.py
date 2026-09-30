"""Actual acquired camera views, visual familiarity and executed navigation."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from .grassland_smoke import position_key,sample_panorama


def render(out,trial,target,speed=4.):
    p=json.loads((out/'protocol.json').read_text())
    row=next(r for r in map(json.loads,(out/'results.jsonl').read_text().splitlines()) if r['id']==trial)
    data=json.loads((out/row['trace']).read_text())
    env=Path(next(w['environment'] for w in p['worlds'] if w['name']==row['world']))
    base=json.loads((env/'protocol.json').read_text())
    rocks=json.loads((env/'world.json').read_text())['obstacles']
    route=np.array(base['route'])
    views=[e for e in data['events'] if e['kind'] in ('observation','avoidance_observation')]
    memory=[e for e in views if e['kind']=='observation']
    times=np.array([e['time_s'] for e in views])
    mtimes=np.array([e['time_s'] for e in memory])
    familiarity=np.array([e['familiarity'] for e in memory])
    moves=data['microtrace']; motion_times=np.array([m['time_s'] for m in moves])
    decisions=data['decisions']; dtimes=np.array([d.get('time_end_s',d['time_s']) for d in decisions])
    path=np.array([views[0]['position']]+[m['position'] for m in moves])
    @lru_cache(maxsize=128)
    def camera(key,yaw):
        with Image.open(env/'panoramas'/f'{key}.png') as im:
            return sample_panorama(np.array(im.convert('RGB')),[yaw],p['shape'])[0].permute(1,2,0).numpy()[:,::-1]
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#e9f0f4',
        'axes.labelcolor':'#bdcbd5','xtick.color':'#aabcca','ytick.color':'#aabcca',
        'axes.edgecolor':'#526171','axes.titlecolor':'#e9f0f4'})
    fig=plt.figure(figsize=(12,7.2),facecolor='#101c29')
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.1],width_ratios=[1.45,1],
                       top=.81,bottom=.12,hspace=.36,wspace=.22)
    eye=fig.add_subplot(gs[0,:]); signal=fig.add_subplot(gs[1,0]); world=fig.add_subplot(gs[1,1])
    for ax in (eye,signal,world):ax.set_facecolor('#192b3c')
    name={'linear_colour':'ApiaViz','sobel_colour':'Sobel + colour','ardin_input':'Ardin'}[row['method']]
    fig.text(.075,.945,f'{name} · visual navigation with obstacle avoidance',fontsize=19,weight='bold')
    status=fig.text(.075,.885,'',fontsize=12,color='#b9d7df')
    fig.text(.075,.04,f'{speed:g}× simulation time  •  Actual acquired images, held between observations  •  Map for display only',
             fontsize=9,color='#b9c8d3')
    image_artist=eye.imshow(camera(position_key(views[0]['position']),views[0]['heading']),aspect='auto',interpolation='nearest')
    eye.set_title('Ant’s view  ·  199 × 51 RGB',loc='left',fontsize=11); eye.axis('off')
    signal.set_title('Visual familiarity  ·  higher values mean a closer memory match',loc='left',fontsize=10)
    signal.set_xlabel('Simulation time (s)',fontsize=9)
    signal.set_ylim(max(0,float(familiarity.min())-.05),min(1.05,float(familiarity.max())+.05))
    signal.set_xlim(0,row['time_s'])
    famline,=signal.plot([],[],color='#66dace',lw=1)
    cursor=signal.axvline(0,color='#f9cd72',lw=1)
    caption=signal.text(.02,.06,'',transform=signal.transAxes,fontsize=10,color='#f9cd72')
    for rock in rocks:
        world.add_patch(Circle((rock['x'],rock['y']),rock['conservative_radius_m'],color='#9b998d',alpha=.5))
    world.plot(*route.T,'--',color='#718797',lw=1,label='Teaching route')
    travelled,=world.plot([],[],color='#66dace',lw=1.5)
    ant,=world.plot([],[],'o',color='#f9cd72',ms=5)
    world.scatter(*route[-1],marker='*',color='#e9f0f4',s=65)
    world.set(xlim=(1.3,8.3),ylim=(3.6,8.8),aspect='equal')
    # Include the full executed trajectory if a failure leaves this route area.
    world.set_xlim(min(1.3,path[:,0].min()-.2),max(8.3,path[:,0].max()+.2))
    world.set_ylim(min(3.6,path[:,1].min()-.2),max(8.8,path[:,1].max()+.2))
    world.set_title('Executed path  ·  star marks the nest',loc='left',fontsize=10)
    world.tick_params(labelsize=8)
    target.parent.mkdir(parents=True,exist_ok=True)
    fps=12; writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2000,extra_args=['-pix_fmt','yuv420p'])
    frames=np.linspace(0,row['time_s'],int(np.ceil(row['time_s']/speed*fps))+1)
    with writer.saving(fig,str(target),dpi=100):
        for i,t in enumerate(frames):
            vi=np.searchsorted(times,t,side='right')-1
            if vi>=0:
                v=views[vi]; image_artist.set_data(camera(position_key(v['position']),v['heading']))
                image_artist.set_visible(True)
            else:image_artist.set_visible(False)
            count=int(np.searchsorted(motion_times,t,side='right'))
            travelled.set_data(path[:count+1,0],path[:count+1,1]); ant.set_data([path[count,0]],[path[count,1]])
            n=int(np.searchsorted(mtimes,t,side='right')); famline.set_data(mtimes[:n],familiarity[:n]); cursor.set_xdata([t,t])
            di=np.searchsorted(dtimes,t,side='right')-1
            navstate=decisions[di].get('state',decisions[di].get('state_before','observing')) if di>=0 else 'observing'
            steering=count>0 and moves[count-1]['state']=='avoid'
            walked=moves[count-1]['path_m'] if count else 0.
            action='Turning around an obstacle' if steering else {'follow':'Following familiar views','cast':'Searching across the route','reorient':'Checking direction'}.get(navstate,'Observing')
            status.set_text(f'{action}     |     {t:.1f} s     |     walked {walked:.2f} m')
            text=''
            if di>=0 and 'comparison' in decisions[di]:
                c=decisions[di]['comparison']
                text=f"Matched view: {c['before']:.3f} → {c['after']:.3f}"
            if i==len(frames)-1:
                text=f"Outcome: {row['termination'].replace('_',' ')}"
            caption.set_text(text)
            writer.grab_frame()
            if i==len(frames)//3:fig.savefig(target.with_suffix('.png'),dpi=130,facecolor=fig.get_facecolor())
        for _ in range(fps*2):writer.grab_frame()
    plt.close(fig)
    print(target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/route-motor-feedback-v2'))
    parser.add_argument('--trial',default='hairpin-19-linear_colour-aligned-familiarity-1')
    parser.add_argument('--video',type=Path,default=Path('docs/route-detour/apiaviz-aligned.mp4'))
    args=parser.parse_args();render(args.output,args.trial,args.video)
