"""Video of every saved synthetic demo, using acquired camera views and real traces."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Polygon
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apiaviz.research.grassland_smoke import position_key,sample_panorama
from apiaviz.research.study import file_hash


def video(out,row):
    data=json.loads((out/row['trace']).read_text())
    env=out/'environment'
    base=json.loads((env/'protocol.json').read_text())
    geometry=json.loads((env/'collision.json').read_text())
    worldmeta=json.loads((env/'world.json').read_text())
    route=np.asarray(base['route'])
    views=[e for e in data['events'] if e['kind'] in ('observation','avoidance_observation')]
    memory=[e for e in views if e['kind']=='observation']
    blocks=[e for e in data['events'] if e['kind']=='blocked_proposal']
    viewtimes=np.array([e['time_s'] for e in views])
    moves=data['microtrace']; movetimes=np.array([e['time_s'] for e in moves])
    path=np.array([data['start']]+[m['position'] for m in moves])
    famtimes=np.array([e['time_s'] for e in memory]); familiarity=[e['familiarity'] for e in memory]
    @lru_cache(maxsize=128)
    def camera(key,yaw):
        with Image.open(env/'panoramas'/f'{key}.png') as im:
            return sample_panorama(np.asarray(im.convert('RGB')),[yaw],(51,199))[0].permute(1,2,0).numpy()[:,::-1]
    plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#edf3f8','axes.labelcolor':'#b4c8d5',
        'axes.titlecolor':'#edf3f8','xtick.color':'#91aaba','ytick.color':'#91aaba','axes.edgecolor':'#425567'})
    fig=plt.figure(figsize=(12.8,7.2),facecolor='#0c1824')
    gs=fig.add_gridspec(2,2,height_ratios=[.85,1.1],width_ratios=[1.6,1],top=.79,bottom=.12,hspace=.37,wspace=.22)
    eye=fig.add_subplot(gs[0,:]); overhead=fig.add_subplot(gs[1,0]); signal=fig.add_subplot(gs[1,1])
    for ax in (eye,overhead,signal): ax.set_facecolor('#152637')
    titles={'clear-lane':'Along the taught route','toward-rock':'Released towards the rock'}
    fig.text(.075,.949,titles[row['id']],fontsize=24,weight='bold')
    fig.text(.075,.902,'NEW SYNTHETIC SCENE  /  ApiaViz RGB + camera-only avoidance  /  Fresh learned route memory',fontsize=10,color='#94b7c9')
    status=fig.text(.075,.845,'',fontsize=12,color='#63dec8')
    fig.text(.075,.038,'4× simulation time  •  Acquired images held between observations  •  Overhead geometry is display-only',fontsize=9,color='#9fb3c2')
    picture=eye.imshow(camera(position_key(views[0]['position']),views[0]['heading']),aspect='auto',interpolation='nearest')
    eye.set_title('Actual camera input  ·  199 × 51 RGB  ·  view turns with the agent',loc='left',fontsize=10,pad=8)
    eye.axis('off')
    for rock in geometry['rocks']:
        overhead.add_collection(PolyCollection(np.asarray(rock['triangles_xy_m']),facecolors='#8b9189',edgecolors='none',alpha=.8))
    for rock in worldmeta['obstacles']:
        overhead.add_patch(Circle((rock['x'],rock['y']),rock['conservative_radius_m'],fill=False,
            edgecolor='#bd885e',linestyle='--',linewidth=.8,alpha=.8))
    overhead.plot(*route.T,':',color='#90b2c8',lw=1.7,label='Taught route')
    overhead.scatter(*route[-1],marker='*',s=100,color='#f5ca76',zorder=5)
    overhead.text(route[-1,0]-.05,route[-1,1]-.14,'goal',fontsize=8,color='#f5ca76')
    travelled,=overhead.plot([],[],color='#59e1c3',lw=2.2,zorder=4)
    ant=Polygon([[0,0],[0,0],[0,0]],color='#ffce73',zorder=7); overhead.add_patch(ant)
    rejected,=overhead.plot([],[],'x',color='#ff7676',ms=9,mew=2,zorder=6)
    lo=np.minimum(path.min(0)-.2,[-.3,-.62]); hi=np.maximum(path.max(0)+.2,[1.8,.45])
    overhead.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal')
    overhead.set_title('Actual path and rock meshes  ·  dashed amber = old discarded discs',loc='left',fontsize=9)
    overhead.tick_params(labelsize=8); overhead.set_xlabel('Position (metres)  ·  agent marker enlarged',fontsize=8)
    signal.set_title('Visual familiarity',loc='left',fontsize=11)
    signal.set(xlim=(0,max(1,row['time_s'])),ylim=(max(0,min(familiarity)-.07),min(1.05,max(familiarity)+.07)))
    signal.set_xlabel('Simulation time (seconds)',fontsize=8); signal.tick_params(labelsize=8)
    line,=signal.plot([],[],color='#63dec8',lw=1.1); cursor=signal.axvline(0,color='#f5ca76',lw=1)
    label=signal.text(.03,.05,'',transform=signal.transAxes,fontsize=10,color='#f5ca76')
    dest=out/'videos'; dest.mkdir(exist_ok=True)
    target=dest/f"{row['id']}.mp4"
    if target.exists(): raise FileExistsError(target)
    fps=12; speed=4.; frames=np.linspace(0,row['time_s'],max(2,int(np.ceil(row['time_s']/speed*fps))+1))
    writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2600,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    with writer.saving(fig,str(target),dpi=100):
        for i,t in enumerate(frames):
            vi=np.searchsorted(viewtimes,t,side='right')-1
            picture.set_visible(vi>=0)
            if vi>=0:
                v=views[vi]; picture.set_data(camera(position_key(v['position']),v['heading']))
            count=int(np.searchsorted(movetimes,t,side='right'))
            travelled.set_data(path[:count+1,0],path[:count+1,1])
            heading=views[max(0,vi)]['heading']; a=np.deg2rad(heading)
            rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            ant.set_xy(np.array([[.033,0],[-.02,.018],[-.02,-.018]])@rot.T+path[count])
            active=[e for e in blocks if 0<=t-e['time_s']<.5]
            rejected.set_data([e['proposed_position'][0] for e in active],[e['proposed_position'][1] for e in active])
            n=np.searchsorted(famtimes,t,side='right'); line.set_data(famtimes[:n],familiarity[:n]); cursor.set_xdata([t,t])
            blocked=sum(e['time_s']<=t for e in blocks)
            walked=moves[count-1]['path_m'] if count else 0.
            action='Visual steering' if count and moves[count-1]['state']=='avoid' else 'Route navigation'
            if active: action='Contact blocked — trial continues'
            status.set_text(f'{action}    |    {t:5.1f} s    |    travelled {walked:.2f} m    |    blocked attempts {blocked}')
            if i==len(frames)-1:
                label.set_text('Outcome: '+row['termination'].replace('_',' ')+'\nNo route resets')
            writer.grab_frame()
            if i==len(frames)//2: fig.savefig(dest/f"{row['id']}.png",dpi=100,facecolor=fig.get_facecolor())
        for _ in range(fps*3): writer.grab_frame()
    plt.close(fig)
    record=dict(result=row,video_sha256=file_hash(target),trace_sha256=file_hash(out/row['trace']),
        video_source_sha256=file_hash(Path(__file__)),fps=fps,playback_speed=speed,
        frames=len(frames)+fps*3,scope='Actual synthetic closed-loop navigation, all prespecified outcomes included')
    (dest/f"{row['id']}.json").write_text(json.dumps(record,indent=2)+'\n')
    print(target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trial',choices=['clear-lane','toward-rock'])
    args=parser.parse_args()
    for row in json.loads((args.output/'results.json').read_text()):
        if args.trial is None or row['id']==args.trial: video(args.output,row)
