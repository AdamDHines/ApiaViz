"""Visual audit of a saved camera-only locomotion trace (no new views rendered)."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import numpy as np
from PIL import Image

from .grassland_smoke import sample_panorama, position_key
from .visual_avoidance import gray


def render(trace_path, environment, output):
    data = json.loads(trace_path.read_text())
    rows = data.get('microtrace', data.get('trace'))
    if 'start' in data:
        start = np.array(data['start'])
    else:
        start = np.array(next(e['position'] for e in data['events'] if 'position' in e))
    metadata = json.loads((environment/'world.json').read_text())
    path = np.array([start]+[r['position'] for r in rows])
    def view(pos, yaw):
        with Image.open(environment/'panoramas'/f'{position_key(pos)}.png') as im:
            return sample_panorama(np.array(im.convert('RGB')), [yaw], (51,199))[0].permute(1,2,0).numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':'#edf3f5',
        'axes.labelcolor':'#c4d0d8', 'xtick.color':'#afbdc9', 'ytick.color':'#afbdc9',
        'axes.edgecolor':'#465569', 'axes.titlecolor':'#edf3f5'})
    fig = plt.figure(figsize=(12,7.2), facecolor='#101c29')
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15,1,.18], width_ratios=[1.5,1],
                         hspace=.46, wspace=.22, top=.83, bottom=.07)
    axview = fig.add_subplot(gs[0,:])
    axflow = fig.add_subplot(gs[1,0])
    axmap = fig.add_subplot(gs[1,1])
    for ax in (axview,axflow,axmap):
        ax.set_facecolor('#18293a')
    title = fig.text(.07,.95,'A rock is a detour', fontsize=23, weight='bold')
    info = fig.text(.07,.90,'',fontsize=11,color='#c4d0d8')
    fig.text(.07,.04,'Camera-only reflex  •  199 × 51 RGB  •  No depth sensor or obstacle coordinates', fontsize=10,color='#c4d0d8')
    writer = FFMpegWriter(fps=6, codec='libx264', bitrate=2200, extra_args=['-pix_fmt','yuv420p'])
    with writer.saving(fig, str(output), dpi=110):
        for i, row in enumerate(rows):
            after = view(row['position'], row['heading'])
            before = view(path[i], row['heading'])
            flow = cv2.calcOpticalFlowFarneback(gray(before),gray(after),None,.5,3,9,5,5,1.1,0)
            axview.clear(); axflow.clear(); axmap.clear()
            # Display mirrored so positive (leftward) yaw is on screen left.
            axview.imshow(after[:,::-1], aspect='auto', interpolation='nearest')
            axview.set_title('Ant’s view',loc='left',fontsize=12,pad=8)
            axview.axis('off')
            axflow.imshow(after[:,::-1], aspect='auto', alpha=.6, interpolation='nearest')
            yy,xx = np.mgrid[24:45:4,3:196:6]
            dx = -flow[yy,198-xx,0]; dy=flow[yy,198-xx,1]
            mask = np.hypot(dx,dy) > .25
            axflow.quiver(xx[mask], yy[mask], dx[mask], dy[mask], color='#69e3d0',
                          angles='xy', scale_units='xy', scale=.25, width=.004)
            axflow.set_xlim(0,198); axflow.set_ylim(50,20)
            axflow.set_title('Image motion between matched views  (arrows ×4)',loc='left',fontsize=10)
            axflow.axis('off')
            lo,hi=path.min(0)-.15,path.max(0)+.15
            span=max(*(hi-lo),.45)
            centre=(lo+hi)/2
            for rock in metadata['obstacles']:
                axmap.add_patch(Circle((rock['x'],rock['y']),rock['conservative_radius_m'],
                                      facecolor='#a9a596',edgecolor='#dfd6b6',alpha=.8))
            axmap.plot(path[:i+2,0],path[:i+2,1],color='#69e3d0',lw=2)
            axmap.scatter(*start,color='#edf3f5',s=25,zorder=4)
            a=np.deg2rad(row['heading']); pos=np.array(row['position'])
            axmap.arrow(*pos,.06*np.cos(a),.06*np.sin(a),width=.005,color='#ffcf73',zorder=5)
            axmap.set_xlim(centre[0]-span/2,centre[0]+span/2)
            axmap.set_ylim(centre[1]-span/2,centre[1]+span/2)
            axmap.set_aspect('equal'); axmap.tick_params(labelsize=7)
            axmap.set_title('Overhead audit view',loc='left',fontsize=11)
            axmap.set_xlabel('Position (m) • map used for display only',fontsize=8)
            state = 'Turning around the obstacle' if row['state']=='avoid' else 'Continuing forward'
            info.set_text(f"{state}     |     walked {row['path_m']:.2f} m     |     simulation time {row['time_s']:.1f} s")
            writer.grab_frame()
            if i == len(rows)//3:
                fig.savefig(output.with_suffix('.png'),dpi=140,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,default=Path('apiaviz/output/visual-avoidance-dev/hairpin-straight.json'))
    parser.add_argument('--environment',type=Path,default=Path('apiaviz/output/navigation-regime/hairpin'))
    parser.add_argument('--output',type=Path,default=Path('docs/visual-avoidance/rock-detour.mp4'))
    args=parser.parse_args()
    render(args.trace,args.environment,args.output)
