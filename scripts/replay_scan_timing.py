"""Real-time diagnostic of unchanged logged body yaw; no navigation/render calls."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tqdm import tqdm
from scripts.audit_scanning_regression import yaw_at
from apiaviz.research import uv_trials as base
from apiaviz.research.dual_camera import position_key,digest,load_cached
from apiaviz.research.uv_trial_report import frame_display
from apiaviz.research.spectral_input import file_sha


def position_at(microtrace,initial,t,speed,observation_s):
    """Interpolate only charged translation intervals, holding during scans."""
    previous=np.asarray(initial,dtype=float)
    for move in microtrace:
        after=np.asarray(move['position'],dtype=float)
        end=move['time_s']-observation_s
        duration=move['commanded_stride_m']/speed
        start=end-duration
        if t<start:return previous
        if t<=end:return previous+(after-previous)*np.clip((t-start)/duration,0,1)
        previous=after
    return previous


def run(study,out):
    if out.exists():raise FileExistsError('Fresh diagnostic movie output required')
    out.mkdir(parents=True)
    p=json.loads((study/'protocol.json').read_text());base.verify_sources(p)
    path=study/'trials/meander-19-apiaviz_uv-aligned-familiarity-1.json'
    detail=json.loads(path.read_text());row=detail['result'];env=study/'worlds/meander'
    cfg=json.loads((env/'render.json').read_text());world=next(w for w in p['worlds'] if w['name']=='meander')
    initial=world['headings'][0];events=detail['events']
    views=[e for e in events if e['kind'] in ('observation','avoidance_observation')]
    times=np.array([e['time_s'] for e in views])
    fps=30;duration=12.;frames=np.linspace(0,duration,int(duration*fps)+1)
    yaw=yaw_at(events,frames,initial)
    assert np.max(abs(np.diff(yaw)))*fps<=p['sensor_settings']['yaw_speed_deg_s']+1e-6
    # Sampled images remain the genuine last acquired image, never interpolated
    # or re-rendered at display-only times. The body marker uses logged yaw.
    references={}
    @lru_cache(maxsize=128)
    def images(key,heading):
        record=env/'camera'/f'{key}.json'
        assert file_sha(record)==detail['camera_frames'][key]
        raw,_=load_cached(env/'camera',key,digest(cfg));references[key]=file_sha(record)
        return frame_display(raw,heading,cfg)
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/apiaviz-scan-review-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Polygon
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    uv,rgb,overhead,signal=axes.flat
    pictures=[ax.imshow(np.zeros((51,199,3)),aspect='auto') for ax in (uv,rgb)]
    for ax,title in [(uv,'Last acquired UV/B/G image (false colours)'),(rgb,'Last acquired visible image')]:ax.set_title(title,fontsize=10);ax.axis('off')
    route=np.asarray(world['route']);overhead.plot(*route.T,':',color='gray',label='Taught route (display only)')
    overhead.set(xlim=(-.15,.65),ylim=(-.15,.55),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Actual body yaw during the initial route segment')
    trail,=overhead.plot([],[],color='#138c85',lw=2)
    marker=Polygon([[0,0],[0,0],[0,0]],color='#db852c');overhead.add_patch(marker)
    gaze,=overhead.plot([],[],'--',color='#3174ad',lw=1,label='Last observed gaze')
    overhead.legend(fontsize=8,loc='upper left')
    signal.plot(frames,yaw-initial,color='#db852c',label='Logged yaw, constant-rate turn interpolation')
    video_frames=np.linspace(0,row['time_s'],96);vi=np.searchsorted(times,video_frames,side='right')-1
    held=np.where(vi>=0,np.array([e['heading'] for e in views])[np.maximum(0,vi)],initial)
    keep=video_frames<=duration
    signal.scatter(video_frames[keep],held[keep]-initial,color='#3174ad',label='Frames kept in original accelerated video',s=25)
    signal.set(xlim=(0,duration),xlabel='Simulation time (s)',ylabel='Accumulated yaw from initial heading (degrees)')
    signal.legend(fontsize=8);signal.grid(alpha=.2);cursor=signal.axvline(0,color='black',lw=.8)
    status=fig.suptitle('Unchanged ApiaViz trial · real-time diagnostic · 30 fps',fontsize=13)
    movie=out/'meander-first-12s-realtime.mp4';writer=FFMpegWriter(fps=fps,codec='libx264',bitrate=2000,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    walked=[]
    with writer.saving(fig,str(movie),dpi=100):
        for i,t in enumerate(tqdm(frames,desc='Real-time scan replay',unit='frame')):
            index=int(np.searchsorted(times,t,side='right')-1)
            if index>=0:
                v=views[index];a,b=images(position_key(v['position'],cfg.get('pose_decimals')),v['heading'])
                for artist,array in zip(pictures,(a,b)):artist.set_data(array)
                look=np.deg2rad(v['heading'])
            else:look=np.deg2rad(initial)
            pos=position_at(detail['microtrace'],row['initial_position'],t,p['sensor_settings']['speed_m_s'],p['sensor_settings']['observation_s'])
            walked.append(pos);points=np.asarray(walked);trail.set_data(*points.T)
            angle=np.deg2rad(yaw[i]);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            marker.set_xy(np.array([[.04,0],[-.025,.02],[-.025,-.02]])@rotation.T+pos)
            end=pos+.09*np.array([np.cos(look),np.sin(look)]);gaze.set_data([pos[0],end[0]],[pos[1],end[1]])
            cursor.set_xdata([t,t]);status.set_text(f'UNCHANGED controller · {t:.2f} simulated seconds · 1× playback · 30 fps')
            if i==round(2.5*fps):fig.savefig(out/'preview.png',dpi=100)
            writer.grab_frame()
    plt.close(fig)
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],check=True,capture_output=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(movie)]))
    assert int(probe['streams'][0]['nb_read_frames'])==len(frames)
    base.atomic_json(out/'provenance.json',dict(protocol_sha256=file_sha(study/'protocol.json'),trace_sha256=file_sha(path),
        sources={str(Path(__file__).relative_to(ROOT)):file_sha(Path(__file__)),
            'scripts/audit_scanning_regression.py':file_sha(ROOT/'scripts/audit_scanning_regression.py'),
            'apiaviz/research/uv_trial_report.py':file_sha(ROOT/'apiaviz/research/uv_trial_report.py')},
        movie_sha256=file_sha(movie),frames=len(frames),fps=fps,simulated_seconds=duration,decoded=True,
        camera_frames=references,limitation='Recorded trial unchanged. Body yaw interpolates charged constant-rate turns; position interpolates only charged translations. Sensor images are genuine last observations held between samples. No new simulation or images.'))
    print(movie)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/graded-navigation-smoke-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'apiaviz/output/scan-regression-review-v1')
    args=parser.parse_args();run(args.study.resolve(),args.output.resolve())
