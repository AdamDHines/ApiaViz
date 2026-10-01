"""Two-command, resumable UV/visible navigation study. Full runs are user-started."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import torch
from tqdm import tqdm

from .collision_geometry import RockGeometry
from .dual_camera import DualCamera,digest,load_cached,position_key
from .evaluation_safety import EvaluationSafety,safe_displacement
from .familiarity_controller import FamiliarityController,ControllerAdapter,Settings,acquisition_calibration
from .frontend_refinements import RefinementEncoder
from .motor_feedback import evaluate
from .route_full import cases
from .route_full_audit import check_motion
from .spectral_input import file_sha
from .spike_overlap import SpikeOverlapMemory
from .study import encode,fingerprint
from .uv_encoder import ApiaVizUVEncoder,UVEncoderConfig
from .visual_avoidance import Settings as AvoidanceSettings

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'apiaviz/output/uv-trials-v1'
METHODS=['apiaviz_uv','sobel_colour','ardin_input']


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


@contextmanager
def owned_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another command owns this output directory')
        yield


def sources():
    files=[p for p in (ROOT/'apiaviz').rglob('*.py') if 'output' not in p.relative_to(ROOT).parts and 'data' not in p.relative_to(ROOT).parts]
    files+=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts/uv_mitsuba').glob('*.py'))
    files += [ROOT/p for p in ('pixi.toml','pixi.lock','scripts/uv_mitsuba/pixi.toml','scripts/uv_mitsuba/pixi.lock')]
    return {str(f.relative_to(ROOT)):file_sha(f) for f in sorted(files)}


def verify_sources(p):
    for name,value in p['source_sha256'].items():
        if file_sha(ROOT/name)!=value: raise ValueError(f'Source changed: {name}. Use a fresh --output; frozen runs cannot mix code.')
    if file_sha(ROOT/'docs/uv-calibration/calibration.json')!=p['calibration_sha256']:
        raise ValueError('Frozen calibration changed')


def route_for(name,smoke=False):
    if smoke: points=np.array([[0.,0.],[.5,0.]])
    elif name=='meander':
        x=np.linspace(0,5.5,300); points=np.column_stack([x,.65*np.sin(x*1.2)])
    elif name=='bend': points=np.array([[0.,0.],[2.,0.],[4.,2.],[5.5,2.]])
    else: points=np.array([[0.,0.],[3.,0.],[3.75,.75],[3.,1.5],[0.,1.5]])
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    stations=np.r_[np.arange(0,distance[-1]-.00001,.1),distance[-1]]
    route=np.column_stack([np.interp(stations,distance,points[:,i]) for i in range(2)])
    delta=np.diff(route,axis=0); headings=np.rad2deg(np.arctan2(delta[:,1],delta[:,0]))
    return route.tolist(),np.r_[headings,headings[-1]].tolist()


def configuration(smoke):
    old=json.loads((ROOT/'docs/route-continuous-full/archive/protocol.json').read_text())
    worlds=[]
    for i,name in enumerate(['meander'] if smoke else ['meander','bend','hairpin']):
        route,headings=route_for(name,smoke)
        worlds.append(dict(name=name,geometry_seed=20261001+i,route=route,headings=headings,
            world_bounds_m=[-1.5,7.,-3.,4.],rock_count=3 if smoke else 60,grass_count=8 if smoke else 180))
    p=dict(schema='apiaviz-uv-trials-v1',smoke=smoke,worlds=worlds,methods=METHODS,
        seeds=[19] if smoke else [19,31,43],controllers=[dict(name='familiarity',phase=h) for h in ([1] if smoke else [1,-1])],
        scenarios=[s for s in old['scenarios'] if not smoke or s['name'] in ('aligned','kick_right50')],
        controller_settings=asdict(Settings()),avoidance_settings=asdict(AvoidanceSettings()),
        sensor_settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,
            time_budget_s=60. if smoke else 360.,observation_budget=500 if smoke else 2600),
        evaluation=dict(max_steps=4 if smoke else 200,kick_before_step=2 if smoke else 36,
            recovery_radius_m=.1,recovery_consecutive_steps=3),
        evaluation_safety=EvaluationSafety().configuration(),body_radius_m=.005,
        encoder=dict(visible_code_dim=8000,uv_code_dim=2000,baseline_code_dim=10000),
        render=dict(width=240 if smoke else 480,height=110 if smoke else 144,spp=4 if smoke else 64,
            threads=2,seed=20261001,sun_azimuth=35.,sun_elevation=38.,elevation_deg=[90.,-20.],
            retina_shape=[51,199],visible_white=1.,
            cache='Lossless compressed float32 six-band NPZ; all acquired scientific inputs retained',
            visible_response='Analytical unit-area photon-weighted Gaussian RGB: peaks 610/540/460nm, widths 35/30/25nm; exactly zero below 400nm; 320-700nm transport',
            visible_mapping='RGB=linear/(linear+1); no UV channel, gamma or per-image gain',
            uv_response='Archived empirical honeybee UV/blue/green weights; linear radiance with scale=1',
            polarization=False,eye_height_m=.01),
        calibration_sha256=file_sha(ROOT/'docs/uv-calibration/calibration.json'),
        source_sha256=sources(),versions=dict(torch=str(torch.__version__),numpy=np.__version__),
        movies=dict(fps=6 if smoke else 12,max_seconds=3 if smoke else 30,
            scenarios=['aligned','kick_right50'],seed=19,phase=1,
            selection='Prespecified cases plus first scheduled failure per world/method; never select best outcomes'),
        comparison='UV ApiaViz versus visible-only Sobel/Ardin; different sensory inputs and stream allocations. All have 10000 KCs, same scene/teaching poses, memory, controller, seeds and sensor/movement budgets; spikes/compute are measured, not assumed matched.',
        limitation='Three independently seeded synthetic layouts in one habitat family; phases/seeds are not independent worlds. Descriptive world-level intervals only; no isolated preprocessing or UV causal claim.',
        collision='Swept projected rock meshes with 5mm body; block and continue. Safely clip releases/kicks; log all doses. Physics remains evaluator-only.',
        history='Fresh study; never resumes or modifies cancelled route-continuous-full or paused controller-full')
    p['trials']=len(list(cases(p)))
    return p


def blender_binary(explicit=None):
    candidates=[explicit,shutil.which('blender'),'/Applications/Blender.app/Contents/MacOS/Blender']
    for path in candidates:
        if path and Path(path).is_file(): return str(Path(path).resolve())
    raise FileNotFoundError('Blender not found; supply --blender /path/to/blender')


def blender_step(binary,script,arguments,log):
    with log.open('a') as handle:
        subprocess.run([binary,'--background','--factory-startup','--python-exit-code','1',
            '--python',str(ROOT/script),'--',*map(str,arguments)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)


def verify_environment(env):
    manifest=json.loads((env/'complete.json').read_text())
    for name,value in manifest['assets'].items():
        if file_sha(env/name)!=value: raise ValueError(f'Environment asset changed: {env/name}')
    return manifest


def environments(out,smoke=False,blender=None):
    with owned_lock(out):
        path=out/'protocol.json'
        if path.exists():
            p=json.loads(path.read_text()); verify_sources(p)
            if p['smoke']!=smoke: raise ValueError('Choose a different output for smoke/full protocols')
        else:
            p=configuration(smoke); atomic_json(path,p)
            with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in p['source_sha256']: archive.write(ROOT/name,name)
        binary=blender_binary(blender)
        manifest=dict(protocol_sha256=file_sha(path),status='preparing',worlds={})
        atomic_json(out/'environments.json',manifest)
        with tqdm(total=4*len(p['worlds']),desc='Build / collision / export / UV preview',unit='stage') as progress:
            for world in p['worlds']:
                env=out/'worlds'/world['name']
                if (env/'complete.json').exists():
                    verify_environment(env); progress.update(4)
                else:
                    if env.exists():
                        env.rename(env.with_name(env.name+'-interrupted-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
                    env.mkdir(parents=True)
                    atomic_json(env/'protocol.json',world)
                    progress.set_postfix_str(world['name'])
                    blender_step(binary,'scripts/build_uv_trial_world.py',['--output',env],env/'build.log'); progress.update(1)
                    blender_step(binary,'scripts/export_collision_geometry.py',['--scene',env/'grassland.blend','--output',env/'collision.json'],env/'build.log'); progress.update(1)
                    blender_step(binary,'scripts/uv_mitsuba/export_scene.py',['--environment',env,'--output',env/'geometry'],env/'build.log'); progress.update(1)
                    meta=json.loads((env/'world.json').read_text())
                    geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
                    if len(geometry.rocks)!=world['rock_count']: raise ValueError('Collision rock count mismatch')
                    for a,b in zip(world['route'][:-1],world['route'][1:]):
                        if geometry.intersects(a,b): raise ValueError('Generated teaching route intersects a rock')
                    releases={}
                    for case in p['scenarios']:
                        h=np.deg2rad(world['headings'][0]); vector=case['lateral']*np.array([-np.sin(h),np.cos(h)])
                        _,releases[case['name']]=safe_displacement(world['route'][0],vector,world['world_bounds_m'],geometry)
                    atomic_json(env/'release-preflight.json',releases)
                    shutil.copyfile(ROOT/'docs/uv-calibration/calibration.json',env/'calibration.json')
                    render=dict(p['render'],calibration_sha256=p['calibration_sha256'],
                        geometry_sha256={f.name:file_sha(f) for f in sorted((env/'geometry').iterdir())})
                    atomic_json(env/'render.json',render)
                    from .uv_trial_report import environment_preview
                    with DualCamera(env,render) as camera:
                        environment_preview(env,world,camera)
                    progress.update(1)
                    assets={str(f.relative_to(env)):file_sha(f) for f in env.rglob('*') if f.is_file() and f.name!='build.log'}
                    # Runtime logs and Matplotlib caches are not immutable assets.
                    assets={name:value for name,value in assets.items() if not name.endswith('.log') and '.mpl-cache' not in Path(name).parts}
                    atomic_json(env/'complete.json',dict(assets=assets,scene_sha256=meta['scene_sha256'],release_preflight=releases))
                manifest['worlds'][world['name']]=file_sha(env/'complete.json')
                atomic_json(out/'environments.json',manifest)
        manifest['status']='complete'; atomic_json(out/'environments.json',manifest)
        (out/'environments.html').write_text('<!doctype html><meta charset="utf-8"><title>Trial environments</title><h1>Fresh UV / visible trial worlds</h1>'+''.join(
            f'<h2>{w["name"]}</h2><p><a href="worlds/{w["name"]}/grassland.blend">Blender scene</a> · <a href="worlds/{w["name"]}/complete.json">Assets and release checks</a></p><img style="max-width:100%" src="worlds/{w["name"]}/preview.png">' for w in p['worlds']))
        print(f'Prepared {len(p["worlds"])} new spectral-ready Blender worlds for {p["trials"]} trials.\nNext: pixi run trials --output {out}',flush=True)


class TrialScorer:
    """Visible camera for the shared reflex; spectral input only inside ApiaViz."""
    def __init__(self,camera,model,memory,method):
        self.world=camera; self.model=model; self.memory=memory; self.method=method
        self.decisions=[]; self.encoding_s=0.; self.active_spikes=None if getattr(model,'response_format',None)=='graded' else 0; self.encoded_views=0
        self.active_response_components=0

    def __call__(self,position,headings):
        images=self.world.scan(position,headings,uv=self.method=='apiaviz_uv')
        start=time.perf_counter(); codes=encode(self.model,images)
        values=self.memory(codes); self.encoding_s+=time.perf_counter()-start
        active=(codes>0); self.active_response_components+=int(active.sum()); self.encoded_views+=len(images)
        if self.active_spikes is not None:self.active_spikes+=int(active.sum())
        values=torch.where(codes.abs().sum(1)>0,values,float('inf'))
        self.decisions.append(dict(position=np.asarray(position).tolist(),headings=np.asarray(headings).tolist(),
            scores=[float(v) if torch.isfinite(v) else None for v in values],active_fraction=float(active.float().mean())))
        return values.numpy()


def make_model(p,method,seed):
    if method=='apiaviz_uv':
        if p['encoder'].get('representation')=='graded-angular-combined-v1':
            from .graded_navigation import GradedNavigationEncoder
            return GradedNavigationEncoder(seed=seed)
        return ApiaVizUVEncoder(UVEncoderConfig(seed=seed,**p['encoder'].get('uv_config',{})))
    return RefinementEncoder(method,seed=seed,code_dim=p['encoder']['baseline_code_dim'])


def make_memory(model,codes):
    from .graded_navigation import memory_and_calibration
    return memory_and_calibration(model,codes)


def teaching_views(env,world,camera,p):
    if p.get('acquisition')=='independent-teaching-recall-v2':
        # Prepared with a separate Monte Carlo seed and frozen in complete.json.
        manifest=json.loads((env/'complete.json').read_text())
        if file_sha(env/'teaching.pt')!=manifest['assets']['teaching.pt']:
            raise ValueError('Independent teaching images changed')
        bank=torch.load(env/'teaching.pt',weights_only=True,map_location='cpu')
        return bank['uv'],bank['rgb']
    teaching=[]
    for position,heading in tqdm(list(zip(world['route'][:-1],world['headings'][:-1])),desc=f'{world["name"]}: teaching views',leave=False):
        teaching.append((camera.scan(position,[heading],uv=True),camera.scan(position,[heading])))
    return torch.cat([v[0] for v in teaching]),torch.cat([v[1] for v in teaching])


def load_completed(out,p):
    result={}
    expected={c[0]:c for c in cases(p)}
    checked_frames=set(); checked_checkpoints={}
    for path in tqdm(sorted((out/'trials').glob('*.json')),desc='Verify saved trials',leave=False,unit='trial'):
        bundle=json.loads(path.read_text()); row=bundle['result']
        if row['id'] not in expected or row['id'] in result: raise ValueError('Unexpected/duplicate completed trial')
        if bundle['protocol_sha256']!=file_sha(out/'protocol.json'): raise ValueError('Trial protocol mismatch')
        key,world,seed,method,scenario,phase=expected[row['id']]
        if (row['world'],row['seed'],row['method'],row['scenario'],row['phase'])!=(world['name'],seed,method,scenario['name'],phase):
            raise ValueError('Saved trial condition mismatch')
        if row['trace']!=f'trials/{key}.json' or path.name!=f'{key}.json': raise ValueError('Trace path mismatch')
        env=out/'worlds'/row['world']; checkpoint=env/f'encoder-{method}-{seed}.pt'
        if checkpoint not in checked_checkpoints: checked_checkpoints[checkpoint]=file_sha(checkpoint)
        if checked_checkpoints[checkpoint]!=row['checkpoint_sha256']: raise ValueError('Trial checkpoint changed')
        render_hash=digest(json.loads((env/'render.json').read_text()))
        for camera_key,sha in bundle['camera_frames'].items():
            camera_path=env/'camera'/f'{camera_key}.json'
            if file_sha(camera_path)!=sha: raise ValueError('Saved camera provenance changed')
            if camera_path not in checked_frames:
                load_cached(env/'camera',camera_key,render_hash); checked_frames.add(camera_path)
        result[row['id']]=row
    return result


def audit_trial(row,detail,p,world,geometry):
    case=next(s for s in p['scenarios'] if s['name']==row['scenario'])
    h=np.deg2rad(world['headings'][0]); start=np.array(world['route'][0])+case['lateral']*np.array([-np.sin(h),np.cos(h)])
    check_motion(start,detail,row,p,world,geometry,case)
    observations=[e for e in detail['events'] if e['kind'] in ('observation','avoidance_observation')]
    elapsed=sum(e['duration_s'] for e in detail['events'] if e['kind']=='turn')+row['commanded_path_m']/p['sensor_settings']['speed_m_s']+len(observations)*p['sensor_settings']['observation_s']
    assert abs(elapsed-row['time_s'])<1e-7 and len(observations)==row['observations']
    assert row['observations']<=p['sensor_settings']['observation_budget']
    assert row['time_s']<=p['sensor_settings']['time_budget_s']+1e-8
    assert row['corrective_resets']==row['motor_state_resets']==0
    assert row['evaluation_safety']==p['evaluation_safety']
    assert len(detail['sensor'])==sum(e['kind']=='observation' for e in detail['events'])


def navigation_components(p,controller=None,evaluator=None):
    controller=FamiliarityController if controller is None else controller
    evaluator=evaluate if evaluator is None else evaluator
    if p.get('navigation_revision')=='direction-before-cast-v1':
        from .confirmed_exploration import ConfirmedExplorationController
        from .coherent_navigation import evaluate as coherent_evaluate
        return ConfirmedExplorationController,coherent_evaluate
    if p.get('navigation_revision') is not None:raise ValueError('Unknown navigation revision')
    return controller,evaluator


def trials(out,movies=True):
    with owned_lock(out):
        p=json.loads((out/'protocol.json').read_text()); verify_sources(p)
        controller_class,evaluate_trial=navigation_components(p)
        environment_manifest=json.loads((out/'environments.json').read_text())
        if environment_manifest['status']!='complete' or environment_manifest['protocol_sha256']!=file_sha(out/'protocol.json'):
            raise ValueError('Run pixi run environments to completion first')
        if movies and shutil.which('ffmpeg') is None: raise FileNotFoundError('FFmpeg is required for the final movies')
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        if p['versions']!=dict(torch=str(torch.__version__),numpy=np.__version__): raise ValueError('Model environment versions changed')
        (out/'trials').mkdir(exist_ok=True); completed=load_completed(out,p); resumed=set(completed)
        manifest=dict(status='running',completed=len(completed),planned=p['trials'])
        atomic_json(out/'progress.json',manifest)
        try:
            with tqdm(total=p['trials'],initial=len(completed),desc='Navigation trials',unit='trial') as total:
                for world in p['worlds']:
                    env=out/'worlds'/world['name']; verify_environment(env)
                    if file_sha(env/'complete.json')!=environment_manifest['worlds'][world['name']]: raise ValueError('World manifest changed')
                    meta=json.loads((env/'world.json').read_text())
                    geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
                    jobs=[c for c in cases(p) if c[1]['name']==world['name'] and c[0] not in completed]
                    if not jobs: continue
                    render=json.loads((env/'render.json').read_text())
                    with DualCamera(env,render) as camera:
                        uv_images,rgb_images=teaching_views(env,world,camera,p)
                        if p.get('acquisition')!='independent-teaching-recall-v2':
                            torch.save(dict(uv=uv_images,rgb=rgb_images),env/'teaching.pt')
                        loaded=None
                        for key,_,seed,method,scenario,phase in jobs:
                            if loaded!=(seed,method):
                                model=make_model(p,method,seed)
                                images=uv_images if method=='apiaviz_uv' else rgb_images
                                codes=encode(model,images); memory,calibration=make_memory(model,codes)
                                model_hash,memory_hash=fingerprint(model),fingerprint(memory)
                                training_hash=hashlib.sha256(images.numpy().tobytes()).hexdigest()
                                checkpoint=env/f'encoder-{method}-{seed}.pt'
                                state=dict(encoder=model.state_dict(),memory=memory.state_dict(),calibration=calibration,
                                    encoder_fingerprint=model_hash,memory_fingerprint=memory_hash,training_images_sha256=training_hash)
                                if checkpoint.exists():
                                    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
                                    for field in ('encoder_fingerprint','memory_fingerprint','training_images_sha256'):
                                        if saved[field]!=state[field]: raise ValueError('Checkpoint or teaching changed on resume')
                                    for part in ('encoder','memory'):
                                        if set(saved[part])!=set(state[part]) or not all(torch.equal(saved[part][k],v) for k,v in state[part].items()):
                                            raise ValueError('Checkpoint tensor content changed')
                                else:
                                    temporary=checkpoint.with_suffix('.pt.tmp'); torch.save(state,temporary); temporary.replace(checkpoint)
                                loaded=(seed,method)
                            scorer=TrialScorer(camera,model,memory,method)
                            factory=lambda:ControllerAdapter(controller_class(calibration,Settings(**p['controller_settings']),phase))
                            begin=time.perf_counter(); before_renders=camera.renders; before_hits=camera.cache_hits
                            with tqdm(total=p['evaluation']['max_steps'],desc=key,unit='move',leave=False) as movement:
                                def update(step,seconds,views):
                                    movement.update(max(0,step-movement.n)); movement.set_postfix(sim_s=f'{seconds:.1f}',views=views,rendered=camera.renders-before_renders)
                                result=evaluate_trial(world['route'],world['headings'],scorer,factory,scenario,p['sensor_settings'],p['evaluation'],world['world_bounds_m'],geometry,
                                    AvoidanceSettings(**p['avoidance_settings']),phase=phase,safety=EvaluationSafety(**p['evaluation_safety']),progress=update)
                            assert fingerprint(model)==model_hash and fingerprint(memory)==memory_hash
                            detail={k:result.pop(k) for k in ('trace','microtrace','events','decisions')}
                            row=dict(id=key,world=world['name'],seed=seed,method=method,scenario=scenario['name'],phase=phase,controller='familiarity',
                                trace=f'trials/{key}.json',**result,wall_time_s=time.perf_counter()-begin,
                                encoding_s=scorer.encoding_s,active_spikes=scorer.active_spikes,encoded_views=scorer.encoded_views,
                                active_response_components=scorer.active_response_components,
                                rendered_positions=camera.renders-before_renders,cache_hits=camera.cache_hits-before_hits,
                                encoder_fingerprint=model_hash,memory_fingerprint=memory_hash,training_images_sha256=training_hash,
                                checkpoint_sha256=file_sha(checkpoint))
                            detail['sensor']=scorer.decisions
                            audit_trial(row,detail,p,world,geometry)
                            frame_keys={camera.key(e['position']) for e in detail['events'] if e['kind'] in ('observation','avoidance_observation')}
                            atomic_json(out/row['trace'],dict(protocol_sha256=file_sha(out/'protocol.json'),result=row,
                                camera_frames={key:camera.references[key] for key in sorted(frame_keys)},**detail))
                            completed[key]=row; total.update(1); total.set_postfix(last=row['termination'])
                            manifest['completed']=len(completed); atomic_json(out/'progress.json',manifest)
                        atomic_json(env/'camera-usage.json',camera.references)
            # Re-audit resumed records too, before presenting a complete result.
            for world in p['worlds']:
                env=out/'worlds'/world['name']; meta=json.loads((env/'world.json').read_text())
                geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
                for row in completed.values():
                    if row['world']==world['name'] and row['id'] in resumed: audit_trial(row,json.loads((out/row['trace']).read_text()),p,world,geometry)
            assert len(completed)==p['trials']
            atomic_json(out/'audit.json',dict(passed=True,trials=len(completed),protocol_sha256=file_sha(out/'protocol.json'),
                traces={row['id']:file_sha(out/row['trace']) for row in completed.values()},
                checks=['Sources, calibration, scene and export hashes','Swept mesh/body walking and safe placement',
                    'Observation, time and commanded/actual movement accounting','No route or motor-state resets',
                    'Frozen encoder, teaching and memory','Raw linear camera frame hashes; resumed assets revalidated']))
            manifest['status']='reporting'; atomic_json(out/'progress.json',manifest)
            from .uv_trial_report import report
            report(out,p,list(completed.values()),movies=movies)
            manifest['status']='complete'; atomic_json(out/'progress.json',manifest)
            summary=json.loads((out/'report/summary.json').read_text())
            print('\nModel              Arrivals    Blocked trials    Adjusted perturbations')
            for group in summary['groups']:
                if group['world']=='all': print(f'{group["method"]:18} {group["arrivals"]:3}/{group["n"]:<3}     {group["blocked_trials"]:5}             {group["adjusted_perturbations"]:5}')
            print(f'Complete: {len(completed)} trials. Open {out/"report/index.html"}',flush=True)
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',error=str(exc),completed=len(completed))
            atomic_json(out/'progress.json',manifest)
            (out/'last-error.txt').write_text(traceback.format_exc())
            print(f'Completed trials are preserved in {out}. Repeat the same command to resume; inspect last-error.txt for errors.',file=sys.stderr)
            raise


def main():
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['environments','trials','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--smoke',action='store_true',help='Prepare a separate six-trial development fixture')
    parser.add_argument('--blender',help='Explicit Blender executable')
    parser.add_argument('--no-movies',action='store_true',help='Explicitly omit movie generation')
    args=parser.parse_args(); out=args.output.resolve()
    if args.smoke and out==DEFAULT: out=ROOT/'apiaviz/output/uv-trials-smoke-v1'
    if args.action=='environments': environments(out,args.smoke,args.blender)
    elif args.action=='trials':
        if args.smoke and not json.loads((out/'protocol.json').read_text())['smoke']:
            raise ValueError('--smoke cannot run a full-study protocol; use a separate output')
        trials(out,not args.no_movies)
    else:
        from .uv_trial_report import report
        with owned_lock(out):
            p=json.loads((out/'protocol.json').read_text()); verify_sources(p)
            report(out,p,list(load_completed(out,p).values()),movies=not args.no_movies)


if __name__=='__main__': main()
