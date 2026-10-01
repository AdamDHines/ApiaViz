"""Parallel execution adapter for an existing, unchanged UV trial protocol.

Run: pixi run python -m apiaviz.research.uv_parallel --workers 6
The scheduler has separate, append-only execution provenance. Scientific source
hashes and completed records are never rewritten to accommodate this adapter.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
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

import numpy as np
import torch
from tqdm import tqdm

from . import uv_trials as base
from .uv_trials import (ROOT,DEFAULT,atomic_json,verify_sources,verify_environment,
    RockGeometry,DualCamera,TrialScorer,ApiaVizUVEncoder,UVEncoderConfig,
    RefinementEncoder,encode,SpikeOverlapMemory,fingerprint,acquisition_calibration,
    ControllerAdapter,FamiliarityController,Settings,AvoidanceSettings,
    EvaluationSafety,evaluate,audit_trial,file_sha,position_key,cases)


@contextmanager
def file_lock(path):
    with Path(path).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        yield


class SharedCamera(DualCamera):
    """One writer per exact pose, with the original renderer/cache validation."""
    def frame(self,position):
        key=self.key(position)
        if key in self.frames:
            return super().frame(position)
        with file_lock(self.cache/f'{key}.lock'):
            return super().frame(position)


def save_teaching(env,uv,rgb):
    with file_lock(env/'teaching.lock'):
        path=env/'teaching.pt'
        if path.exists():
            saved=torch.load(path,weights_only=True,map_location='cpu')
            if not torch.equal(saved['uv'],uv) or not torch.equal(saved['rgb'],rgb):
                raise ValueError('Existing teaching images changed')
        else:
            temporary=path.with_suffix('.pt.tmp')
            torch.save(dict(uv=uv,rgb=rgb),temporary); temporary.replace(path)


def batches(p,completed):
    """Exclusive checkpoint ownership, with worlds interleaved across workers."""
    groups={}
    for case in cases(p):
        key,world,seed,method,_,_=case
        if key not in completed:
            groups.setdefault((seed,method,world['name']),[]).append(key)
    return [groups[k] for k in sorted(groups)]


def interrupted(signum,frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def worker(out,execution,index):
    signal.signal(signal.SIGTERM,interrupted)
    record=json.loads(execution.read_text())
    if record['adapter_sha256']!=file_sha(Path(__file__)):
        raise ValueError('Parallel adapter changed after launch')
    p=json.loads((out/'protocol.json').read_text()); verify_sources(p)
    controller_class,evaluate_trial=base.navigation_components(p,FamiliarityController,evaluate)
    if record['protocol_sha256']!=file_sha(out/'protocol.json'):
        raise ValueError('Execution protocol changed')
    assigned=set(record['batches'][index]); jobs=[c for c in cases(p) if c[0] in assigned]
    if len(jobs)!=len(assigned) or len({(c[1]['name'],c[2],c[3]) for c in jobs})!=1:
        raise ValueError('Invalid worker assignment')
    if any((out/'trials'/f'{key}.json').exists() for key in assigned):
        raise ValueError('Worker assignment overlaps completed records')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    world=jobs[0][1]; env=out/'worlds'/world['name']; verify_environment(env)
    meta=json.loads((env/'world.json').read_text())
    geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
    render=json.loads((env/'render.json').read_text())
    status_path=execution.parent/f'worker-{index:02d}.json'; finished=[]
    atomic_json(status_path,dict(status='teaching',completed=0,pid=os.getpid()))
    with SharedCamera(env,render) as camera:
        uv_images,rgb_images=base.teaching_views(env,world,camera,p)
        save_teaching(env,uv_images,rgb_images)
        loaded=None
        for key,_,seed,method,scenario,phase in jobs:
            if loaded!=(seed,method):
                model=base.make_model(p,method,seed)
                images=uv_images if method=='apiaviz_uv' else rgb_images
                codes=encode(model,images); memory,calibration=base.make_memory(model,codes)
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
                    atomic_json(status_path,dict(status='running',trial=key,step=step,sim_s=seconds,views=views,completed=len(finished),pid=os.getpid()))
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
            atomic_json(out/row['trace'],dict(protocol_sha256=file_sha(out/'protocol.json'),execution_sha256=file_sha(execution),result=row,
                camera_frames={key:camera.references[key] for key in sorted(frame_keys)},**detail))
            finished.append(key)
            atomic_json(status_path,dict(status='running',trial=key,completed=len(finished),pid=os.getpid()))
    atomic_json(status_path,dict(status='complete',completed=len(finished),pid=os.getpid()))


def stop_workers(active):
    for process,handle in active.values():
        if process.poll() is None: process.terminate()
    deadline=time.monotonic()+20
    while any(process.poll() is None for process,_ in active.values()) and time.monotonic()<deadline:
        time.sleep(.1)
    for process,handle in active.values():
        if process.poll() is None: process.kill()
        process.wait(); handle.close()


def run(out,workers,movies=True):
    if workers<1: raise ValueError('At least one worker required')
    with base.owned_lock(out):
        p=json.loads((out/'protocol.json').read_text()); verify_sources(p)
        if p['schema']!='apiaviz-uv-trials-v1': raise ValueError('Only UV trial protocols are supported')
        if p['versions']!=dict(torch=str(torch.__version__),numpy=np.__version__):
            raise ValueError('Model environment versions changed')
        manifest=json.loads((out/'environments.json').read_text())
        if manifest['status']!='complete' or manifest['protocol_sha256']!=file_sha(out/'protocol.json'):
            raise ValueError('Environment preparation is incomplete')
        if movies and shutil.which('ffmpeg') is None: raise FileNotFoundError('FFmpeg required')
        for world in p['worlds']:
            env=out/'worlds'/world['name']; verify_environment(env)
            if file_sha(env/'complete.json')!=manifest['worlds'][world['name']]:
                raise ValueError('Environment manifest changed')
        (out/'trials').mkdir(exist_ok=True)
        completed=base.load_completed(out,p); jobs=batches(p,completed)
        directory=out/'parallel'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        directory.mkdir(parents=True)
        execution=directory/'execution.json'
        shutil.copyfile(__file__,directory/'adapter.py')
        atomic_json(execution,dict(schema='apiaviz-uv-parallel-v1',protocol_sha256=file_sha(out/'protocol.json'),
            adapter_sha256=file_sha(Path(__file__)),workers=workers,threads_per_worker=2,
            pid=os.getpid(),batches=jobs,prior_trials={r['id']:file_sha(out/r['trace']) for r in completed.values()},
            note='Scheduling and locked cache access only; original scientific source hashes remain enforced. Wall time/cache ownership depend on concurrency.'))
        status=dict(status='running',completed=len(completed),planned=p['trials'],workers=workers,
            coordinator_pid=os.getpid(),execution=str(execution.relative_to(out)))
        atomic_json(out/'progress.json',status)
        active={}; next_job=0
        try:
            with tqdm(total=p['trials'],initial=len(completed),desc='Parallel navigation',unit='trial') as bar:
                while next_job<len(jobs) or active:
                    while next_job<len(jobs) and len(active)<workers:
                        log=(directory/f'worker-{next_job:02d}.log').open('w')
                        try:
                            process=subprocess.Popen([sys.executable,'-m','apiaviz.research.uv_parallel',
                                '--output',str(out),'--execution',str(execution),'--worker-index',str(next_job)],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        except BaseException:
                            log.close(); raise
                        active[next_job]=(process,log); next_job+=1
                    for index,(process,log) in list(active.items()):
                        if process.poll() is not None:
                            log.close(); del active[index]
                            if process.returncode: raise RuntimeError(f'Worker {index} failed; inspect {directory/f"worker-{index:02d}.log"}')
                    count=len(list((out/'trials').glob('*.json')))
                    bar.update(max(0,count-bar.n)); bar.set_postfix(workers=len(active),refresh=False)
                    status.update(completed=count,active_workers=len(active))
                    atomic_json(out/'progress.json',status)
                    time.sleep(.5)
            status['status']='auditing'; atomic_json(out/'progress.json',status)
            completed=base.load_completed(out,p)
            if len(completed)!=p['trials']: raise ValueError('Workers did not complete the entire matrix')
            for world in p['worlds']:
                env=out/'worlds'/world['name']; meta=json.loads((env/'world.json').read_text())
                geometry=RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
                for row in tqdm([r for r in completed.values() if r['world']==world['name']],desc=f'Audit {world["name"]}',leave=False):
                    audit_trial(row,json.loads((out/row['trace']).read_text()),p,world,geometry)
            for key,sha in json.loads(execution.read_text())['prior_trials'].items():
                if file_sha(out/'trials'/f'{key}.json')!=sha: raise ValueError('Prior completed trial changed')
            atomic_json(out/'audit.json',dict(passed=True,trials=len(completed),protocol_sha256=file_sha(out/'protocol.json'),
                execution_sha256=file_sha(execution),traces={r['id']:file_sha(out/r['trace']) for r in completed.values()},
                checks=['Original source and environment hashes','Swept mesh/body safety and placement',
                    'Time/view/movement accounting','No resets','Checkpoint and raw camera hashes','Prior trial bytes preserved']))
            status['status']='reporting'; atomic_json(out/'progress.json',status)
            from .uv_trial_report import report
            report(out,p,list(completed.values()),movies=movies)
            status['status']='complete'; atomic_json(out/'progress.json',status)
            print(f'Complete: {len(completed)} trials. Open {out/"report/index.html"}',flush=True)
        except BaseException as exc:
            stop_workers(active)
            status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',
                completed=len(list((out/'trials').glob('*.json'))),active_workers=0,error=str(exc))
            atomic_json(out/'progress.json',status)
            (directory/'error.txt').write_text(traceback.format_exc())
            raise


def main():
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--no-movies',action='store_true')
    parser.add_argument('--execution',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--worker-index',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args(); out=args.output.resolve()
    if args.execution is not None:
        worker(out,args.execution.resolve(),args.worker_index)
    else: run(out,args.workers,not args.no_movies)


if __name__=='__main__': main()
