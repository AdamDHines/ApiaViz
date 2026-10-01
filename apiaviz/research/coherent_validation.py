"""Bounded matched aligned-route validation of the v3 controller repair.

Fresh protocol, unchanged v2 encoders/teaching/physics/budgets, all three methods,
three worlds and both phases (18 trials). No kick recovery or UV advantage claim.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from tqdm import tqdm

from . import uv_trials as base, uv_parallel as parallel
from .coherent_navigation import DirectionConfirmedController, evaluate
from .spectral_input import file_sha


def prepare(source, out):
    if out.exists(): raise FileExistsError('Use a fresh versioned output')
    out.mkdir(parents=True)
    p=deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(variant='confirmed-direction-coherent-steering-v3',
        protocol_revision='coherent-controller-validation-v3',source_sha256=base.sources(),
        parent_study=dict(path=str(source),protocol_sha256=file_sha(source/'protocol.json')),
        comparison='All 18 aligned cases from v2 repeated with identical encoders, teaching, memory, acquisition, seeds, phases, physics and budgets. Only continuous controller/steering integration changes.',
        limitation='Development validation on inspected worlds. No UV-benefit or displacement-recovery claim; no automatic larger study.',
        controller_variant=dict(direction='confirm decline with charged directional scan before casting',
            avoidance='coherent-image-steering-v1',stationary='textured-image-stall-recovery-v2'))
    p['scenarios']=[s for s in p['scenarios'] if s['name']=='aligned']
    p['trials']=len(list(base.cases(p)))
    p['readiness_rule']=dict(aligned_arrivals_per_model=4,aligned_trials_per_model=6,
                             larger_study_authorized=False)
    base.atomic_json(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in p['source_sha256']:z.write(base.ROOT/name,name)
    manifest=dict(status='complete',protocol_sha256=file_sha(out/'protocol.json'),worlds={})
    for w in tqdm(p['worlds'],desc='Verify and reuse frozen inputs'):
        old=source/'worlds'/w['name'];env=out/'worlds'/w['name']
        complete=base.verify_environment(old);env.mkdir(parents=True)
        for name in complete['assets']:
            target=env/name;target.parent.mkdir(parents=True,exist_ok=True)
            os.link(old/name,target)
        shutil.copyfile(old/'complete.json',env/'complete.json')
        manifest['worlds'][w['name']]=file_sha(env/'complete.json')
        # Verify each raw frame before hard-linking immutable cache inputs.
        render=json.loads((env/'render.json').read_text())
        (env/'camera').mkdir(exist_ok=True)
        for record in tqdm(list((old/'camera').glob('*.json')),desc=w['name']+' cache',leave=False):
            base.load_cached(old/'camera',record.stem,base.digest(render))
            for path in [record,record.with_suffix('.npz')]:
                target=env/'camera'/path.name
                if not target.exists():os.link(path,target)
    base.atomic_json(out/'environments.json',manifest)


def run(out, workers):
    if not 1<=workers<=6:raise ValueError('Use 1..6 workers for this bounded validation')
    with base.owned_lock(out):
        p=json.loads((out/'protocol.json').read_text());base.verify_sources(p)
        manifest=json.loads((out/'environments.json').read_text())
        if manifest['protocol_sha256']!=file_sha(out/'protocol.json'):raise ValueError('Environment protocol mismatch')
        (out/'trials').mkdir(exist_ok=True)
        completed=base.load_completed(out,p);jobs=parallel.batches(p,completed)
        directory=out/'parallel'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');directory.mkdir(parents=True)
        execution=directory/'execution.json'
        base.atomic_json(execution,dict(protocol_sha256=file_sha(out/'protocol.json'),
            adapter_sha256=file_sha(Path(parallel.__file__)),wrapper_sha256=file_sha(Path(__file__)),
            batches=jobs,workers=workers,variant=p['variant']))
        active={};index=0
        try:
            with tqdm(total=p['trials'],initial=len(completed),desc='Matched aligned validation',unit='trial') as bar:
                while index<len(jobs) or active:
                    while index<len(jobs) and len(active)<workers:
                        log=(directory/f'worker-{index:02d}.log').open('w')
                        process=subprocess.Popen([sys.executable,'-m','apiaviz.research.coherent_validation','worker',
                            '--output',str(out),'--execution',str(execution),'--index',str(index)],
                            cwd=base.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        active[index]=(process,log);index+=1
                    for i,(process,log) in list(active.items()):
                        if process.poll() is not None:
                            log.close();del active[i]
                            if process.returncode:raise RuntimeError(f'Worker failed: {directory}/worker-{i:02d}.log')
                    count=len(list((out/'trials').glob('*.json')));bar.update(max(0,count-bar.n))
                    base.atomic_json(out/'progress.json',dict(status='running',completed=count,planned=p['trials'],active_workers=len(active)))
                    time.sleep(.5)
        except BaseException:
            parallel.stop_workers(active)
            base.atomic_json(out/'progress.json',dict(status='interrupted',active_workers=0))
            raise
        rows=list(base.load_completed(out,p).values())
        if len(rows)!=p['trials']:raise ValueError('Incomplete validation')
        for w in p['worlds']:
            env=out/'worlds'/w['name'];meta=json.loads((env/'world.json').read_text())
            geometry=base.RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
            for row in [r for r in rows if r['world']==w['name']]:
                base.audit_trial(row,json.loads((out/row['trace']).read_text()),p,w,geometry)
        base.atomic_json(out/'audit.json',dict(passed=True,trials=len(rows),protocol_sha256=file_sha(out/'protocol.json'),
            traces={r['id']:file_sha(out/r['trace']) for r in rows}))
        from .uv_trial_report import report
        report(out,p,rows,movies=True)
        results=[dict(method=m,arrivals=sum(r['reached_nest'] for r in rows if r['method']==m),n=sum(r['method']==m for r in rows)) for m in p['methods']]
        base.atomic_json(out/'readiness.json',dict(results=results,aligned_gate_passed=all(r['arrivals']>=4 for r in results),
            ready_for_larger_study=False,reason='Displacement recovery and isolated controller ablations remain untested.'))
        base.atomic_json(out/'progress.json',dict(status='complete',completed=len(rows),planned=p['trials'],active_workers=0))
        print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('--source',type=Path,default=base.ROOT/'apiaviz/output/uv-validation-v2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--execution',type=Path);parser.add_argument('--index',type=int)
    a=parser.parse_args();signal.signal(signal.SIGTERM,parallel.interrupted)
    if a.action=='prepare':prepare(a.source.resolve(),a.output.resolve())
    elif a.action=='run':run(a.output.resolve(),a.workers)
    else:
        parallel.FamiliarityController=DirectionConfirmedController
        parallel.evaluate=evaluate
        parallel.worker(a.output.resolve(),a.execution.resolve(),a.index)
