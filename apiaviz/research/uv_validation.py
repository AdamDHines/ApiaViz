"""Prepare and assess versioned, complete-route UV validation without a full study.

Preparation reuses verified geometry, never historical images or trial results.
Teaching and recall use independent Monte Carlo seeds; no controller tuning.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch
from tqdm import tqdm
from . import uv_trials as base
from .dual_camera import DualCamera
from .spectral_input import file_sha
from .route_full import cases

DEFAULT=base.ROOT/'apiaviz/output/uv-validation-v2'


def protocol(source,uv_off=False):
    p=deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(smoke=False,variant='uv-v2-uv-off-control' if uv_off else 'uv-v2-complete-route-validation',
        acquisition='independent-teaching-recall-v2',protocol_revision='uv-validation-v2',
        seeds=[19],controllers=[dict(name='familiarity',phase=h) for h in [1,-1]],
        source_sha256=base.sources(),versions=dict(torch=str(torch.__version__),numpy=np.__version__),
        parent_study=dict(path=str(source.resolve()),protocol_sha256=file_sha(source/'protocol.json')),
        history='New corrected-input development validation. Original trials remain unchanged.',
        limitation='Three previously inspected development worlds, one wiring seed, two phases. Validation is not an independent comparative study.',
        comparison='Same geometry, independent-seed teaching/recall, fixed controller, memory, seeds and budgets for all models. ApiaViz alone receives UV with bounded receptor response; visible-only baselines unchanged.',
        readiness_rule=dict(aligned_arrivals_per_model=4,aligned_trials_per_model=6,
            kick_arrivals_per_model=2,kick_trials_per_model=6,
            require_complete=True,require_audit=True,selection='Frozen before corrected route outcomes; same thresholds for every model. No automatic full-study launch.'))
    p['scenarios']=[s for s in p['scenarios'] if s['name'] in ['aligned','kick_right50']]
    # Validation cannot inherit the old four-move smoke ceiling.
    p['evaluation'].update(max_steps=200,kick_before_step=36)
    p['sensor_settings'].update(time_budget_s=360.,observation_budget=2600)
    p['encoder']['uv_config']=dict(schema='apiaviz-uv-v2',response_half=1.,uv_enabled=not uv_off)
    p['render'].update(pose_decimals=8,seed_policy='common-v2',seed=20261002,spp=64)
    p['render']['uv_response']='Raw empirical receptor radiance input; encoder applies fixed x/(x+1) receptor response before spatial/opponent features.'
    p['teaching_seed']=20261001
    p['movies'].update(fps=8,max_seconds=12)
    if uv_off:
        p['methods']=['apiaviz_uv']
        p['comparison']='Separate mechanistic control: identical corrected ApiaViz input/visible stream, UV stream disabled in teaching and recall. 2000 UV cells remain allocated but silent. Compare to matched v2 UV-on records; not a matched-active-spike comparison.'
    p['trials']=len(list(cases(p)))
    return p


def prepare(source,out,uv_off=False):
    torch.set_num_threads(2)
    with base.owned_lock(out):
        path=out/'protocol.json'
        if path.exists():
            p=json.loads(path.read_text());base.verify_sources(p)
            if bool(not p['encoder']['uv_config']['uv_enabled'])!=uv_off:raise ValueError('Output belongs to another variant')
        else:
            p=protocol(source,uv_off);base.atomic_json(path,p)
            with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
                for name in p['source_sha256']:archive.write(base.ROOT/name,name)
        manifest=dict(status='preparing',protocol_sha256=file_sha(path),worlds={});base.atomic_json(out/'environments.json',manifest)
        for world in tqdm(p['worlds'],desc='Prepare independent acquisitions'):
            original=source/'worlds'/world['name'];env=out/'worlds'/world['name']
            old=base.verify_environment(original)
            if (env/'complete.json').exists():base.verify_environment(env)
            else:
                if env.exists():raise FileExistsError(f'Incomplete preparation: preserve {env}, then use a fresh output')
                env.mkdir(parents=True)
                # Copy only frozen environment assets, not trial/checkpoint state.
                if json.loads((source/'protocol.json').read_text()).get('acquisition')==p['acquisition']:
                    for name in old['assets']:
                        if Path(name).parts[0]=='camera':continue
                        target=env/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original/name,target)
                else:
                    for name in ['grassland.blend','collision.json','world.json','protocol.json','calibration.json']:
                        shutil.copy2(original/name,env/name)
                    shutil.copytree(original/'geometry',env/'geometry')
                render=dict(p['render'],calibration_sha256=p['calibration_sha256'],
                    geometry_sha256={f.name:file_sha(f) for f in sorted((env/'geometry').iterdir())})
                base.atomic_json(env/'render.json',render)
                if not (env/'teaching.pt').exists():
                    teacher=env/'teaching-camera';teacher.mkdir()
                    shutil.copytree(env/'geometry',teacher/'geometry');shutil.copyfile(env/'calibration.json',teacher/'calibration.json')
                    teaching_config=dict(render,seed=p['teaching_seed']);base.atomic_json(teacher/'render.json',teaching_config)
                    uv=[];rgb=[]
                    with DualCamera(teacher,teaching_config) as camera:
                        for pos,h in tqdm(list(zip(world['route'][:-1],world['headings'][:-1])),desc=world['name']+' teaching',leave=False):
                            uv.append(camera.scan(pos,[h],uv=True));rgb.append(camera.scan(pos,[h]))
                    torch.save(dict(uv=torch.cat(uv),rgb=torch.cat(rgb)),env/'teaching.pt')
                # The copied teaching bank must have the intended independent seed.
                teacher_config=json.loads((env/'teaching-camera/render.json').read_text())
                if teacher_config!=dict(render,seed=p['teaching_seed']):raise ValueError('Teaching camera protocol mismatch')
                from .uv_trial_report import environment_preview
                with DualCamera(env,render) as camera:environment_preview(env,world,camera)
                assets={str(f.relative_to(env)):file_sha(f) for f in env.rglob('*') if f.is_file() and not f.name.endswith(('.log','.lock')) and '.mpl-cache' not in f.parts}
                base.atomic_json(env/'complete.json',dict(assets=assets,scene_sha256=old['scene_sha256'],
                    parent_complete_sha256=file_sha(original/'complete.json'),teaching_seed=p['teaching_seed'],recall_seed=render['seed']))
            manifest['worlds'][world['name']]=file_sha(env/'complete.json');base.atomic_json(out/'environments.json',manifest)
        manifest['status']='complete';base.atomic_json(out/'environments.json',manifest)
        print(f'Prepared {p["trials"]} full-budget validation trials at {out}',flush=True)


def assess(out):
    p=json.loads((out/'protocol.json').read_text());base.verify_sources(p)
    rows=list(base.load_completed(out,p).values());audit=json.loads((out/'audit.json').read_text())
    complete=len(rows)==p['trials'] and audit['trials']==len(rows) and audit['passed']
    if audit['protocol_sha256']!=file_sha(out/'protocol.json'):raise ValueError('Audit protocol mismatch')
    for row in rows:
        if audit['traces'][row['id']]!=file_sha(out/row['trace']):raise ValueError('Audited trace changed')
    by_method=[]
    for method in p['methods']:
        aligned=[r for r in rows if r['method']==method and r['scenario']=='aligned']
        kicked=[r for r in rows if r['method']==method and r['scenario']=='kick_right50']
        a=sum(r['reached_nest'] for r in aligned)
        full=[r for r in kicked if r['disturbance_applied'] and r['displacement'] is not None and
            not r['displacement']['adjusted'] and r['displacement']['applied_distance_m']>=.5-1e-8]
        k=sum(r['reached_nest'] for r in full)
        rule=p['readiness_rule']
        by_method.append(dict(method=method,aligned_arrivals=a,aligned_n=len(aligned),full_kick_arrivals=k,kick_n=len(kicked),full_kicks_applied=len(full),
            passed=(len(aligned)==rule['aligned_trials_per_model'] and len(kicked)==rule['kick_trials_per_model'] and
                a>=rule['aligned_arrivals_per_model'] and k>=rule['kick_arrivals_per_model'])))
    result=dict(protocol_sha256=file_sha(out/'protocol.json'),complete=complete,methods=by_method,
        ready_for_larger_study=bool(complete and all(r['passed'] for r in by_method)),
        rule=p['readiness_rule'],note='A failed gate is preserved, not tuned away. Passing is an engineering development check, not generalization evidence.')
    base.atomic_json(out/'readiness.json',result);print(json.dumps(result,indent=2));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','assess'])
    parser.add_argument('--source',type=Path,default=base.DEFAULT);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--uv-off',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();out=args.output.resolve()
    if args.action=='prepare':prepare(args.source.resolve(),out,args.uv_off)
    elif args.action=='assess':assess(out)
    else:
        from .uv_parallel import run
        run(out,args.workers);assess(out)


if __name__=='__main__':main()
