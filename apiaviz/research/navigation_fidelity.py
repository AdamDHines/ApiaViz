"""Fresh matched validation: detailed surfaces, integrated retina, checked casting.

Eighteen aligned full-route development trials; never auto-launch a full study.
The historical commands/protocols remain available and are not reinterpreted.
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

DEFAULT=base.ROOT/'apiaviz/output/navigation-fidelity-v1'


def protocol(source):
    p=deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(variant='surface-retina-direction-validation-v1',protocol_revision='navigation-fidelity-v1',
        navigation_revision='direction-before-cast-v1',acquisition='independent-teaching-recall-v2',
        parent_study=dict(path=str(source),protocol_sha256=file_sha(source/'protocol.json')),
        source_sha256=base.sources(),
        comparison='All models share new surfaces, retinal integration, teaching poses, memory, confirmed-direction controller, coherent avoidance and budgets. Only ApiaViz receives UV. Joint engineering validation, not attribution of an individual change.',
        limitation='18 aligned development trials on three previously inspected worlds. Not held-out evidence or displacement-recovery validation. Surface noise, spectral spatial modulation and BRDF/BTDF are declared approximations.',
        readiness_rule=dict(aligned_arrivals_per_model=4,aligned_trials_per_model=6,larger_study_authorized=False))
    p['seeds']=[19];p['controllers']=[dict(name='familiarity',phase=s) for s in (-1,1)]
    p['scenarios']=[s for s in p['scenarios'] if s['name']=='aligned']
    p['encoder']['uv_config']=dict(schema='apiaviz-uv-v2',response_half=1.,uv_enabled=True)
    p['render'].update(retinal_sampling='solid-angle-box-before-response-v1',
        surface_detail='source-informed-surfaces-v1',pose_decimals=8,seed_policy='common-v2',seed=20261002,
        visible_mapping='Linear solid-angle box integration then x/(x+1); identical footprint to UV.',
        uv_response='Linear solid-angle box integration; encoder applies x/(x+1).',
        display='Fixed sRGB transfer after response for presentation only; no autoexposure.')
    p['teaching_seed']=20261001
    for w in p['worlds']:w['surface_detail']='source-informed-surfaces-v1'
    p['trials']=len(list(base.cases(p)))
    if p['trials']!=18:raise ValueError('Require matched 3-world × 3-model × 2-phase aligned validation')
    return p


def geometry_equivalence(old,new):
    a=json.loads((old/'geometry.json').read_text());b=json.loads((new/'geometry.json').read_text())
    assert [m['material'] for m in a['meshes']]==[m['material'] for m in b['meshes']]
    for x,y in zip(a['meshes'],b['meshes']):
        with np.load(old/x['file']) as source,np.load(new/y['file']) as changed:
            for key in ('vertices','faces'):np.testing.assert_array_equal(source[key],changed[key])
            normals=changed['normals']
            assert np.isfinite(normals).all()
            np.testing.assert_allclose(np.linalg.norm(normals,axis=1),1,atol=2e-6)
    return dict(triangles=sum(m['triangles'] for m in b['meshes']),vertices_and_faces='exactly equal',
                source_geometry_sha256=file_sha(old/'geometry.json'),new_geometry_sha256=file_sha(new/'geometry.json'))


def prepare(source,out):
    if out.exists():raise FileExistsError('Use a fresh versioned output; preserve incomplete runs')
    torch.set_num_threads(2);out.mkdir(parents=True)
    p=protocol(source);base.atomic_json(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in p['source_sha256']:z.write(base.ROOT/name,name)
    binary=base.blender_binary();manifest=dict(status='preparing',protocol_sha256=file_sha(out/'protocol.json'),worlds={})
    base.atomic_json(out/'environments.json',manifest)
    for w in tqdm(p['worlds'],desc='Surface / retina environments',unit='world'):
        old=source/'worlds'/w['name'];base.verify_environment(old)
        env=out/'worlds'/w['name'];env.mkdir(parents=True)
        base.atomic_json(env/'protocol.json',w)
        base.blender_step(binary,'scripts/build_uv_trial_world.py',['--output',env],env/'build.log')
        base.blender_step(binary,'scripts/export_collision_geometry.py',['--scene',env/'grassland.blend','--output',env/'collision.json'],env/'build.log')
        base.blender_step(binary,'scripts/uv_mitsuba/export_scene.py',['--environment',env,'--output',env/'geometry','--surface-detail'],env/'build.log')
        base.atomic_json(env/'geometry-equivalence.json',geometry_equivalence(old/'geometry',env/'geometry'))
        meta=json.loads((env/'world.json').read_text())
        geometry=base.RockGeometry.load(env/'collision.json',meta['scene_sha256'],p['body_radius_m'])
        assert len(geometry.rocks)==w['rock_count']
        for a,b in zip(w['route'][:-1],w['route'][1:]):assert not geometry.intersects(a,b)
        shutil.copyfile(old/'calibration.json',env/'calibration.json')
        render=dict(p['render'],calibration_sha256=p['calibration_sha256'],
            geometry_sha256={f.name:file_sha(f) for f in (env/'geometry').iterdir()})
        base.atomic_json(env/'render.json',render)
        teacher=env/'teaching-camera';teacher.mkdir()
        shutil.copytree(env/'geometry',teacher/'geometry');shutil.copyfile(env/'calibration.json',teacher/'calibration.json')
        teaching_config=dict(render,seed=p['teaching_seed']);base.atomic_json(teacher/'render.json',teaching_config)
        uv=[];rgb=[]
        with DualCamera(teacher,teaching_config) as camera:
            for pos,h in tqdm(list(zip(w['route'][:-1],w['headings'][:-1])),desc=w['name']+' teaching',leave=False):
                uv.append(camera.scan(pos,[h],uv=True));rgb.append(camera.scan(pos,[h]))
        torch.save(dict(uv=torch.cat(uv),rgb=torch.cat(rgb)),env/'teaching.pt')
        from .uv_trial_report import environment_preview
        with DualCamera(env,render) as camera:environment_preview(env,w,camera)
        assets={str(f.relative_to(env)):file_sha(f) for f in env.rglob('*') if f.is_file() and not f.name.endswith(('.log','.lock')) and '.mpl-cache' not in f.parts}
        base.atomic_json(env/'complete.json',dict(assets=assets,scene_sha256=meta['scene_sha256'],
            parent_complete_sha256=file_sha(old/'complete.json'),teaching_seed=p['teaching_seed'],recall_seed=render['seed']))
        manifest['worlds'][w['name']]=file_sha(env/'complete.json');base.atomic_json(out/'environments.json',manifest)
    manifest['status']='complete';base.atomic_json(out/'environments.json',manifest)


def assess(out):
    p=json.loads((out/'protocol.json').read_text());rows=list(base.load_completed(out,p).values())
    audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['trials']==p['trials']==len(rows)
    assert audit['protocol_sha256']==file_sha(out/'protocol.json')
    for r in rows:assert audit['traces'][r['id']]==file_sha(out/r['trace'])
    results=[dict(method=m,arrivals=sum(r['reached_nest'] for r in rows if r['method']==m),
        n=sum(r['method']==m for r in rows),blocked=sum(r['blocked_proposals'] for r in rows if r['method']==m)) for m in p['methods']]
    base.atomic_json(out/'readiness.json',dict(results=results,aligned_gate_passed=all(r['arrivals']>=4 for r in results),
        ready_for_larger_study=False,reason='No automatic larger study; displacement recovery, isolated surface effects and independent environments remain unvalidated.'))
    print(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','assess'])
    p.add_argument('--source',type=Path,default=base.ROOT/'apiaviz/output/uv-validation-v2')
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.source.resolve(),a.output.resolve())
    elif a.action=='assess':assess(a.output.resolve())
    else:
        if not 1<=a.workers<=6:raise ValueError('Use 1..6 workers')
        from .uv_parallel import run
        run(a.output.resolve(),a.workers);assess(a.output.resolve())


if __name__=='__main__':main()
