"""Freeze a fresh relocated continuous-navigation protocol with mesh physics.

Preparation only: never launches trials or renderer workers. Requires restored
scenes/checkpoints and separately exported collision geometry for every world.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from apiaviz.research.collision_geometry import RockGeometry, SCHEMA, sha256
from apiaviz.research.controller_experiments import setup
from apiaviz.research.evaluation_safety import EvaluationSafety, safe_displacement
import numpy as np


def prepare(args):
    source=Path(args.source).resolve()
    p=json.loads(source.read_text())
    if p.get('integration_mode') != 'motor_feedback':
        raise ValueError('Source must use continuous motor-feedback integration')
    if any(c['name'] != 'familiarity' or c['phase'] not in (-1,1) for c in p['controllers']):
        raise ValueError('Unsupported controller matrix for continuous navigation')
    if args.output.exists(): raise FileExistsError('Choose a fresh versioned output directory')
    supplied={name:(Path(env).resolve(),Path(mesh).resolve()) for name,env,mesh in args.world}
    if len(supplied) != len(args.world) or set(supplied) != {w['name'] for w in p['worlds']}:
        raise ValueError('Supply exactly one environment and collision export for every source world')
    assets={}
    for w in p['worlds']:
        env,mesh=supplied[w['name']]
        scene_hash=sha256(env/'grassland.blend')
        if scene_hash != p['scene_sha256'][w['name']]:
            raise ValueError('Restored scene differs from source protocol; prepare a new scene study explicitly')
        meta=json.loads((env/'world.json').read_text())
        if meta['scene_sha256'] != scene_hash: raise ValueError('World metadata scene mismatch')
        if sha256(env/'protocol.json') != w['protocol_sha256']:
            raise ValueError('Restored world protocol differs from source; relocation must not change teaching/scene settings')
        geometry=RockGeometry.load(mesh,scene_hash,args.body_radius_m,'block')
        if len(geometry.rocks) != len(meta['obstacles']): raise ValueError('Rock count differs from world metadata')
        base=json.loads((env/'protocol.json').read_text())
        # Validate and record all releases before any experiment is frozen.
        h=np.deg2rad(base['headings'][0])
        releases={}
        for case in p['scenarios']:
            _,record=safe_displacement(base['route'][0],case['lateral']*np.array([-np.sin(h),np.cos(h)]),
                                      base['world_bounds_m'],geometry)
            releases[case['name']]=record
        assets[w['name']]=dict(path=str(mesh),sha256=sha256(mesh),release_preflight=releases)
        w.update(environment=str(env),protocol_sha256=sha256(env/'protocol.json'))
    encoder=args.encoder_environment.resolve()
    for seed in p['seeds']:
        if sha256(encoder/f'encoder-{seed}.pt') != p['checkpoint_sha256'][str(seed)]:
            raise ValueError('Restored checkpoint differs from source protocol')
    p.update(encoder_environment=str(encoder),
        evaluation_safety=EvaluationSafety().configuration(),
        physics=dict(schema=SCHEMA,body_radius_m=args.body_radius_m,contact_response='block',worlds=assets,
            accounting='Commanded translation consumes time and post-command camera observation; rejected movement adds zero walked distance; no contact feedback to policies'),
        source_protocol=dict(path=str(source),sha256=sha256(source)),
        role='Separate navigation-safety-v1 mesh-contact protocol; not compatible with historical disc or mesh-v1 outcomes',
        report_directory=str((args.output/'report').resolve()),
        comparison='Same source teaching regime, memory, encoders, controller settings and budgets; updated evaluator safeguards and visual-clearance/zero-net-heading bug fixes',
        fixed='Original source parameter values except recorded physics, evaluation safety and relocated artifact paths; behavior uses newly frozen source hashes',
        advance='Prepared only; no trial or renderer started')
    p['analysis']['success']='20 cm endpoint radius checked after each executed substep; rock/field contact blocks translation and continues; report budget failures and every adjusted/skipped perturbation'
    p['analysis']['perturbation']='Sweep releases/kicks before applying; shorten only along requested direction. Preserve requested/applied doses; adjusted conditions must be stratified, not called full 50 cm perturbations.'
    p['analysis']['contact']='Report blocked proposals and trials with any blocked proposal, including trials that later arrive'
    args.output.mkdir(parents=True)
    def freeze(out,record):
        (out/'protocol.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        setup(out)
        manifest=out/'manifest.json'
        m=json.loads(manifest.read_text()); m['status']='prepared'
        manifest.write_text(json.dumps(m,indent=2)+'\n')
    freeze(args.output,p)
    report=args.output/'report'; report.mkdir()
    (report/'PROTOCOL.md').write_text('# Mesh-contact navigation protocol\n\n'
        'Prepared as a separate study; no outcomes exist yet.\n\n'
        '[Frozen settings and hashes](../protocol.json)\n')
    for w in p['worlds']:
        shard=args.output/'worlds'/w['name']; shard.mkdir(parents=True)
        q=deepcopy(p); q.update(worlds=[w],trials=p['trials']//len(p['worlds']))
        freeze(shard,q)
    print(json.dumps(dict(output=str(args.output),status='prepared',physics=p['physics'])))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--world',nargs=3,action='append',required=True,metavar=('NAME','ENVIRONMENT','COLLISION_JSON'))
    parser.add_argument('--encoder-environment',type=Path,required=True)
    parser.add_argument('--body-radius-m',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args())
