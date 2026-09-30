"""Record two prespecified synthetic ApiaViz runs with camera-only mesh contact.

Fresh wiring and teaching memory; no historical checkpoints or outcomes reused.
All outcomes are saved and shown. The renderer is stopped when this command exits.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from apiaviz.mbant.config import NavigationConfig
from apiaviz.research.active_navigation import segment_collision
from apiaviz.research.collision_geometry import RockGeometry, SCHEMA
from apiaviz.research.familiarity_controller import FamiliarityController, ControllerAdapter, acquisition_calibration, Settings
from apiaviz.research.frontend_refinements import RefinementEncoder
from apiaviz.research.grassland_resolution import ResolutionWorld
from apiaviz.research.motor_feedback import evaluate
from apiaviz.research.navigation import Scorer
from apiaviz.research.route_full_audit import check_motion
from apiaviz.research.spike_overlap import SpikeOverlapMemory
from apiaviz.research.study import encode, fingerprint, file_hash, write_json
from apiaviz.research.visual_avoidance import Settings as AvoidanceSettings


def run(out):
    out=out.resolve(); out.mkdir(parents=True,exist_ok=False)
    env=out/'environment'; env.mkdir(); (env/'queue').mkdir(); (env/'panoramas').mkdir()
    route=np.column_stack([np.arange(-.1,1.501,.1),np.full(17,-.24)])
    base=dict(role='Synthetic video scene; geometry fixed before outcomes',geometry_seed=20260930,
        route=route.tolist(),headings=[0.]*len(route),world_bounds_m=[-.8,2.1,-1.1,1.1],
        nav=asdict(NavigationConfig()),
        renderer=dict(engine='Cycles CPU',samples=32,seed=20260930,panorama_size=[720,152],
                      latitude_bounds_deg=[-15.5,60.5],eye_height_m=.01),
        rocks=[dict(position=[.65,0],scale=[.24,.14,.16],legacy_radius_m=.3),
               dict(position=[-.45,.7],scale=[.14,.13,.24],legacy_radius_m=.36),
               dict(position=[.25,-.85],scale=[.18,.16,.20],legacy_radius_m=.30),
               dict(position=[1.4,.7],scale=[.24,.18,.35],legacy_radius_m=.525),
               dict(position=[1.8,-.75],scale=[.15,.12,.27],legacy_radius_m=.405)])
    p=dict(role='Two prespecified illustrative runs, not comparative research; all outcomes retained',
        encoder='RefinementEncoder linear_colour, fresh seed 19',seed=19,shape=[51,199],
        body_radius_m=.005,physics_schema=SCHEMA,contact_response='block',
        controller_settings=asdict(Settings()),avoidance_settings=asdict(AvoidanceSettings()),
        sensor_settings=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,time_budget_s=60.,observation_budget=1000),
        evaluation=dict(max_steps=40,kick_before_step=20,recovery_radius_m=.1,recovery_consecutive_steps=3),
        scenarios=[dict(name='clear-lane',lateral=0.,heading=0.,kick=0.),
                   dict(name='toward-rock',lateral=.24,heading=0.,kick=0.)],
        teaching='16 centreline views at 10 cm spacing, original RGB input; no model weight training',
        note='UV model adaptation is not included; navigation receives RGB only')
    write_json(env/'protocol.json',base); write_json(out/'protocol.json',p)
    sources=list((ROOT/'apiaviz/research').glob('*.py'))+[Path(__file__),ROOT/'scripts/navigation_demo_world.py',ROOT/'scripts/export_collision_geometry.py']
    manifest=dict(status='preparing',protocol_sha256=file_hash(out/'protocol.json'),
        sources={str(f.relative_to(ROOT)):file_hash(f) for f in sources})
    write_json(out/'manifest.json',manifest)
    log=(out/'renderer.log').open('w')
    process=subprocess.Popen(['/Applications/Blender.app/Contents/MacOS/Blender','--background','--factory-startup',
        '--python-exit-code','1','--python',str(ROOT/'scripts/navigation_demo_world.py'),'--','--output',str(env)],stdout=log,stderr=subprocess.STDOUT)
    try:
        started=time.monotonic()
        while not (env/'ready.json').exists():
            if process.poll() is not None: raise RuntimeError('Blender exited; inspect renderer.log')
            if time.monotonic()-started>90: raise TimeoutError('Scene setup timeout')
            time.sleep(.1)
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        world=ResolutionWorld(env,p['shape'])
        print('Rendering 16 teaching views',flush=True)
        images=world.render(route[:-1],np.zeros(len(route)-1))
        torch.save(images,out/'teaching.pt')
        model=RefinementEncoder(method='linear_colour',seed=p['seed']).eval()
        codes=encode(model,images); memory=SpikeOverlapMemory(codes)
        calibration=acquisition_calibration(codes.numpy())
        torch.save(dict(state_dict=model.state_dict(),seed=p['seed']),out/'encoder.pt')
        torch.save(memory.state_dict(),out/'memory.pt')
        metadata=json.loads((env/'world.json').read_text())
        geometry=RockGeometry.load(env/'collision.json',metadata['scene_sha256'],p['body_radius_m'])
        manifest.update(status='running',scene_sha256=metadata['scene_sha256'],collision_sha256=file_hash(env/'collision.json'),
            training_images_sha256=hashlib.sha256(images.numpy().tobytes()).hexdigest(),
            encoder_fingerprint=fingerprint(model),memory_fingerprint=fingerprint(memory),calibration=calibration)
        write_json(out/'manifest.json',manifest)
        rows=[]
        for case in p['scenarios']:
            print('RUN',case['name'],flush=True)
            scorer=Scorer(world,model,memory,encode,'clean',0.,base['geometry_seed'])
            factory=lambda:ControllerAdapter(FamiliarityController(calibration,Settings(),1))
            result=evaluate(route,np.zeros(len(route)),scorer,factory,case,p['sensor_settings'],p['evaluation'],
                base['world_bounds_m'],geometry,AvoidanceSettings(),phase=1)
            assert fingerprint(model)==manifest['encoder_fingerprint']
            assert fingerprint(memory)==manifest['memory_fingerprint']
            detail={k:result.pop(k) for k in ('trace','microtrace','events','decisions')}
            start=route[0]+[0,case['lateral']]
            check_motion(start,detail,result,p,base,geometry,case)
            assert result['time_s']<=p['sensor_settings']['time_budget_s']+1e-8
            assert result['observations']<=p['sensor_settings']['observation_budget']
            previous=start; old_disc_segments=0
            for move in detail['microtrace']:
                if move['stride_m']>0 and segment_collision(previous,move['position'],metadata['obstacles']): old_disc_segments+=1
                previous=np.asarray(move['position'])
            detail.update(sensor=scorer.decisions,start=start.tolist())
            write_json(out/f"{case['name']}.json",detail)
            row=dict(id=case['name'],trace=f"{case['name']}.json",old_disc_intersections=old_disc_segments,**result)
            rows.append(row); write_json(out/'results.json',rows)
            print(json.dumps(row),flush=True)
        manifest.update(status='complete',camera_sha256={f.name:file_hash(f) for f in (env/'panoramas').glob('*.png')})
        write_json(out/'manifest.json',manifest)
    finally:
        (env/'stop').touch()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=10)
        log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
