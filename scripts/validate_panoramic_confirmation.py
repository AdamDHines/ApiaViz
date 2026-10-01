"""Paid scan replay at two diagnosed corner views, all models and both phases.

This tests heading confirmation only, not complete-route success. It changes no
existing protocol or result and never supplies route geometry to a policy.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from apiaviz.research import uv_trials as base
from apiaviz.research.active_navigation import Observations
from apiaviz.research.confirmed_exploration import ConfirmedExplorationController
from apiaviz.research.panoramic_confirmation import PanoramicConfirmationController
from apiaviz.research.familiarity_controller import SensorView,Settings
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.spectral_input import file_sha


def run(study,output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(2)
    p=json.loads((study/'protocol.json').read_text());records=[];inputs={}
    for world,step in [('bend',24),('hairpin',54)]:
        path=study/'trials'/f'{world}-19-apiaviz_uv-aligned-familiarity--1.json'
        d=json.loads(path.read_text());inputs[str(path.relative_to(study))]=file_sha(path)
        previous=next(v for v in d['decisions'] if v['step']==step-1)
        pos=d['trace'][step-2]['position'];heading=previous['reference_heading']
        env=study/'worlds'/world;cfg=json.loads((env/'render.json').read_text())
        bank=torch.load(env/'teaching.pt',weights_only=True)
        camera=DualCamera(env,cfg)
        for method in p['methods']:
            model=base.make_model(p,method,19)
            codes=base.encode(model,bank['uv' if method=='apiaviz_uv' else 'rgb'])
            memory=base.SpikeOverlapMemory(codes);calibration=base.acquisition_calibration(codes.numpy())
            for phase in (-1,1):
                for cls in (ConfirmedExplorationController,PanoramicConfirmationController):
                    sensor=Observations(base.TrialScorer(camera,model,memory,method),pos,heading,p['sensor_settings'])
                    def observe(h):return sensor.observe(h),sensor.heading,sensor.time
                    def rotate(h):return sensor.rotate(h),sensor.heading,sensor.time
                    view=SensorView(heading,0,0,observe,rotate,lambda:None,lambda _:None)
                    policy=cls(calibration,Settings(**p['controller_settings']),phase)
                    scan=policy._scan(view,heading,wide=True)
                    assert scan is not None
                    assert sensor.count==len(scan['samples'])
                    assert sensor.time<=p['sensor_settings']['time_budget_s']
                    records.append(dict(world=world,recorded_step=step,method=method,phase=phase,
                        controller=cls.__name__,scan=scan,observations=sensor.count,
                        time_s=sensor.time,rotation_deg=sensor.rotation_deg))
        inputs.update({f'worlds/{world}/camera/{key}.json':value
                       for key,value in camera.references.items()})
    base.atomic_json(output,dict(protocol_sha256=file_sha(study/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),controller_sha256=file_sha(ROOT/'apiaviz/research/panoramic_confirmation.py'),
        inputs=inputs,records=records,
        limitation='Retrospective scan-only diagnostic at two selected failed-trial poses. No movement or new arrival result; full-route cost/benefit remains unvalidated.'))
    for r in records:
        print(r['world'],r['method'],r['phase'],r['controller'],r['scan']['target'],r['scan']['supported'],r['observations'],round(r['time_s'],3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=ROOT/'apiaviz/output/navigation-fidelity-v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.study,a.output)
