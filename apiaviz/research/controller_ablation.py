"""Small mechanistic ablations on analytic fields; no policy fitting."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .active_navigation import evaluate
from .controller_synthetic import field
from .familiarity_controller import ControllerAdapter, FamiliarityController, Settings
from .study import write_json, file_hash


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    variants=dict(complete=Settings(),without_temporal=replace(Settings(),use_temporal=False),
        without_contrast=replace(Settings(),use_contrast=False),
        without_periodic_exploration=replace(Settings(),periodic_exploration=False))
    fields=['useful','heading_only','flat_high','appearance_step']
    write_json(out/'protocol.json',dict(variants={k:asdict(v) for k,v in variants.items()},
        fields=fields,phases=[-1,1],source_sha256=file_hash(Path(__file__)),
        scope='Controlled-field ablations only; do not substitute for rendered-world comparisons.'))
    route=np.column_stack([np.arange(0,6.01,.1),np.zeros(61)])
    sensor=dict(speed_m_s=.1,yaw_speed_deg_s=180.,observation_s=.05,time_budget_s=360.,observation_budget=2600)
    evaluation=dict(max_steps=150,kick_before_step=20,recovery_radius_m=.1,recovery_consecutive_steps=3)
    rows=[]
    for name in fields:
        for variant,settings in variants.items():
            for phase in (-1,1):
                policy=FamiliarityController(dict(scale=.3),settings,phase)
                scorer=lambda p,hs:-np.array([field(name,p,h) for h in hs])
                result=evaluate(route,np.zeros(60),scorer,ControllerAdapter(policy),
                    dict(lateral=.5,heading=0.,kick=0.),sensor,evaluation,[-10,20,-10,10])
                trace,events=result.pop('trace'),result.pop('events')
                write_json(out/f'{name}-{variant}-{phase}.json',dict(trajectory=trace,events=events,
                    decisions=policy.decisions,transitions=policy.transitions))
                rows.append(dict(field=name,variant=variant,phase=phase,**result))
    write_json(out/'results.json',rows)
    print('Completed',len(rows),'analytic ablations',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/familiarity-controller/ablations'))
    run(parser.parse_args().output)
