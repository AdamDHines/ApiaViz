"""Controlled fields diagnose policy behaviour, not rendered-world performance."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .active_navigation import evaluate
from .familiarity_controller import ControllerAdapter, FamiliarityController, Settings
from .study import write_json


def field(name, position, heading):
    x, y = position
    angular = .3*np.cos(np.radians(heading))
    spatial = .6*np.exp(-y*y/1.2)
    if name == 'flat_high': return .95
    if name == 'flat_low': return .05
    if name == 'heading_only': return .6+angular
    if name == 'broad_angular': angular = .03*np.cos(np.radians(heading))
    if name == 'false_maximum': spatial += .45*np.exp(-(y-1.0)**2/.03)
    if name == 'delayed_gradient': spatial = .6*np.exp(-max(abs(y)-.25,0)**2/1.2)
    value = spatial+angular
    if name == 'score_offset': value -= .4
    if name == 'score_gain': value *= 2
    if name == 'appearance_step': value -= .3 if x > 2 else 0
    if name == 'appearance_drift': value -= .02*x
    return float(value)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    route = np.column_stack([np.arange(0,6.01,.1), np.zeros(61)])
    sensor = dict(speed_m_s=.1, yaw_speed_deg_s=180., observation_s=.05,
                  time_budget_s=360., observation_budget=2600)
    evaluation = dict(max_steps=150, kick_before_step=20, recovery_radius_m=.1,
                      recovery_consecutive_steps=3)
    names = ['useful','flat_high','flat_low','heading_only','broad_angular',
             'false_maximum','delayed_gradient','score_offset','score_gain',
             'appearance_step','appearance_drift']
    write_json(out/'protocol.json',dict(fields=names, phases=[-1,1],
        settings=asdict(Settings()), sensor=sensor, evaluation=evaluation,
        limitations='Analytic scalar fields only. Appearance changes here are score changes, not rendered lighting changes.'))
    rows = []
    for name in names:
        for phase in (-1,1):
            policy = FamiliarityController(dict(scale=.6 if name=='score_gain' else .3), phase=phase)
            scorer = lambda p, hs: -np.array([field(name,p,h) for h in hs])
            result = evaluate(route, np.zeros(60), scorer, ControllerAdapter(policy),
                dict(lateral=.5,heading=0.,kick=0.),sensor,evaluation,[-10,20,-10,10])
            trace,events = result.pop('trace'),result.pop('events')
            write_json(out/f'{name}-{phase}.json',dict(trajectory=trace,events=events,
                decisions=policy.decisions,transitions=policy.transitions))
            rows.append(dict(field=name,phase=phase,**result))
    write_json(out/'results.json', rows)
    for row in rows:
        print(row['field'],row['phase'],row['termination'],row['recovered'],row['steps'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/familiarity-controller/synthetic'))
    run(parser.parse_args().output)
