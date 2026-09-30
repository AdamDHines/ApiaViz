"""Frozen detour-handoff experiment and matched camera-sampling control."""
import argparse
from functools import partial
import json
from pathlib import Path

from . import avoidance_experiments as pipeline
from .controller_experiments import setup
from .route_avoidance import evaluate
from .study import write_json


def prepare(out,mode,methods=None):
    if (out/'protocol.json').exists(): raise FileExistsError('Use a new output or resume run')
    p=json.loads((pipeline.DEFAULT/'protocol.json').read_text())
    if methods: p['methods']=methods
    p.update(role='Development handoff experiment; fixed before outcomes, previous hairpin scene.',
        integration_mode=mode,integration_settings=dict(clear_m=.06,max_detour_m=2.),
        trials=len(p['worlds'])*len(p['methods'])*len(p['scenarios']),
        changed='Detour mode: hold intended direction until 6 cm of visually clear travel or a 2 m detour budget. Suspend route decisions during the detour, then check familiarity about the saved visual anchor; no whole-controller resets. Sampling control: same RGB acquisitions/flow computation and 5 mm initial/2 cm subsequent moves, without evasive steering.',
        comparison='Same teaching, models, memory, flow settings, physics and budgets as first avoidance smoke. Sampling control isolates the effect of additional camera acquisitions on an undiverted trajectory; action-dependent slowing differs when obstacles trigger a detour.',
        limitation='One development world, seed 19 and phase +1; no significance or held-out claim.',
        primary_endpoint='Nest arrival without physics corrections; report all failures and costs.')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',p)
    setup(out)
    print(json.dumps(dict(mode=mode,trials=p['trials'],output=str(out))),flush=True)


def run(out):
    p=json.loads((out/'protocol.json').read_text())
    pipeline.evaluate=partial(evaluate,mode=p['integration_mode'],**p['integration_settings'])
    pipeline.run(out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['detour','sampling_control'],default='detour')
    parser.add_argument('--methods',nargs='+',choices=['linear_colour','sobel_colour','ardin_input'])
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output,args.mode,args.methods)
    elif args.action=='run': run(args.output)
    else: pipeline.report(args.output)
