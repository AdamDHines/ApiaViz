"""Continuous route guidance with feedback about executed visual steering.

Familiarity before/after a detour is still a valid matched-gaze observation.
Treat it as evidence about the combined motor action, not a fictitious straight
cast. Route state, temporal evidence and bounded cast counters remain live.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import avoidance_experiments as pipeline
from . import route_avoidance as engine
from .controller_experiments import setup
from .study import write_json
from .visual_avoidance import VisualAvoidance, wrap


class FeedbackReflex(VisualAvoidance):
    active=False  # Never suspend the route controller.
    resume=None

    def __init__(self,settings,phase,*unused_episode_limits):
        super().__init__(settings,phase)
        self.commands=[]
        self.entries=self.exits=0
        self.avoiding=False
        self.path=0.
        self.transitions=[]

    def choose(self,desired_heading):
        heading,stride,record=super().choose(desired_heading)
        avoiding=record['state']=='avoid'
        if avoiding != self.avoiding:
            if avoiding: self.entries += 1
            else: self.exits += 1
            self.transitions.append(dict(kind='steering_start' if avoiding else 'steering_end',
                path_m=self.path,heading=float(desired_heading)))
        self.avoiding=avoiding
        record['behaviour']='visual_steering' if avoiding else 'route'
        return heading,stride,record

    def moved(self,before,after,heading,distance):
        diagnostic=super().moved(before,after,heading,distance)
        self.commands.append(dict(heading=float(wrap(heading)),distance_m=float(distance)))
        self.path += distance
        return diagnostic


class FeedbackNavigator(engine.RouteNavigator):
    def __init__(self,policy,reflex):
        super().__init__(policy)
        self.reflex=reflex

    def step(self,sensor,distance,locomotion_step):
        commands=self.reflex.commands
        self.reflex.commands=[]
        feedback=None
        if self.policy.pending is not None:
            if not commands: raise ValueError('A temporal comparison requires executed motor commands')
            requested=self.policy.pending['movement']
            vectors=np.array([[np.cos(np.deg2rad(c['heading'])),np.sin(np.deg2rad(c['heading']))]
                              for c in commands])
            vector=np.sum(vectors*np.array([c['distance_m'] for c in commands])[:,None],axis=0)
            actual=float(np.rad2deg(np.arctan2(vector[1],vector[0])))
            self.policy.pending['movement']=actual
            feedback=dict(requested_heading=float(wrap(requested)),net_heading=actual,
                net_displacement_m=float(np.linalg.norm(vector)),
                walked_m=float(sum(c['distance_m'] for c in commands)),commands=commands,
                diverted=any(abs(wrap(c['heading']-requested))>1e-8 for c in commands))
        count=len(self.policy.decisions)
        previous_step=self.policy.decisions[-1].get('locomotion_step') if count else None
        result=super().step(sensor,distance,locomotion_step)
        if len(self.policy.decisions)==count and count:
            # A budget-limited observation can return before a new decision is
            # recorded. Do not attach this interval to the preceding decision.
            if previous_step is None: self.policy.decisions[-1].pop('locomotion_step',None)
            else: self.policy.decisions[-1]['locomotion_step']=previous_step
        if feedback and len(self.policy.decisions)>count:
            self.policy.decisions[-1]['motor_feedback']=feedback
        return result


def evaluate(*args,**kwargs):
    """Use the identical evaluator with explicitly scoped strategy replacement.

    Research variants run in separate processes. No world/model/training code
    is replaced, and the bindings are restored even if a trial raises.
    """
    context={}
    class Reflex(FeedbackReflex):
        def __init__(self,*a,**k):
            super().__init__(*a,**k)
            context['reflex']=self
    class Navigator(FeedbackNavigator):
        def __init__(self,policy): super().__init__(policy,context['reflex'])
    with patch.multiple(engine,DetourReflex=Reflex,RouteNavigator=Navigator):
        return engine.evaluate(*args,**kwargs)


def prepare(out):
    if (out/'protocol.json').exists(): raise FileExistsError('Use a fresh output')
    p=json.loads(Path('apiaviz/output/route-detour/protocol.json').read_text())
    p.update(integration_mode='motor_feedback',
        role='Development test of continuous familiarity guidance during image-driven evasive steering.',
        changed='No navigator restart or suspension. Matched-gaze familiarity changes describe the actual combined route/reflex movement. Its command path and net direction are recorded; cast counters continue normally. Original 16 cm reflex settings retained.',
        comparison='Compare with frozen detour/sampling protocols and previous restart integration. Same training, memory, flow settings, physics and budgets.',
        integration_settings={},
        limitation='Further development after first detour trial failed. No held-out or statistical significance claim.')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',p)
    setup(out)
    print(json.dumps(dict(trials=p['trials'],output=str(out))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,default=Path('apiaviz/output/route-motor-feedback-v2'))
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output)
    elif args.action=='run':
        pipeline.evaluate=evaluate
        pipeline.run(args.output)
    else: pipeline.report(args.output)
