"""Preserve route-controller direction across a visual detour.

The first integration restarted the entire navigator after every detour. This
variant invalidates only the pending movement comparison and its trend/counters.
Navigation direction, search phase and scheduled checks remain continuous.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from . import avoidance_experiments as experiments
from .avoidance_navigation import evaluate as evaluate_with_avoidance
from .controller_experiments import setup
from .study import write_json

DEFAULT = Path('apiaviz/output/visual-avoidance-continuity')


class ContinuousFactory:
    def __init__(self, controller):
        self.controller = controller
        self.steps = 0
        self.previous = None

    def __call__(self):
        policy = self.controller.policy
        if self.previous is not None:
            self.previous.end = len(policy.decisions)
            policy.pending = None
            policy.trend = 0.
            policy.poor = policy.good = 0
        factory = self

        class Adapter:
            def __init__(self):
                self.start, self.end = len(policy.decisions), None

            @property
            def policy(self):
                return SimpleNamespace(decisions=policy.decisions[self.start:self.end])

            def step(self, sensor, ignored_episode_step):
                factory.steps += 1
                return factory.controller.step(sensor, factory.steps)

        self.previous = Adapter()
        return self.previous


def evaluate(route, headings, scorer, controller_factory, *args, **kwargs):
    return evaluate_with_avoidance(route, headings, scorer,
                                  ContinuousFactory(controller_factory()), *args, **kwargs)


def prepare(out, methods=None):
    if (out/'protocol.json').exists():
        raise FileExistsError('Choose a new output directory')
    p = json.loads((experiments.DEFAULT/'protocol.json').read_text())
    if methods:
        p['methods'] = methods
    p.update(trials=len(p['worlds'])*len(p['methods'])*len(p['scenarios']),
        integration='ContinuousFactory: preserve direction, phase and check schedule; clear pending comparison, trend and consecutive improvement/decline counters after a detour. Same visual avoidance settings.',
        changed='Same 5 mm/2 cm camera-only locomotion as the initial avoidance integration. Preserve the navigator motor state; invalidate only the temporal evidence affected by a detour.',
        role='Development integration correction after initial smoke revealed homing regression. No independent confirmation or significance claim.',
        parent_protocol=str(experiments.DEFAULT/'protocol.json'),
        comparison='Historical no-avoidance results change movement sampling, sensing costs and reflex together. Compare initial avoidance integration separately to study motor-state continuity.')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', p)
    setup(out)
    print(json.dumps(dict(trials=p['trials'], output=str(out))), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--methods',nargs='+',choices=['linear_colour','sobel_colour','ardin_input'])
    args=parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output,args.methods)
    elif args.action == 'run':
        # Reuse the exact frozen model/training/run pipeline, changing only its
        # explicit evaluator function. Separate protocol and source archive.
        experiments.evaluate = evaluate
        experiments.run(args.output)
    else:
        experiments.report(args.output)


if __name__ == '__main__':
    main()
