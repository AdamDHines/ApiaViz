"""Replay the first blocked pose in a saved trial as a labelled reflex control.

Uses a fixed commanded heading, not a route navigator. Neither policy receives
the archived contact event; it supplies only the evaluator's initial condition.
Both policies start with the same archived visual surface memory. All outcomes
are retained, with two passing-side phases and identical budgets.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apiaviz.research.active_navigation import Observations
from apiaviz.research.avoidance_navigation import VisualLocomotion
from apiaviz.research.collision_geometry import RockGeometry, sha256
from apiaviz.research.dual_camera import DualCamera
from apiaviz.research.evaluation_safety import EvaluationSafety, proposal_reason
from apiaviz.research.visual_avoidance import Settings, VisualAvoidance
from apiaviz.research.visual_stall_recovery import VisualStallRecovery
from apiaviz.research.uv_trials import atomic_json, sources


def run(study, trial, output, cache_source=None):
    if output.exists():
        raise FileExistsError('Use a fresh output')
    torch.set_num_threads(1)
    p = json.loads((study/'protocol.json').read_text())
    detail = json.loads(trial.read_text())
    event = next(e for e in detail['events'] if e['kind'] == 'blocked_proposal')
    world = next(w for w in p['worlds'] if w['name'] == detail['result']['world'])
    original = study/'worlds'/world['name']
    output.mkdir(parents=True)
    env = output/'environment'
    env.mkdir()
    for name in ['collision.json', 'grassland.blend', 'render.json', 'calibration.json']:
        shutil.copy2(original/name, env/name)
    shutil.copytree(original/'geometry', env/'geometry')
    config = json.loads((env/'render.json').read_text())
    if cache_source is not None:
        assert sha256(cache_source/'render.json') == sha256(env/'render.json')
        shutil.copytree(cache_source/'camera', env/'camera')
    atomic_json(output/'protocol.json', dict(kind='Reflex failure replay; not a route trial',
        source_trial=str(trial), source_trial_sha256=sha256(trial), source_sha256=sources(),
        control_schema=VisualStallRecovery.schema,
        cache_source=str(cache_source) if cache_source is not None else None,
        phases=[-1, 1], commanded_moves=10, move_length_m=.1, escape_distance_m=.1,
        sensor_settings=p['sensor_settings'], initial_event=event))
    geometry = RockGeometry.load(env/'collision.json', sha256(env/'grassland.blend'), p['body_radius_m'])
    rows = []
    with DualCamera(env, config) as camera:
        def image(position, heading):
            return camera.scan(position, [heading])[0].permute(1, 2, 0).numpy()
        for policy_class in (VisualAvoidance, VisualStallRecovery):
            for phase in (-1, 1):
                sensor = Observations(None, event['position'], event['heading'], p['sensor_settings'])
                motion = VisualLocomotion(sensor, image, world['world_bounds_m'], geometry,
                                          Settings(**p['avoidance_settings']), phase, safety=EvaluationSafety())
                policy = policy_class(Settings(**p['avoidance_settings']), phase)
                policy.points = np.asarray(event['visual_points']).reshape(-1, 2)
                policy.ages = np.asarray(event['visual_point_ages_m'])
                policy.frame_heading = event['visual_frame_heading']
                policy.side = event['decision'].get('side', 0)
                motion.policy = policy
                for _ in tqdm(range(10), desc=f'{policy_class.__name__} phase {phase}'):
                    if not motion.advance(event['heading'])[0]: break
                prior = np.asarray(event['position'])
                for step in motion.trace:
                    assert proposal_reason(prior, step['position'], world['world_bounds_m'], geometry) is None
                    prior = np.asarray(step['position'])
                distance = float(np.linalg.norm(sensor.position-event['position']))
                row = dict(policy=policy_class.__name__, phase=phase, distance_from_start_m=distance,
                           escaped=distance >= .1, path_m=motion.path, commanded_path_m=motion.commanded_path,
                           blocked_proposals=motion.blocked_proposals, observations=sensor.count,
                           time_s=sensor.time, termination=sensor.termination, safety_passed=True,
                           trace=f'{policy_class.__name__}-{phase}.json')
                atomic_json(output/row['trace'], dict(result=row, microtrace=motion.trace, events=sensor.events))
                rows.append(row)
    atomic_json(output/'summary.json', dict(protocol_sha256=sha256(output/'protocol.json'), results=rows,
        limitations='One selected failure pose, fixed desired heading, static deterministic camera; not route arrival or generalization evidence.'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-source', type=Path)
    args = parser.parse_args()
    run(args.study.resolve(), args.trial.resolve(), args.output.resolve(),
        args.cache_source.resolve() if args.cache_source is not None else None)
