"""Matched-substep reflex-on/off check at the original hairpin starting rock."""
import json
from pathlib import Path

import numpy as np

from .active_navigation import Observations, segment_collision
from .avoidance_navigation import VisualLocomotion
from .grassland_resolution import ResolutionWorld
from .study import file_hash, write_json


def run():
    out = Path('apiaviz/output/visual-avoidance-local')
    out.mkdir(parents=True, exist_ok=True)
    env = Path('apiaviz/output/navigation-regime/hairpin')
    base = json.loads((env/'protocol.json').read_text())
    old = json.loads(Path('apiaviz/output/familiarity-controller/protocol.json').read_text())
    rocks = json.loads((env/'world.json').read_text())['obstacles']
    angle = np.deg2rad(base['headings'][0])
    start = np.array(base['route'][0])+.5*np.array([-np.sin(angle),np.cos(angle)])
    world = ResolutionWorld(env,(51,199))
    results=[]
    for enabled in (True,False):
        sensor = Observations(None,start,base['headings'][0],old['sensor_settings'])
        motion = VisualLocomotion(sensor,
            lambda pos,yaw: world.scan(pos,[yaw])[0].permute(1,2,0).numpy(),
            base['world_bounds_m'], rocks)
        if not enabled:
            # Same camera processing and step schedule, but no evasive steering.
            def straight(desired):
                stride = .005 if motion.policy.frame_heading is None else .02
                return desired, stride, dict(state='reflex_off',points=len(motion.policy.points))
            motion.policy.choose = straight
        for _ in range(10):
            if not motion.advance(base['headings'][0])[0]:
                break
        points=[start]+[r['position'] for r in motion.trace]
        assert not any(segment_collision(a,b,rocks) for a,b in zip(points[:-1],points[1:]))
        row=dict(reflex=enabled, path_m=motion.path, termination=sensor.termination or 'completed_1m',
                 time_s=sensor.time, observations=sensor.count, substeps=len(motion.trace))
        write_json(out/f"{'on' if enabled else 'off'}.json",dict(**row,start=start.tolist(),
            heading=base['headings'][0],trace=motion.trace,events=sensor.events))
        results.append(row)
    write_json(out/'summary.json',dict(results=results,
        role='Local development check; fixed heading, no route training or homing claim.',
        scene_sha256=file_hash(env/'grassland.blend'),
        sources={str(f):file_hash(f) for f in [Path(__file__),Path('apiaviz/research/visual_avoidance.py'),Path('apiaviz/research/avoidance_navigation.py')]}))
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__':
    run()
