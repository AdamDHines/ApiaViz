"""Check camera-only interaction trials, including actual motor feedback."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

import numpy as np

from .avoidance_audit import audit as physics_audit
from .study import write_json
from .visual_avoidance import wrap


def audit(out):
    # Reuse source, image, collision and resource accounting, then check the
    # different semantics of continuous (rather than discarded) comparisons.
    with redirect_stdout(io.StringIO()): physics_audit(out)
    result=json.loads((out/'audit.json').read_text())
    result['checks']=[('One completed motion interval per familiarity comparison'
                       if c=='No pre-detour familiarity comparison reused' else c)
                      for c in result['checks']]
    p=json.loads((out/'protocol.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text().splitlines()]
    comparisons=diverted=handoffs=0
    for row in rows:
        d=json.loads((out/row['trace']).read_text())
        assert row['motor_state_resets']==0
        for decision in d['decisions']:
            if 'comparison' not in decision: continue
            c=decision['comparison']
            if p['integration_mode']!='motor_feedback': continue
            m=decision['motor_feedback']
            end=decision['self_motion_m']; start=end-c['distance_m']
            moves=[v for v in d['microtrace'] if start+1e-9 < v['path_m'] <= end+1e-9]
            assert len(moves)==len(m['commands'])
            for move,command in zip(moves,m['commands']):
                assert abs(wrap(move['heading']-command['heading']))<1e-8
                assert abs(move['stride_m']-command['distance_m'])<1e-10
            vector=np.sum([v['stride_m']*np.array([np.cos(np.deg2rad(v['heading'])),
                           np.sin(np.deg2rad(v['heading']))]) for v in moves],axis=0)
            assert abs(float(np.linalg.norm(vector))-m['net_displacement_m'])<1e-9
            assert abs(c['distance_m']-m['walked_m'])<1e-9
            assert abs(wrap(c['movement_heading']-m['net_heading']))<1e-8
            assert m['diverted']==any(abs(wrap(v['heading']-m['requested_heading']))>1e-8 for v in moves)
            comparisons+=1; diverted+=m['diverted']
        for event in d['events']:
            if event['kind']!='route_reacquisition': continue
            following=[v for v in d['decisions'] if v['time_s'] >= event['time_s']-1e-9]
            if following:
                assert 'comparison' not in following[0]
                handoffs+=1
        for transition in row.get('detour_transitions',[]):
            if transition['kind']=='detour_end':
                assert transition['detour_path_m'] <= p['integration_settings']['max_detour_m']+.02+1e-8
    result.update(motor_comparisons_checked=comparisons,diverted_comparisons_checked=diverted,
                  reacquisitions_checked=handoffs)
    result['checks'].append('Logged motor commands match executed substeps; no controller recreation')
    write_json(out/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
