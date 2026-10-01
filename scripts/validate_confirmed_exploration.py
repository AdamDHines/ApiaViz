"""Two full-budget meander failure-case controls, both phases, no new tuning.

Same v2 UV encoding and v3 avoidance; only periodic casting becomes a charged
direction check. This is a selected mechanism check, not comparative validation.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apiaviz.research import uv_trials as base,uv_parallel as parallel
from apiaviz.research.coherent_navigation import evaluate
from apiaviz.research.confirmed_exploration import ConfirmedExplorationController
from apiaviz.research.spectral_input import file_sha


def run(source,out):
    if out.exists():raise FileExistsError('Use a fresh output')
    out.mkdir(parents=True);(out/'trials').mkdir()
    p=deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(variant='scheduled-direction-confirmation-v4-meander-control',
        protocol_revision='scheduled-direction-confirmation-v4',source_sha256=base.sources(),
        parent_study=dict(path=str(source),protocol_sha256=file_sha(source/'protocol.json')),
        methods=['apiaviz_uv'],worlds=[w for w in p['worlds'] if w['name']=='meander'],
        limitation='Two selected development failure cases, one world and one encoder. Neither generalization nor UV-benefit evidence.',
        comparison='Compare with the same two meander cases in coherent-validation-v3. The only additional change is that scheduled exploration checks direction before casting; thresholds, interval and resource budgets are unchanged.')
    p['scenarios']=[s for s in p['scenarios'] if s['name']=='aligned']
    p['trials']=len(list(base.cases(p)));assert p['trials']==2
    p['controller_variant']=dict(direction='confirm both declining familiarity and scheduled exploration with the existing charged scan',
        avoidance='coherent-image-steering-v1',stationary='textured-image-stall-recovery-v2')
    p.pop('readiness_rule',None)
    base.atomic_json(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in p['source_sha256']:z.write(ROOT/name,name)
    old=source/'worlds/meander';env=out/'worlds/meander';env.mkdir(parents=True)
    complete=base.verify_environment(old)
    for name in complete['assets']:
        target=env/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(old/name,target)
    shutil.copyfile(old/'complete.json',env/'complete.json')
    # Verified primary recall cache only; later running v3 frames are not copied.
    render=json.loads((env/'render.json').read_text());(env/'camera').mkdir(exist_ok=True)
    for record in (old/'camera').glob('*.json'):
        base.load_cached(old/'camera',record.stem,base.digest(render))
        for path in [record,record.with_suffix('.npz')]:
            target=env/'camera'/path.name
            if not target.exists():os.link(path,target)
    base.atomic_json(out/'environments.json',dict(status='complete',protocol_sha256=file_sha(out/'protocol.json'),worlds={'meander':file_sha(env/'complete.json')}))
    execution=out/'execution.json'
    base.atomic_json(execution,dict(protocol_sha256=file_sha(out/'protocol.json'),adapter_sha256=file_sha(Path(parallel.__file__)),
        wrapper_sha256=file_sha(Path(__file__)),batches=[[c[0] for c in base.cases(p)]],workers=1))
    parallel.FamiliarityController=ConfirmedExplorationController;parallel.evaluate=evaluate
    parallel.worker(out,execution,0)
    rows=list(base.load_completed(out,p).values());assert len(rows)==2
    base.atomic_json(out/'audit.json',dict(passed=True,trials=2,protocol_sha256=file_sha(out/'protocol.json'),
        traces={r['id']:file_sha(out/r['trace']) for r in rows},
        checks='Both traces passed the full worker audit; reload verifies checkpoints and all acquired raw-frame hashes.'))
    from apiaviz.research.uv_trial_report import report
    report(out,p,rows,movies=True)
    base.atomic_json(out/'results.json',dict(results=rows,ready_for_larger_study=False,
        limitation=p['limitation']))
    print(json.dumps([dict(id=r['id'],termination=r['termination'],blocked=r['blocked_proposals']) for r in rows],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'apiaviz/output/uv-validation-v2')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.source.resolve(),a.output.resolve())
