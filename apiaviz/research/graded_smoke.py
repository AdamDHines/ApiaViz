"""Nine prespecified full-route trials; no automatic exhaustive study."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import zipfile
from tqdm import tqdm
from . import uv_trials as base
from .spectral_input import file_sha


def protocol(source):
    p=deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(variant='graded-angular-UV-bounded-v3', protocol_revision='graded-navigation-smoke-v3-bounded',
        parent_study=dict(path=str(source),protocol_sha256=file_sha(source/'protocol.json')),
        source_sha256=base.sources(),
        comparison='Nine matched aligned route trials: three worlds, seed 19, positive search phase, three models. ApiaViz uses graded angular features and combined UV opponents; baselines retain visible-only binary codes. All share cosine template retrieval, teaching-only score calibration rule, teaching poses, camera acquisition, bounded sector reorientation, continuous avoidance, physics and budgets. Representation, sensory inputs and stream allocation differ; this is a joint engineering smoke test.',
        limitation='One trial per model per previously inspected world. No held-out, displacement-recovery, isolated UV-effect or statistical superiority claim. Exact route retracing is not required. No exhaustive run is authorized by this command.',
        readiness_rule=dict(arrivals_per_model=3,trials_per_model=3,larger_study_authorized=False))
    p['seeds']=[19];p['controllers']=[dict(name='familiarity',phase=1)]
    p['scenarios']=[s for s in p['scenarios'] if s['name']=='aligned']
    p['encoder'].update(representation='graded-angular-combined-v1',
        graded_config=dict(response='x/(x+1)',adaptation=False,opponents=['UV-(B+G)/2','B-G'],
            stream_normalization='equal L2',memory='max cosine of continuous nonnegative currents',
            spike_cost='not applicable; active response components recorded separately'))
    p['controller_settings'].update(scan_step=5.,scan_extents=[20.,60.])
    p.setdefault('controller_variant',{}).update(
        direction='Complete +/-60 degree reorientation; +/-10 degree tracking; five-degree spacing; angular-modulation contrast; full-circle scanning removed; all views and turns charged',
        recovery='coherent-image-steering-v2: retain visually stalled bearings until lateral commanded progress reaches the existing lookahead distance; avoidance course changes capped at 30 degrees')
    p['movies'].update(seed=19,phase=1,scenarios=['aligned'],selection='All nine prespecified trials, including every failure')
    p['trials']=len(list(base.cases(p)))
    assert p['trials']==9 and len(p['worlds'])==3 and len(p['methods'])==3
    return p


def prepare(source,out):
    if out.exists():raise FileExistsError('Use a fresh output; preserve prior results')
    out.mkdir(parents=True)
    p=protocol(source);base.atomic_json(out/'protocol.json',p)
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in p['source_sha256']:archive.write(base.ROOT/name,name)
    manifest=dict(status='preparing',protocol_sha256=file_sha(out/'protocol.json'),worlds={})
    base.atomic_json(out/'environments.json',manifest)
    for world in tqdm(p['worlds'],desc='Verify frozen scene and teaching inputs'):
        old=source/'worlds'/world['name'];env=out/'worlds'/world['name']
        complete=base.verify_environment(old);env.mkdir(parents=True)
        for name in complete['assets']:
            target=env/name;target.parent.mkdir(parents=True,exist_ok=True)
            os.link(old/name,target)
        shutil.copyfile(old/'complete.json',env/'complete.json')
        manifest['worlds'][world['name']]=file_sha(env/'complete.json')
        render=json.loads((env/'render.json').read_text());(env/'camera').mkdir(exist_ok=True)
        for record in tqdm(list((old/'camera').glob('*.json')),desc=world['name']+' verified cache',leave=False):
            base.load_cached(old/'camera',record.stem,base.digest(render))
            for path in (record,record.with_suffix('.npz')):
                target=env/'camera'/path.name
                if not target.exists():os.link(path,target)
    manifest['status']='complete';base.atomic_json(out/'environments.json',manifest)


def assess(out):
    p=json.loads((out/'protocol.json').read_text());base.verify_sources(p)
    rows=list(base.load_completed(out,p).values());audit=json.loads((out/'audit.json').read_text())
    assert audit['passed'] and audit['trials']==len(rows)==p['trials']==9
    assert audit['protocol_sha256']==file_sha(out/'protocol.json')
    for row in rows:assert audit['traces'][row['id']]==file_sha(out/row['trace'])
    results=[dict(method=m,arrivals=sum(r['reached_nest'] for r in rows if r['method']==m),n=3,
        blocked=sum(r['blocked_proposals'] for r in rows if r['method']==m)) for m in p['methods']]
    base.atomic_json(out/'readiness.json',dict(results=results,
        aligned_gate_passed=all(r['arrivals']==3 for r in results),ready_for_larger_study=False,
        reason='User review required; nine development trials do not validate displacement recovery or independent worlds.'))
    print(json.dumps(results,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','assess'])
    parser.add_argument('--source',type=Path,default=base.ROOT/'apiaviz/output/navigation-fidelity-v1')
    parser.add_argument('--output',type=Path,default=base.ROOT/'apiaviz/output/graded-navigation-smoke-v3-bounded')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source.resolve(),args.output.resolve())
    elif args.action=='assess':assess(args.output.resolve())
    else:
        if not 1<=args.workers<=6:raise ValueError('Use 1..6 workers')
        from .uv_parallel import run
        run(args.output.resolve(),args.workers);assess(args.output.resolve())


if __name__=='__main__':main()
