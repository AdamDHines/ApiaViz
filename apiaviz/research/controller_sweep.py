"""Run the frozen full controller matrix in independent per-world processes.

Only scheduling changes. The existing evaluator, policies, memory and renderer
are used without modification. Imported smoke trials retain their provenance.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .controller_experiments import prepare as prepare_matrix, setup
from .study import file_hash, write_json

DEFAULT = Path('apiaviz/output/controller-full')
SMOKE = Path('apiaviz/output/familiarity-controller')


def prepare(out):
    prepare_matrix(out, full=True)
    p = json.loads((out/'protocol.json').read_text())
    p['analysis'] = dict(
        primary=['arrival', 'sustained_return_after_intended_displacement'],
        comparisons=['new_vs_previous_active', 'new_vs_exhaustive', 'frontends_with_new_controller'],
        pairing='Average new-controller phases within world/seed/method/scenario, then average wiring seeds within world. Worlds are the independent units.',
        intervals='Exact enumeration of the 27 size-three world bootstrap samples, 95% percentile intervals. Descriptive with only three worlds; no significance claim.',
        main_population='All scheduled trials, including collisions, failure before a scheduled kick, and invalid imposed kicks.',
        sensitivity='Separately exclude an entire world/seed/scenario block across all methods and policies if any imposed kick enters a rock or exits the field. This is outcome-dependent selection, not the primary estimate.',
        retention='After first three consecutive post-disturbance positions within 10 cm, count renewed loss if three consecutive later positions exceed 20 cm.',
        unchanged='Controller settings, image resolution, teaching density, memory, speed, step size, budgets, scene geometry and lighting.',
        limitation='More seeds and scenarios describe robustness within these three worlds. They do not add independent environments.')
    p['role'] = 'Expanded fixed-controller development matrix; authorized after the initial smoke study.'
    p['scheduling'] = 'One process and append-only result stream per world; deterministic trials do not share writable state.'
    write_json(out/'protocol.json', p)
    _, manifest, _ = setup(out)
    old_manifest = json.loads((SMOKE/'manifest.json').read_text())
    for source, digest in old_manifest['source_sha256'].items():
        # Every source used for the smoke is unchanged, not just its policy.
        assert file_hash(Path(source)) == digest, f'Cannot import changed implementation: {source}'
    old_protocol = json.loads((SMOKE/'protocol.json').read_text())
    for key in ('shape','methods','modes','controller_settings','sensor_settings','evaluation','scene_sha256','checkpoint_sha256'):
        assert p[key] == old_protocol[key], f'Cannot import changed protocol: {key}'
    old_rows = [json.loads(s) for s in (SMOKE/'results.jsonl').read_text().splitlines()]
    imported = []
    for world in p['worlds']:
        shard = out/'worlds'/world['name']
        shard.mkdir(parents=True)
        q = dict(p, worlds=[world], trials=p['trials']//len(p['worlds']))
        write_json(shard/'protocol.json', q)
        setup(shard)
        selected = [r for r in old_rows if r['world'] == world['name']]
        with (shard/'results.jsonl').open('w') as f:
            for row in selected:
                shutil.copyfile(SMOKE/row['trace'], shard/row['trace'])
                f.write(json.dumps(row, allow_nan=False)+'\n')
                imported.append(dict(id=row['id'], source=str(SMOKE/row['trace']),
                                     trace_sha256=file_hash(SMOKE/row['trace'])))
    write_json(out/'imports.json', dict(source_manifest_sha256=file_hash(SMOKE/'manifest.json'),
        source_protocol_sha256=file_hash(SMOKE/'protocol.json'), trials=imported))
    manifest.update(imported_trials=len(imported), expected_trials=p['trials'])
    write_json(out/'manifest.json', manifest)
    print(json.dumps(dict(total=p['trials'], imported=len(imported), new=p['trials']-len(imported))), flush=True)


def collect(out):
    p, manifest, _ = setup(out)
    rows=[]
    for world in p['worlds']:
        shard=out/'worlds'/world['name']
        child=json.loads((shard/'manifest.json').read_text())
        assert child['status']=='complete', world['name']
        for name,digest in child['source_sha256'].items():
            assert manifest['source_sha256'][name]==digest, name
        selected=[json.loads(s) for s in (shard/'results.jsonl').read_text().splitlines()]
        assert len(selected)==p['trials']//len(p['worlds'])
        for row in selected:
            shutil.copyfile(shard/row['trace'],out/row['trace'])
        for path in shard.glob('*-calibration.json'):
            shutil.copyfile(path,out/path.name)
        rows.extend(selected)
    assert len(rows)==p['trials'] and len({r['id'] for r in rows})==len(rows)
    temporary=out/'results.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    temporary.replace(out/'results.jsonl')
    manifest.update(status='complete',trials=len(rows))
    write_json(out/'manifest.json',manifest)


def run(out):
    p,_,_=setup(out)
    jobs=[]
    try:
        for world in p['worlds']:
            shard=out/'worlds'/world['name']
            if json.loads((shard/'manifest.json').read_text())['status']=='complete':continue
            log=(out/f"{world['name']}.log").open('a')
            process=subprocess.Popen([sys.executable,'-m','apiaviz.research.controller_experiments',
                                      'run','--output',str(shard)],stdout=log,stderr=subprocess.STDOUT)
            jobs.append((world['name'],process,log))
        while any(process.poll() is None for _,process,_ in jobs):
            failed=[name for name,process,_ in jobs if process.poll() not in (None,0)]
            if failed:raise RuntimeError(f'World process failed: {failed}; inspect per-world logs')
            counts={w['name']:len((out/'worlds'/w['name']/'results.jsonl').read_text().splitlines()) for w in p['worlds']}
            print(json.dumps(dict(completed=sum(counts.values()),total=p['trials'],worlds=counts)),flush=True)
            time.sleep(10)
        if any(process.returncode for _,process,_ in jobs):raise RuntimeError('World process failed')
        collect(out)
        print('COMPLETE',flush=True)
    finally:
        for _,process,log in jobs:
            if process.poll() is None:process.terminate()
            process.wait()
            log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','collect'])
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    globals()[args.action](args.output)
