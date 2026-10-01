"""Compare every aligned trace's consecutive rejected commands and costs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from apiaviz.research.spectral_input import file_sha
from apiaviz.research.uv_trials import atomic_json


def measurements(detail):
    longest=run=0
    for step in detail['microtrace']:
        run=run+1 if step['blocked'] else 0
        longest=max(longest,run)
    result=detail['result']
    return dict(termination=result['termination'],blocked_proposals=result['blocked_proposals'],
        longest_blocked_streak=longest,commanded_path_m=result['commanded_path_m'],
        actual_path_m=result['path_length_m'],observations=result['observations'],time_s=result['time_s'],
        blocked_command_fraction=result['blocked_proposals']/max(1,len(detail['microtrace'])),
        motor_state_resets=result['motor_state_resets'],corrective_resets=result['corrective_resets'])


def run(old,new,out):
    if out.exists():raise FileExistsError('Preserve existing analysis')
    audit=json.loads((new/'audit.json').read_text())
    assert audit['passed'] and audit['trials']==18
    result=dict(script_sha256=file_sha(Path(__file__)),new_audit_sha256=file_sha(new/'audit.json'),pairs=[])
    for key,digest in audit['traces'].items():
        paths=[d/'trials'/f'{key}.json' for d in [old,new]]
        assert file_sha(paths[1])==digest
        details=[json.loads(p.read_text()) for p in paths]
        result['pairs'].append(dict(id=key,method=details[1]['result']['method'],
            old_sha256=file_sha(paths[0]),new_sha256=digest,old=measurements(details[0]),new=measurements(details[1])))
    result['max_consecutive']={variant:max(r[variant]['longest_blocked_streak'] for r in result['pairs']) for variant in ['old','new']}
    atomic_json(out,result)
    print(json.dumps(result['max_consecutive']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',type=Path,default=ROOT/'apiaviz/output/uv-validation-v2')
    p.add_argument('--new',type=Path,default=ROOT/'apiaviz/output/coherent-validation-v3')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.old,a.new,a.output)
