"""Verify the separate UV removal controls and save compact paired summaries."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research import uv_trials as base
from apiaviz.research.spectral_input import file_sha
from summarize_visual_response_controls import aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--components',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    protocol=json.loads((a.components/'protocol.json').read_text())
    complete=json.loads((a.components/'complete.json').read_text())
    assert complete['protocol_sha256']==file_sha(a.components/'protocol.json')
    assert protocol['source_protocol_sha256']==file_sha(a.source/'protocol.json')
    for name,sha in protocol['source_sha256'].items():assert file_sha(ROOT/name)==sha
    rows=[];inputs={}
    for world,sha in complete['results'].items():
        path=a.components/f'{world}.json';assert file_sha(path)==sha
        d=json.loads(path.read_text());parent_path=a.source/world/'results.json'
        parent=json.loads(parent_path.read_text())
        assert d['raw_records']==parent['raw_records']
        assert d['teaching_sha256']==parent['source_teaching_sha256']
        assert len(d['records'])==28*3*3
        rows.extend(d['records']+[r for r in parent['records'] if r['condition']=='api_uv'])
        inputs[str(path)]=sha;inputs[str(parent_path)]=file_sha(parent_path)
    index={(r['condition'],r['world'],r['seed'],r['station'],r['kind']):r for r in rows}
    assert len(index)==len(rows)==84*3*4
    pairs=[]
    for name in protocol['masks']:
        arows=[r for r in rows if r['condition']=='api_uv' and '20' in r['kind']]
        brows=[index[name,r['world'],r['seed'],r['station'],r['kind']] for r in arows]
        change=np.array([y['grid5']['after_m']-x['grid5']['after_m'] for x,y in zip(arows,brows)])
        pairs.append(dict(control=name,n=len(change),closer=int((change < -1e-9).sum()),
            farther=int((change > 1e-9).sum()),equal=int((abs(change)<=1e-9).sum()),
            mean_projected_distance_change_m=float(change.mean())))
    summary=aggregate(rows)
    base.atomic_json(a.output,dict(summary=summary,paired_projected_distance=pairs,inputs=inputs,
        protocol_sha256=file_sha(a.components/'protocol.json'),script_sha256=file_sha(Path(__file__)),
        passed=True,unique_new_records=84*3*3,limitation=protocol['limitation']))
    for r in summary:
        if r['world']=='all' and r['grid']=='grid5':
            print(r['condition'],r['kind'],r['inward'],r['n'],'heading',round(r['heading_error_mean'],3),'5deg loss',round(r['mean_five_degree_loss'],3))


if __name__=='__main__':main()
