"""Verify the follow-up history controls and summarize their paired scan effects."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research import uv_trials as base
from apiaviz.research.spectral_input import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    protocol=json.loads((a.study/'protocol.json').read_text());complete=json.loads((a.study/'complete.json').read_text())
    assert complete['protocol_sha256']==file_sha(a.study/'protocol.json')
    for name,sha in protocol['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    source=Path(protocol['source']);assert protocol['source_protocol_sha256']==file_sha(source/'protocol.json')
    rows=[];inputs={}
    for world,sha in complete['results'].items():
        path=a.study/f'{world}.json';assert file_sha(path)==sha
        d=json.loads(path.read_text());old=json.loads((source/world/'results.json').read_text())
        assert d['raw_records']==old['raw_records']
        assert d['teaching_sha256']==old['source_teaching_sha256']
        assert len(d['records'])==28*2*2*2*3*6
        rows.extend(d['records']);inputs[str(path)]=sha
    index={(r['world'],r['station'],r['kind'],r['seed'],r['direction'],r['variant'],r['mode'],r['readout']):r for r in rows}
    assert len(index)==len(rows)
    summary=[]
    for world in ['all','meander','bend','hairpin']:
        for variant,mode,readout in dict.fromkeys((r['variant'],r['mode'],r['readout']) for r in rows):
            for kind in ['all','taught','midpoint','offroute']:
                group=[r for r in rows if (world=='all' or r['world']==world) and (r['variant'],r['mode'],r['readout'])==(variant,mode,readout)
                    and (kind=='all' or kind==r['kind'] or (kind=='offroute' and '20' in r['kind']))]
                scan=[]
                for r in group:
                    if r['direction']!=1:continue
                    other=index[r['world'],r['station'],r['kind'],r['seed'],-1,variant,mode,readout]
                    scan.append(abs((r['heading']-other['heading']+180)%360-180))
                summary.append(dict(world=world,variant=variant,mode=mode,readout=readout,kind=kind,n=len(group),
                    heading_error_mean=float(np.mean([r['heading_error'] for r in group])),
                    scan_direction_change_mean=float(np.mean(scan)),within20=sum(r['heading_error']<=20+1e-6 for r in group)))
    base.atomic_json(a.output,dict(summary=summary,inputs=inputs,protocol_sha256=file_sha(a.study/'protocol.json'),
        script_sha256=file_sha(Path(__file__)),audit=dict(passed=True,records=len(rows)),limitation=protocol['limitation']))
    for r in summary:
        if r['world']=='all' and r['kind']=='all' and r['readout'] in ['graded','spikes']:
            print(r['variant'],r['mode'],r['readout'],round(r['heading_error_mean'],3),round(r['scan_direction_change_mean'],3))


if __name__=='__main__':main()
