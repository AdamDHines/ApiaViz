"""Re-run saved common-pose probes at a fixed 5% count per KC stream.

This additional explanatory analysis is outside the primary navigation protocol.
Ranking continuous drives also removes finite timing/binning. Persistence of an
effect is useful evidence, but cannot identify the unique causal effect of rate.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research.mechanism_control import MatchedCountEncoder
from apiaviz.research.mechanisms import ALL_METHODS,code_statistics,probe_summary
from apiaviz.research.spike_overlap import SpikeOverlapMemory
from apiaviz.research.study import encode,file_hash
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument("--ants",nargs="+",type=int,required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    index=json.loads((HERE/"runs.json").read_text());rows=[]
    out=HERE/("fixed-count-"+"-".join(map(str,args.ants))+".json")
    for ant in args.ants:
        matches=[ROOT/r for r in index["runs"] if (ROOT/r/f"ant-{ant}-images.pt").exists()]
        assert len(matches)==1
        run=matches[0];manifest=json.loads((run/"manifest.json").read_text())
        images=torch.load(run/f"ant-{ant}-images.pt",weights_only=True,map_location="cpu")
        geo=json.loads((run/f"ant-{ant}-geometry.json").read_text())
        for seed in manifest["settings"]["seeds"]:
            model=MatchedCountEncoder(seed=seed)
            model.load_state_dict(torch.load(run/f"encoder-{seed}.pt",weights_only=True,map_location="cpu")["state_dict"])
            for method in ALL_METHODS:
                model.method=method;codes=encode(model,images["training"])
                memory=SpikeOverlapMemory(codes)
                probe,_=probe_summary(model,memory,images["probes"],geo["probes"],np.array(geo["scan_offsets"]))
                rows.append(dict(ant=ant,seed=seed,preprocessing=method,**code_statistics(codes),**probe))
            print(f"Fixed-count probes: ant {ant}, seed {seed}",flush=True)
    out.write_text(json.dumps(dict(rows=rows,source_sha256={str(f.relative_to(ROOT)):file_hash(f) for f in (Path(__file__),ROOT/"apiaviz/research/mechanism_control.py")},
                                  scope=__doc__),indent=2)+"\n")

if __name__=="__main__":main()
