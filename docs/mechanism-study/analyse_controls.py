"""Summarise extra explanatory controls, without adding primary hypothesis tests."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apiaviz.research.mechanisms import ALL_METHODS
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
from analyse import LABELS,save

def main():
    paths=[HERE/"fixed-count-4-6-7-9.json",HERE/"fixed-count-10-12-13-15.json"]
    fixed=[r for path in paths for r in json.loads(path.read_text())["rows"]]
    protocol=json.loads((HERE/"protocol.json").read_text())
    expected={(a,s,m) for a in protocol["ants"] for s in protocol["wiring_seeds"] for m in ALL_METHODS}
    assert len(fixed)==440 and {(r["ant"],r["seed"],r["preprocessing"]) for r in fixed}==expected
    assert all(abs(r["memory_active_fraction"]-.05)<1e-7 for r in fixed)
    actual=json.loads((HERE/"statistics.json").read_text())["summary"]
    fields=("clean_heading_error_deg","clean_tangent_margin","clean_restoring_fraction",
            "brightness_heading_error_deg","brightness_heading_change_deg","neighbour_overlap","distant_overlap","effective_rank")
    summary=[]
    for m in ALL_METHODS:
        group=[r for r in fixed if r["preprocessing"]==m]
        real=next(r for r in actual if r["method"]==m)
        summary.append(dict(method=m,label=LABELS[m],fixed_count={k:float(np.mean([r[k] for r in group])) for k in fields},
                            finite_timing={k:real[k] for k in fields}))
    features=json.loads((HERE/"feature-geometry.json").read_text())["rows"]
    feature_summary=[]
    for m in ALL_METHODS:
        for stream in ("form","colour"):
            group=[r for r in features if r["method"]==m and r["stream"]==stream]
            assert len(group)==8
            feature_summary.append(dict(method=m,stream=stream,**{k:np.mean([r[k] for r in group],axis=0).tolist()
                for k in ("neighbour_cosine","distant_cosine","effective_rank","plane_energy_shares")}))
    (HERE/"controls-summary.json").write_text(json.dumps(dict(count_control=summary,feature_geometry=feature_summary,
         scope="Explanatory controls; no extra confirmatory p-values. Fixed-count ranking changes timing as well as spike budget. Pre-projection features do not vary across wiring seeds."),indent=2)+"\n")
    with (HERE/"fixed-count-results.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,list(fixed[0]));w.writeheader();w.writerows(fixed)
    selected=("apiaviz","gray","sobel_colour","no_hex","no_adapt","no_dog","apia_simple_colour","sobel_apia_colour")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.5))
    for i,m in enumerate(selected):
        row=next(r for r in summary if r["method"]==m)
        a,b=[row[k]["clean_heading_error_deg"] for k in ("finite_timing","fixed_count")]
        ax.plot([a,b],[i,i],color="gray",lw=1)
        ax.scatter(a,i,color="#087f8c",s=35,label="Finite spiking model" if i==0 else None)
        ax.scatter(b,i,color="#ce7b32",s=35,label="Fixed-count diagnostic" if i==0 else None)
        f=next(r for r in feature_summary if r["method"]==m and r["stream"]=="form")
        bx.scatter(f["neighbour_cosine"],i,color="#087f8c",s=35,label="Neighbouring views" if i==0 else None)
        bx.scatter(f["distant_cosine"],i,color="#ce7b32",s=35,label="Distant views" if i==0 else None)
        bx.plot([f["distant_cosine"],f["neighbour_cosine"]],[i,i],color="gray",lw=1)
    for axis in (ax,bx):axis.set(yticks=range(len(selected)),yticklabels=[LABELS[m] for m in selected]);axis.invert_yaxis()
    ax.set(xlabel="Centreline heading error (degrees)",title="A  Does equalising activity preserve the pattern?")
    bx.set(xlabel="Cosine similarity before KC projection",title="B  Form representation before random wiring")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=7)
    bx.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=7)
    fig.tight_layout(w_pad=2);save(fig,"controls")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":main()
