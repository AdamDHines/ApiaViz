"""Analyse the fixed candidate matrix; all claims remain developmental."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apiaviz.research.statistics import paired_effect, holm_adjust
HERE=Path(__file__).parent
METHODS = tuple(json.loads((HERE/"protocol.json").read_text())["methods"])
LABELS=dict(zip(METHODS,["ApiaViz","Sobel + colour","Unit-gain low-pass","Balanced form","Oriented form","Linear colour","Balanced + linear colour"]))


def load():
    protocol=json.loads((HERE/"protocol.json").read_text());rows=[]
    for name in json.loads((HERE/"runs.json").read_text())["runs"]:
        run=ROOT/name;manifest=json.loads((run/"manifest.json").read_text())
        assert manifest["status"]=="complete" and manifest["protocol"]==protocol
        rows.extend(dict(json.loads(line),run=name) for line in (run/"results.jsonl").read_text().splitlines())
    assert len(rows)==protocol["navigation_trials"]
    assert {(r["ant"],r["seed"],r["preprocessing"]) for r in rows}=={(a,s,m) for a in protocol["ants"] for s in protocol["wiring_seeds"] for m in METHODS}
    return protocol,rows


def matrix(rows,protocol,method,field):
    lookup={(r["ant"],r["seed"]):r[field] for r in rows if r["preprocessing"]==method}
    return np.array([[lookup[a,s] for s in protocol["wiring_seeds"]] for a in protocol["ants"]],dtype=float)


def main():
    protocol,rows=load()
    comparisons={}
    for reference in METHODS[:2]:
        results=[]
        for method in METHODS[2:]:
            delta=100*(matrix(rows,protocol,method,"route_deviation_mean_m")-matrix(rows,protocol,reference,"route_deviation_mean_m"))
            stats=paired_effect(delta)
            stats.pop("routes_favouring_apia");stats.pop("seeds_favouring_apia")
            route_effects=np.asarray(stats["route_effects"])
            leave_one_out=(route_effects.sum()-route_effects)/(len(route_effects)-1)
            stats["leave_one_route_out_effect_range_cm"]=[float(leave_one_out.min()),float(leave_one_out.max())]
            results.append(dict(method=method,reference=reference,**stats,
                routes_with_lower_error=int((delta.mean(1)<0).sum()),seeds_with_lower_error=int((delta.mean(0)<0).sum())))
        for row,p in zip(results,holm_adjust([r["p_raw"] for r in results])):row["p_holm_exploratory"]=p
        comparisons[reference]=results
    fields=("route_deviation_mean_m","polyline_mean_m","first50_polyline_mean_m","deviation_median_m",
            "clean_heading_error_deg","clean_tangent_margin","clean_restoring_fraction","brightness_heading_change_deg",
            "memory_active_fraction","form_heading_error_deg","colour_heading_error_deg","form_tangent_margin",
            "colour_tangent_margin","form_matched_progress_error_m","colour_matched_progress_error_m")
    summary=[]
    for method in METHODS:
        group=[r for r in rows if r["preprocessing"]==method]
        summary.append(dict(method=method,label=LABELS[method],n=len(group),completed=sum(r["reached_nest"] for r in group),
                       **{key:float(np.mean([r[key] for r in group])) for key in fields}))
    branch_rows=[]
    for ant in protocol["ants"]:
        for seed in protocol["wiring_seeds"]:
            pair={m:next(r for r in rows if r["ant"]==ant and r["seed"]==seed and r["preprocessing"]==m) for m in METHODS[:2]}
            saved={m:json.loads((ROOT/r["run"]/r["trace"]).read_text()) for m,r in pair.items()}
            a,b=[saved[m]["trajectory"] for m in METHODS[:2]]
            first=next((i for i,(x,y) in enumerate(zip(a,b)) if x["heading"]!=y["heading"]),None)
            if first is not None:
                item=dict(ant=ant,seed=seed,first_different_step=first+1,
                          turn_difference_deg=abs(a[first]["heading"]-b[first]["heading"]))
                for m in METHODS[:2]:
                    score=np.sort(saved[m]["decisions"][first]["scores"])
                    item[m+"_score_gap"]=float(score[1]-score[0])
                branch_rows.append(item)
    result=dict(summary=summary,comparisons=comparisons,protocol=protocol,reference_first_disagreement=branch_rows,
                inference="Exploratory paired effects on known routes. Negative differences favour the candidate. Eight route-mean sign-flip tests condition on the five seeds; Holm over five candidates for each reference separately. Unadjusted crossed bootstrap intervals. No held-out or between-world significance claim.")
    (HERE/"statistics.json").write_text(json.dumps(result,indent=2)+"\n")
    with (HERE/"results.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,list(rows[0]));w.writeheader();w.writerows(rows)
    lines=["| Input | Completed | Mean deviation (cm) | Early deviation (cm) | Heading error (degrees) | Brightness heading change (degrees) |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['label']} | {r['completed']}/40 | {100*r['route_deviation_mean_m']:.2f} | {100*r['first50_polyline_mean_m']:.2f} | {r['clean_heading_error_deg']:.2f} | {r['brightness_heading_change_deg']:.2f} |")
    (HERE/"results-table.md").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4.1),gridspec_kw={"width_ratios":[1,1.2]})
    for i,r in enumerate(summary):
        values=100*matrix(rows,protocol,r["method"],"route_deviation_mean_m").mean(1)
        ax.scatter(values,np.full(8,i)+np.linspace(-.12,.12,8),s=17,alpha=.7)
        ax.plot([values.mean()]*2,[i-.25,i+.25],color="black",lw=2)
    ax.set(yticks=range(len(summary)),yticklabels=[r["label"] for r in summary],xlabel="Whole-run deviation (cm)",title="A  Eight routes, five seeds per point");ax.invert_yaxis()
    for i,r in enumerate(comparisons["apiaviz"]):
        bx.plot(r["crossed_ci95"],[i,i],color="#087f8c",lw=2);bx.scatter(r["effect"],i,color="#087f8c")
    bx.axvline(0,color="gray",ls="--");bx.set(yticks=range(5),yticklabels=[LABELS[r["method"]] for r in comparisons["apiaviz"]],xlabel="Candidate minus ApiaViz deviation (cm)",title="B  Development effects and crossed 95% intervals");bx.invert_yaxis()
    fig.tight_layout(w_pad=2);out=HERE/"figures";out.mkdir(exist_ok=True)
    fig.savefig(out/"refinements.pdf",bbox_inches="tight");fig.savefig(out/"refinements.png",dpi=180,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    for j,(key,label) in enumerate((("form_heading_error_deg","Form only"),("colour_heading_error_deg","Colour only"),("clean_heading_error_deg","Both streams"))):
        ax.bar(np.arange(len(summary))+(j-1)*.25,[r[key] for r in summary],width=.25,label=label)
    ax.set(xticks=range(len(summary)),xticklabels=[r["label"] for r in summary],ylabel="Centreline heading error (degrees)",title="Common-pose diagnostics: what each stream contributes")
    ax.tick_params(axis="x",labelrotation=25)
    ax.legend(fontsize=8,loc="upper left",bbox_to_anchor=(1.01,1))
    fig.tight_layout()
    fig.savefig(out/"streams.pdf",bbox_inches="tight");fig.savefig(out/"streams.png",dpi=180,bbox_inches="tight")
    print("\n".join(lines));print(json.dumps(comparisons,indent=2))


if __name__=="__main__":main()
