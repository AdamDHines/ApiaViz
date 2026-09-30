"""Build all paired statistics and figures after the fixed matrix finishes."""
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
from apiaviz.research.statistics import paired_effect,holm_adjust
from apiaviz.research.mechanisms import ALL_METHODS,ABLATIONS
from apiaviz.research.paper_baselines import METHODS

HERE=Path(__file__).parent
LABELS=dict(zip(ALL_METHODS,["ApiaViz","Grayscale","Raw colour","Simple opponency","Sobel + colour","Ardin-style input",
                           "Without hex smoothing","Without adaptation","Without form filter bank","Apia form + simple colour","Sobel form + Apia colour"]))
plt.rcParams.update({"font.size":9,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})


def load():
    protocol=json.loads((HERE/"protocol.json").read_text())
    rows=[];manifests=[]
    for path in json.loads((HERE/"runs.json").read_text())["runs"]:
        run=ROOT/path
        manifest=json.loads((run/"manifest.json").read_text())
        if manifest["status"]!="complete":raise RuntimeError(f"Incomplete run: {run}")
        assert manifest["protocol"]==protocol
        manifests.append(manifest)
        for line in (run/"results.jsonl").read_text().splitlines():rows.append(dict(json.loads(line),run=path))
    wanted={(a,s,m) for a in protocol["ants"] for s in protocol["wiring_seeds"] for m in ALL_METHODS}
    assert len(rows)==len(wanted)==protocol["navigation_trials"]
    assert {(r["ant"],r["seed"],r["preprocessing"]) for r in rows}==wanted
    for field in ("encoders","dataset_sha256","definitions","protocol_sha256"):
        assert all(m[field]==manifests[0][field] for m in manifests),field
    for ant in protocol["ants"]:
        group=[r for r in rows if r["ant"]==ant]
        for field in ("training_images_sha256","acquisition_sha256","probe_images_sha256","memory_views"):
            assert len({r[field] for r in group})==1,(ant,field)
    return protocol,rows


def matrix(rows,protocol,method,field):
    lookup={(r["ant"],r["seed"]):r[field] for r in rows if r["preprocessing"]==method}
    return np.array([[lookup[a,s] for s in protocol["wiring_seeds"]] for a in protocol["ants"]],dtype=float)


def comparisons(rows,protocol,methods,field,scale=100):
    reference=matrix(rows,protocol,"apiaviz",field)
    output=[]
    for method in methods:
        diff=(matrix(rows,protocol,method,field)-reference)*scale
        output.append(dict(method=method,label=LABELS[method],endpoint=field,**paired_effect(diff)))
    for row,p in zip(output,holm_adjust([r["p_raw"] for r in output])):row["p_holm"]=p
    return output


def save(fig,name):
    out=HERE/"figures";out.mkdir(exist_ok=True)
    fig.savefig(out/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(out/f"{name}.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


def plots(rows,protocol,summary,baseline,ablation):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.7),gridspec_kw={"width_ratios":[1,1.2]})
    for idx,m in enumerate(METHODS):
        vals=matrix(rows,protocol,m,"route_deviation_mean_m").mean(1)*100
        ax.scatter(np.full(len(vals),idx)+np.linspace(-.12,.12,len(vals)),vals,s=22,alpha=.8)
        ax.plot([idx-.25,idx+.25],[vals.mean()]*2,color="black",lw=2)
    ax.set(xticks=range(len(METHODS)),xticklabels=[LABELS[m] for m in METHODS],ylabel="Mean route deviation (cm)")
    ax.tick_params(axis="x",labelrotation=35);ax.set_title("A  Eight routes; each point averages five seeds",loc="left")
    for i,r in enumerate(baseline):
        lo,hi=r["crossed_ci95"]
        bx.plot([lo,hi],[i,i],color="#087f8c",lw=2)
        bx.scatter(r["effect"],i,color="#087f8c",s=35)
    bx.axvline(0,color="gray",ls="--",lw=1)
    bx.set(yticks=range(len(baseline)),yticklabels=[r["label"] for r in baseline],xlabel="Baseline minus ApiaViz deviation (cm)")
    bx.invert_yaxis();bx.set_title("B  Paired effect and crossed 95% interval",loc="left")
    fig.tight_layout(w_pad=2);save(fig,"navigation")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10,3.8))
    for i,r in enumerate(ablation):
        lo,hi=r["crossed_ci95"]
        ax.plot([lo,hi],[i,i],color="#8256a3",lw=2);ax.scatter(r["effect"],i,color="#8256a3",s=30)
    ax.axvline(0,color="gray",ls="--",lw=1);ax.invert_yaxis()
    ax.set(yticks=range(len(ablation)),yticklabels=[r["label"] for r in ablation],xlabel="Intervention minus ApiaViz deviation (cm)")
    ax.set_title("A  Removing or exchanging input stages",loc="left")
    selected=("apiaviz","sobel_colour","no_hex","no_adapt","no_dog")
    for i,m in enumerate(selected):
        group=[r for r in rows if r["preprocessing"]==m]
        near=np.mean([r["neighbour_overlap"] for r in group]);far=np.mean([r["distant_overlap"] for r in group])
        bx.plot([far,near],[i,i],color="#aeb7bd",lw=2)
        bx.scatter(near,i,color="#087f8c",s=35,label="Neighbouring views" if i==0 else None)
        bx.scatter(far,i,color="#ce7b32",s=35,label="Distant views" if i==0 else None)
    bx.set(yticks=range(len(selected)),yticklabels=[LABELS[m] for m in selected],xlabel="Normalised overlap of active KCs")
    bx.invert_yaxis();bx.legend(loc="lower right",fontsize=8)
    bx.set_title("B  How much visual similarity is retained?",loc="left")
    fig.tight_layout(w_pad=2);save(fig,"mechanisms")
    # Familiarity curves use every route and seed, with no illustrative-case selection.
    curves={m:{s:[] for s in (-.3,0.,.3)} for m in ("apiaviz","sobel_colour","no_hex","no_adapt","no_dog")}
    for row in rows:
        m=row["preprocessing"]
        if m not in curves:continue
        saved=json.loads((ROOT/row["run"]/row["trace"]).read_text())
        for p in saved["probes"]:
            if p["condition"]=="clean" and p["lateral_m"] in curves[m]:
                scores=np.asarray(p["scores"])
                curves[m][p["lateral_m"]].append(scores-scores[6])
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),sharey=True)
    for ax,side in zip(axes,(-.3,0.,.3)):
        for m,values in curves.items():ax.plot(np.arange(60,-61,-10),np.mean(values[side],axis=0),label=LABELS[m])
        ax.set(title=f"Lateral offset {side:+.1f} m",xlabel="Heading relative to route (degrees)")
        ax.axvline(0,color="gray",ls="--",lw=.8);ax.axhline(0,color="gray",lw=.5)
    axes[0].set_ylabel("Unfamiliarity minus tangent value")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,-.17),ncol=3,fontsize=8)
    fig.tight_layout();save(fig,"familiarity")


def main():
    protocol,rows=load()
    with (HERE/"results.csv").open("w",newline="") as handle:
        fields=list(rows[0]);writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(rows)
    metrics=("route_deviation_mean_m","polyline_mean_m","first50_polyline_mean_m","deviation_median_m",
             "memory_active_fraction","clean_heading_error_deg","clean_tangent_margin","clean_restoring_fraction",
             "brightness_heading_error_deg","brightness_heading_change_deg","neighbour_overlap","distant_overlap",
             "effective_rank","lifetime_recruited_fraction")
    summary=[]
    for m in ALL_METHODS:
        group=[r for r in rows if r["preprocessing"]==m]
        summary.append(dict(method=m,label=LABELS[m],trials=len(group),completed=sum(r["reached_nest"] for r in group),
                            **{k:float(np.mean([r[k] for r in group])) for k in metrics}))
    baseline=comparisons(rows,protocol,METHODS[1:],"route_deviation_mean_m")
    ablation=comparisons(rows,protocol,ABLATIONS,"route_deviation_mean_m")
    secondary={key:comparisons(rows,protocol,METHODS[1:],key) for key in ("polyline_mean_m","first50_polyline_mean_m","deviation_median_m")}
    aa=matrix(rows,protocol,"apiaviz","route_deviation_mean_m")*100
    sa=matrix(rows,protocol,"sobel_apia_colour","route_deviation_mean_m")*100
    ass=matrix(rows,protocol,"apia_simple_colour","route_deviation_mean_m")*100
    ss=matrix(rows,protocol,"sobel_colour","route_deviation_mean_m")*100
    factorial={"processed_colour_benefit_with_apia_form":paired_effect(ass-aa),
               "processed_colour_benefit_with_sobel_form":paired_effect(ss-sa),
               "apia_form_benefit_with_processed_colour":paired_effect(sa-aa),
               "apia_form_benefit_with_simple_colour":paired_effect(ss-ass),
               "colour_by_form_interaction":paired_effect((ass-aa)-(ss-sa))}
    output=dict(protocol=protocol,summary=summary,baseline=baseline,ablation=ablation,secondary=secondary,factorial_exploratory=factorial,
                interpretation="Positive effects favour ApiaViz. P-values use 8 route clusters, conditional on the five wiring seeds. Holm families are separate for five baselines and five mechanistic interventions. Secondary endpoint families are exploratory. Bootstrap intervals are unadjusted percentile 95% intervals; crossed intervals resample both route and seed axes. One simulated world and fixed colour assignment.")
    (HERE/"statistics.json").write_text(json.dumps(output,indent=2)+"\n")
    plots(rows,protocol,summary,baseline,ablation)
    lines=["| Input | Completed | Mean deviation (cm) | Difference vs ApiaViz (cm) | Crossed 95% CI | Holm p |",
           "|---|---:|---:|---:|---:|---:|"]
    contrasts={r["method"]:r for r in baseline+ablation}
    for r in summary:
        c=contrasts.get(r["method"])
        tail="— | — | —" if c is None else f"{c['effect']:.2f} | [{c['crossed_ci95'][0]:.2f}, {c['crossed_ci95'][1]:.2f}] | {c['p_holm']:.4f}"
        lines.append(f"| {r['label']} | {r['completed']}/40 | {r['route_deviation_mean_m']*100:.2f} | {tail} |")
    (HERE/"results-table.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
