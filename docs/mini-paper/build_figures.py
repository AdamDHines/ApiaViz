"""Build manuscript figures and tables directly from the paired navigation runs."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from apiaviz.mbant.io_utils import load_ant_data, prepare_route
from apiaviz.research.paper_baselines import METHODS

HERE = Path(__file__).parent
FIG = HERE / "figures"
LABELS = {"apiaviz": "ApiaViz", "gray": "Grayscale pixels", "raw_colour": "Raw colour",
          "opponent": "Simple opponency", "sobel_colour": "Sobel + colour", "ardin_input": "Ardin-style input"}
COLOURS = dict(zip(METHODS, ["#087f8c", "#909090", "#ce7b32", "#8256a3", "#4669b1", "#b34351"]))
ANTS = (5, 8, 11, 14)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def methodology():
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set(xlim=(0, 10), ylim=(0, 3.3)); ax.axis("off")
    def box(x, y, w, h, text, face="#eef2f4", edge="#8b989f", size=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                   linewidth=.8, edgecolor=edge, facecolor=face))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=size)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>", mutation_scale=11,
                                    color="#4d5962", linewidth=1))
    box(.04, 1.1, 1.25, .9, "Same\npanoramic\nG/B images")
    box(1.68, .25, 2.5, 2.6, "IMAGE PROCESSING\n\nApiaViz\nGrayscale pixels\nRaw colour\nSimple opponency\nSobel + colour\nArdin-style input", "#e7f4f2", "#087f8c", 9)
    ax.text(2.93, 3.02, "Only this stage changes", ha="center", color="#087f8c", weight="bold", fontsize=9)
    box(4.57, 1.1, 1.55, .9, "Fixed wiring\n8,000 KCs\nFirst spikes")
    box(6.48, 1.1, 1.48, .9, "Same training\nSpike-pattern\nmemory")
    box(8.32, 1.1, 1.58, .9, "Same scan\nand steering\nNo resets")
    for x1,x2 in [(1.29,1.68),(4.18,4.57),(6.12,6.48),(7.96,8.32)]:arrow(x1,1.55,x2,1.55)
    ax.text(7.25, .45, "Fixed timing, inhibition,\nmemory and movement", ha="center", fontsize=8)
    save(fig, "methodology")


def load_results():
    rows = []
    manifests = []
    runs = json.loads((HERE / "runs.json").read_text())["runs"]
    for run in [ROOT / name / "manifest.json" for name in runs]:
        m = json.loads(run.read_text())
        if m["status"] != "complete":
            raise RuntimeError(f"Run not finished: {run.parent}")
        manifests.append(m)
        for line in (run.parent / "results.jsonl").read_text().splitlines():
            r = json.loads(line)
            r["run"] = str(run.parent.relative_to(ROOT))
            rows.append(r)
    expected = {(a, method) for a in ANTS for method in METHODS}
    if len(rows) != len(expected) or {(r["ant"],r["preprocessing"]) for r in rows} != expected:
        raise RuntimeError("Expected exactly four matched routes for every frontend")
    for field in ("encoder", "circuit", "encoder_fingerprint", "dataset_sha256"):
        assert all(m[field] == manifests[0][field] for m in manifests), field
    for ant in ANTS:
        group = [r for r in rows if r["ant"] == ant]
        for key in ("training_images_sha256", "acquisition_sha256", "memory_views"):
            assert len({r[key] for r in group}) == 1, (ant,key)
        assert all(r["corrective_resets"] == 0 for r in group)
    # Confirm the unchanged ApiaViz pipeline reproduces its prior frozen trials.
    old_dirs = [ROOT / "apiaviz/output/openloop" / name for name in
                ("20260929T011500.321299Z-62148", "20260929T011500.321299Z-62149")]
    old = {r["ant"]: (run,r) for run in old_dirs for r in
           [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]}
    for row in (r for r in rows if r["preprocessing"] == "apiaviz"):
        run, prior = old[row["ant"]]
        for key in ("reached_nest", "steps", "route_deviation_mean_m", "final_nest_distance_m"):
            assert row[key] == prior[key], (row["ant"],key)
        a = json.loads((ROOT / row["run"] / row["trace"]).read_text())["trajectory"]
        b = json.loads((run / prior["trace"]).read_text())["trajectory"]
        assert a == b, "ApiaViz trajectory changed"
    return rows


def results_figure(rows):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 2.7), gridspec_kw={"width_ratios":[1.15,1]})
    cells = np.array([[int(next(r for r in rows if r["preprocessing"]==m and r["ant"]==a)["reached_nest"])
                       for a in ANTS] for m in METHODS])
    from matplotlib.colors import ListedColormap
    ax.imshow(cells, cmap=ListedColormap(["#f0dce0", "#c8e6df"]), vmin=0, vmax=1, aspect="auto")
    for i in range(6):
        for j in range(4):ax.text(j,i,"yes" if cells[i,j] else "no",ha="center",va="center",fontsize=8)
    ax.set(xticks=range(4),xticklabels=[f"Ant {a}" for a in ANTS],yticks=range(6),yticklabels=[LABELS[m] for m in METHODS])
    ax.set_title("A  Reached the nest",loc="left",weight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    jitter = np.linspace(-.17,.17,4)
    for i,m in enumerate(METHODS):
        values = [100*next(r for r in rows if r["preprocessing"]==m and r["ant"]==a)["route_deviation_mean_m"] for a in ANTS]
        bx.scatter(values, i+jitter, color=COLOURS[m], s=22, zorder=3)
        bx.plot([np.median(values)]*2,[i-.27,i+.27],color="black",lw=1.4)
    bx.set_xscale("log");bx.set_ylim(5.5,-.5)
    bx.set(yticks=range(6),yticklabels=[],xlabel="Mean route deviation (cm; log scale)")
    bx.set_title("B  All trials, including failures",loc="left",weight="bold")
    bx.grid(axis="x",alpha=.2);bx.tick_params(axis="y",length=0)
    fig.tight_layout(w_pad=1.5)
    save(fig,"comparison")


def trajectories(rows):
    ants = load_ant_data(str(ROOT / "apiaviz/mbant/data/antview/AntData.mat"))
    fig, axes = plt.subplots(1,2,figsize=(7,2.45))
    for ax, ant in zip(axes, (5,8)):
        positions,_,_ = prepare_route(ants[f"Ant{ant}"]["routes"]["Route2"])
        along = positions[-1] - positions[0]
        along = along / np.linalg.norm(along)
        basis = np.stack([along, [-along[1], along[0]]], axis=1)
        def rotate(points):
            return (points - positions[0]) @ basis
        for method in METHODS:
            row = next(r for r in rows if r["ant"]==ant and r["preprocessing"]==method)
            trace = json.loads((ROOT / row["run"] / row["trace"]).read_text())["trajectory"]
            points = rotate(np.array([positions[0].tolist()] + [t["position"] for t in trace]))
            ax.plot(*points.T,color=COLOURS[method],lw=1.5 if method=="apiaviz" else .85,
                    alpha=1 if method=="apiaviz" else .85,label=LABELS[method],zorder=3 if method=="apiaviz" else 2)
            ax.scatter(*points[-1],color=COLOURS[method],s=9)
        plotted_route = rotate(positions)
        ax.plot(*plotted_route.T,"k--",lw=1.2,label="Taught route",zorder=4)
        ax.scatter(*plotted_route[0],marker="s",s=25,color="black",zorder=5)
        ax.scatter(*plotted_route[-1],marker="*",s=75,color="black",zorder=5)
        ax.set(title=f"Ant {ant}, Route 2",xlabel="Along start–nest axis (m)",ylabel="Lateral (m)")
        ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc="lower center",bbox_to_anchor=(.5,-.16),fontsize=8,frameon=False)
    fig.tight_layout()
    save(fig,"trajectories")


def tables(rows):
    with (HERE / "results.csv").open("w",newline="") as h:
        fields = ["preprocessing","ant","route","reached_nest","steps","route_deviation_mean_m",
                  "route_deviation_max_m","memory_active_fraction","corrective_resets","run","trace"]
        w=csv.DictWriter(h,fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    summary=[]
    for method in METHODS:
        group=[r for r in rows if r["preprocessing"]==method]
        summary.append({"method":method,"label":LABELS[method],"successes":sum(r["reached_nest"] for r in group),
                        "mean_deviation_cm":float(np.mean([r["route_deviation_mean_m"]*100 for r in group])),
                        "mean_activity_pct":float(np.mean([r["memory_active_fraction"]*100 for r in group])),
                        "activity_min_pct":min(r["memory_active_fraction"]*100 for r in group),
                        "activity_max_pct":max(r["memory_active_fraction"]*100 for r in group)})
    (HERE / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=[r"\begin{tabular}{@{}lrrr@{}}\toprule",r"Image preprocessing & Completed & Mean deviation & Active KCs \\",r" & (of 4) & (cm) & (\%) \\\midrule"]
    for r in summary:
        lines.append(f"{r['label']} & {r['successes']}/4 & {r['mean_deviation_cm']:.1f} & {r['mean_activity_pct']:.2f} \\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (HERE / "results-table.tex").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    methodology()
    if "--methodology-only" not in sys.argv:
        rows=load_results();results_figure(rows);trajectories(rows);tables(rows)
