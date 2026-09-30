"""Analyse the fixed 2x2 frontend comparison and two matched controls."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apiaviz.research.statistics import paired_effect, holm_adjust
from apiaviz.research.study import file_hash

HERE = Path(__file__).parent
LABELS = {
    "apiaviz": "Original ApiaViz",
    "oriented_form": "Oriented form",
    "linear_colour": "Linear colour",
    "oriented_linear": "Both changes",
    "sobel_colour": "Sobel + colour",
    "ardin_input": "Ardin-style input",
}
CANDIDATES = ("oriented_form", "linear_colour", "oriented_linear")
REFERENCES = ("apiaviz", "sobel_colour", "ardin_input")


def load():
    protocol = json.loads((HERE / "protocol.json").read_text())
    rows = []
    for name in json.loads((HERE / "runs.json").read_text())["runs"]:
        run = ROOT / name
        manifest = json.loads((run / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert manifest["protocol"] == protocol
        assert manifest["protocol_sha256"] == file_hash(HERE / "protocol.json")
        rows.extend(dict(json.loads(line), run=name)
                    for line in (run / "results.jsonl").read_text().splitlines())
    expected = {(a, s, m) for a in protocol["ants"]
                for s in protocol["wiring_seeds"] for m in protocol["methods"]}
    assert len(rows) == len(expected) == protocol["navigation_trials"]
    assert {(r["ant"], r["seed"], r["preprocessing"]) for r in rows} == expected
    rows.sort(key=lambda r: (r["ant"], r["seed"], protocol["methods"].index(r["preprocessing"])))
    return protocol, rows


def matrix(rows, protocol, method, field="route_deviation_mean_m"):
    values = {(r["ant"], r["seed"]): r[field]
              for r in rows if r["preprocessing"] == method}
    return np.array([[values[a, s] for s in protocol["wiring_seeds"]]
                     for a in protocol["ants"]], dtype=float)


def effect(delta, **metadata):
    stats = paired_effect(delta)
    stats.pop("routes_favouring_apia")
    stats.pop("seeds_favouring_apia")
    route = delta.mean(1)
    seed = delta.mean(0)
    return dict(**metadata, **stats,
                routes_with_negative_effect=int((route < 0).sum()),
                seeds_with_negative_effect=int((seed < 0).sum()),
                leave_one_route_out_effects_cm=((route.sum() - route) / (len(route) - 1)).tolist(),
                leave_one_seed_out_effects_cm=((seed.sum() - seed) / (len(seed) - 1)).tolist())


def compare_previous(rows):
    """Require all rerun single variants and two references to reproduce exactly."""
    previous = {}
    for name in json.loads((ROOT / "docs/frontend-deep-dive/runs.json").read_text())["runs"]:
        run = ROOT / name
        assert json.loads((run / "manifest.json").read_text())["status"] == "complete"
        for row in map(json.loads, (run / "results.jsonl").read_text().splitlines()):
            previous[row["ant"], row["seed"], row["preprocessing"]] = (run, row)
    count = 0
    for row in rows:
        key = row["ant"], row["seed"], row["preprocessing"]
        if key not in previous:
            continue
        run, old = previous[key]
        assert json.loads((ROOT / row["run"] / row["trace"]).read_text()) == json.loads((run / old["trace"]).read_text())
        for field in old:
            if field != "elapsed_s":
                assert row[field] == old[field], (key, field)
        count += 1
    assert count == 160
    return dict(exact_previous_trials=count,
                scope="Every saved trajectory, scan score, probe and result field except elapsed time, for original, Sobel and both separate variants.")


def joint_probe_diagnostics(rows, protocol):
    """Descriptive common-pose choices; these are not independent replicates."""
    lookup = {(r["ant"], r["seed"], r["preprocessing"]): r for r in rows}
    counts = dict(centreline_probes=0, combination_better_than_both_singles=0,
                  combination_worse_than_both_singles=0, all_three_equal_error=0,
                  combination_changed_turn_from_original=0)
    for ant in protocol["ants"]:
        for seed in protocol["wiring_seeds"]:
            probes = {}
            for method in ("apiaviz", *CANDIDATES):
                row = lookup[ant, seed, method]
                saved = json.loads((ROOT / row["run"] / row["trace"]).read_text())
                probes[method] = [p for p in saved["probes"]
                                  if p["condition"] == "clean" and p["lateral_m"] == 0]
                assert len(probes[method]) == 8
            for index in range(8):
                p = {m: group[index] for m, group in probes.items()}
                assert len({r["route_index"] for r in p.values()}) == 1
                error = {m: 180. if r["no_evidence"] else abs(r["turn_deg"]) for m, r in p.items()}
                a, b, c = [error[m] for m in CANDIDATES]
                counts["centreline_probes"] += 1
                counts["combination_better_than_both_singles"] += int(c < min(a, b))
                counts["combination_worse_than_both_singles"] += int(c > max(a, b))
                counts["all_three_equal_error"] += int(a == b == c)
                counts["combination_changed_turn_from_original"] += int(p["oriented_linear"]["turn_deg"] != p["apiaviz"]["turn_deg"])
    counts["scope"] = "Descriptive post hoc diagnostic at 320 fixed route/seed probe poses; no additional significance tests."
    return counts


def main():
    protocol, rows = load()
    methods = protocol["methods"]
    values = {m: 100 * matrix(rows, protocol, m) for m in methods}
    comparisons = []
    for reference in REFERENCES:
        for candidate in CANDIDATES:
            comparisons.append(effect(values[candidate] - values[reference],
                                      candidate=candidate, reference=reference, kind="pairwise"))
    for reference in CANDIDATES[:2]:
        comparisons.append(effect(values["oriented_linear"] - values[reference],
                                  candidate="oriented_linear", reference=reference, kind="pairwise"))
    interaction = values["oriented_linear"] - values["oriented_form"] - values["linear_colour"] + values["apiaviz"]
    comparisons.append(effect(interaction, kind="interaction",
                              definition="combined - oriented - linear + original (cm)"))
    assert len(comparisons) == 12
    for item, adjusted in zip(comparisons, holm_adjust([item["p_raw"] for item in comparisons])):
        item["p_holm_exploratory"] = adjusted

    fields = ("polyline_mean_m", "first50_polyline_mean_m", "deviation_median_m",
              "clean_heading_error_deg", "clean_restoring_fraction", "brightness_heading_change_deg",
              "memory_active_fraction", "form_heading_error_deg", "colour_heading_error_deg",
              "form_matched_progress_error_m", "colour_matched_progress_error_m")
    summary = []
    for method in methods:
        group = [r for r in rows if r["preprocessing"] == method]
        summary.append(dict(method=method, label=LABELS[method], n=len(group),
                            completed=sum(r["reached_nest"] for r in group),
                            deviation_mean_cm=float(values[method].mean()),
                            deviation_trial_median_cm=float(np.median(values[method])),
                            deviation_trial_q90_cm=float(np.quantile(values[method], .9)),
                            route_means_cm=values[method].mean(1).tolist(),
                            seed_means_cm=values[method].mean(0).tolist(),
                            **{key: float(np.mean([r[key] for r in group])) for key in fields}))

    # Independent stream composition must survive encoding and stored probe readout.
    for field, single in (("form_heading_error_deg", "oriented_form"),
                          ("form_matched_progress_error_m", "oriented_form"),
                          ("colour_heading_error_deg", "linear_colour"),
                          ("colour_matched_progress_error_m", "linear_colour")):
        np.testing.assert_array_equal(matrix(rows, protocol, "oriented_linear", field),
                                      matrix(rows, protocol, single, field))

    result = dict(protocol=protocol, summary=summary, comparisons=comparisons,
                  previous_reproduction=compare_previous(rows),
                  joint_probe_diagnostics=joint_probe_diagnostics(rows, protocol),
                  stream_composition="Exact agreement for all 40 paired stream-specific heading and matched-progress probe results.",
                  inference="Exploratory: one 12-test Holm family. Exact sign flips of eight route means conditional on the five wiring seeds. Unadjusted 95% crossed route/seed bootstrap intervals. No independent-world or held-out inference.")
    (HERE / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (HERE / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| Frontend | Mean deviation (cm) | Median trial deviation (cm) | Reached nest | First 50 steps (cm) | Heading error (degrees) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['label']} | {row['deviation_mean_cm']:.2f} | {row['deviation_trial_median_cm']:.2f} | {row['completed']}/{row['n']} | {100*row['first50_polyline_mean_m']:.2f} | {row['clean_heading_error_deg']:.2f} |")
    (HERE / "results-table.md").write_text("\n".join(lines) + "\n")
    plot(summary, comparisons, values, protocol)
    print("\n".join(lines))
    print(json.dumps(comparisons, indent=2))


def plot(summary, comparisons, values, protocol):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    colours = ["#666666", "#2378ad", "#bd6a28", "#087f73", "#9453a6", "#aa4444"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 4.1))
    for i, row in enumerate(summary):
        route = np.array(row["route_means_cm"])
        ax.scatter(route, np.full(len(route), i) + np.linspace(-.13, .13, len(route)),
                   s=22, color=colours[i], alpha=.8)
        ax.plot([route.mean()] * 2, [i-.27, i+.27], color="black", lw=2)
    ax.set(yticks=range(len(summary)), yticklabels=[r["label"] for r in summary],
           xlabel="Mean route deviation (cm; lower is better)",
           title="A  Each dot is one route, averaged over five seeds")
    ax.invert_yaxis()
    shown = [r for r in comparisons if r.get("reference") == "apiaviz"]
    for i, row in enumerate(shown):
        bx.plot(row["crossed_ci95"], [i, i], color=colours[i+1], lw=2)
        bx.scatter(row["effect"], i, color=colours[i+1], s=35)
    bx.axvline(0, color="gray", ls="--", lw=1)
    bx.set(yticks=range(3), yticklabels=[LABELS[r["candidate"]] for r in shown],
           xlabel="Change from original ApiaViz (cm)",
           title="B  Paired effects and crossed 95% intervals")
    bx.invert_yaxis()
    fig.tight_layout(w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"performance.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    names = protocol["methods"]
    data = np.stack([values[m].mean(1) for m in names], axis=1)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white" if data[i,j] > .58*data.max() else "black")
    ax.set(xticks=range(len(names)), xticklabels=[LABELS[m] for m in names],
           yticks=range(len(protocol["ants"])), yticklabels=[f"Ant {a}" for a in protocol["ants"]],
           title="A  Deviation by route (cm; five-seed average)")
    ax.tick_params(axis="x", labelrotation=35)
    fig.colorbar(im, ax=ax, shrink=.7, label="cm")
    for j, (field, label) in enumerate((("form_heading_error_deg", "Form only"),
                                       ("colour_heading_error_deg", "Colour only"),
                                       ("clean_heading_error_deg", "Combined readout"))):
        bx.bar(np.arange(4) + (j-1)*.25, [r[field] for r in summary[:4]], width=.25, label=label)
    bx.set(xticks=range(4), xticklabels=[r["label"] for r in summary[:4]],
           ylabel="Heading error (degrees)", title="B  Identical-pose memory probes")
    bx.tick_params(axis="x", labelrotation=35)
    bx.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(w_pad=2)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"routes-and-streams.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
