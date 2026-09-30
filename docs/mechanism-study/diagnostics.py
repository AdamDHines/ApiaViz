"""Descriptive checks of steering, seed variation and long excursions."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse import ROOT, load, matrix, LABELS


def main():
    protocol, rows = load()
    summaries = []
    for method in LABELS:
        group = [r for r in rows if r["preprocessing"] == method]
        turns = dict(inward=0, parallel=0, outward=0, no_evidence=0)
        for row in group:
            probes = json.loads((ROOT / row["run"] / row["trace"]).read_text())["probes"]
            for p in probes:
                if p["condition"] != "clean" or p["lateral_m"] == 0:
                    continue
                key = ("no_evidence" if p["no_evidence"] else "parallel" if p["turn_deg"] == 0
                       else "inward" if p["restoring"] else "outward")
                turns[key] += 1
        summaries.append(dict(method=method,
            off_route_probe_counts=turns,
            off_route_probe_fractions={k: v / sum(turns.values()) for k, v in turns.items()},
            activity_fraction_range=[min(r["memory_active_fraction"] for r in group),
                                     max(r["memory_active_fraction"] for r in group)],
            mean_activity_fraction=float(np.mean([r["memory_active_fraction"] for r in group])),
            first50_polyline_mean_cm=100 * float(np.mean([r["first50_polyline_mean_m"] for r in group])),
            median_trial_mean_deviation_cm=100 * float(np.median([r["route_deviation_mean_m"] for r in group])),
            successful_trial_mean_cm=100 * float(np.mean([r["route_deviation_mean_m"] for r in group if r["reached_nest"]])),
            failed_trial_mean_cm=100 * float(np.mean([r["route_deviation_mean_m"] for r in group if not r["reached_nest"]]))))
    delta = 100 * (matrix(rows, protocol, "sobel_colour", "route_deviation_mean_m")
                   - matrix(rows, protocol, "apiaviz", "route_deviation_mean_m"))
    result = dict(summary=summaries,
        sobel_minus_apia_by_seed_cm=dict(zip(map(str, protocol["wiring_seeds"]), delta.mean(0).tolist())),
        sobel_minus_apia_by_route_cm=dict(zip(map(str, protocol["ants"]), delta.mean(1).tolist())),
        scope="Post-outcome descriptive diagnostics, without additional hypothesis tests. Success/failure subsets differ across methods and are not matched comparisons. Probe counts are dependent, not independent sample sizes.")
    (HERE / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
