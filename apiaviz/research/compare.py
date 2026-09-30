"""Paired comparisons from saved studies: python -m apiaviz.research.compare RUN A B."""
import argparse
import json
from pathlib import Path
import numpy as np

from .metrics import cluster_bootstrap, reward_metrics


def compare(run, baseline, candidate, samples=1000, seed=7):
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    left = [row for row in rows if row["encoder"] == baseline]
    right = [row for row in rows if row["encoder"] == candidate]
    if not left or not right:
        raise ValueError("Both encoder IDs must occur in this run")
    keys = ("task", "flower_task", "condition", "severity", "fixations", "shots_per_class",
            "persist_fixations", "readout", "protocol", "viewpoints")
    def key(row):
        return tuple(row.get(k) for k in keys)
    output = []
    for condition in sorted({key(row) for row in left}, key=str):
        a = [row for row in left if key(row) == condition]
        b = [row for row in right if key(row) == condition]
        if not b:
            continue
        if a[0]["task"] == "nav":
            index = lambda row: (row["ant"], row["route"], row["environment_seed"])
            aa, bb = {index(row): row for row in a}, {index(row): row for row in b}
            if aa.keys() != bb.keys():
                raise ValueError("Navigation comparisons require identical ant/route/environment trials")
            ids = sorted(aa)
            fields = [name for name in ("reached_nest", "heading_error_deg", "final_nest_distance_m",
                      "route_deviation_mean_m", "path_length_m", "corrective_resets") if name in a[0]]
            metrics = {field: cluster_bootstrap([float(bb[i][field]) - float(aa[i][field]) for i in ids],
                                               [i[0] for i in ids], seed, samples) for field in fields}
        else:
            if len(a) != 1 or len(b) != 1:
                raise ValueError("Ambiguous flower comparison")
            aa = json.loads((run / a[0]["predictions"]).read_text())
            bb = json.loads((run / b[0]["predictions"]).read_text())
            if aa["image_id"] != bb["image_id"]:
                raise ValueError("Flower predictions must have aligned image IDs and repeats")
            ids = np.asarray(aa["image_id"])
            if "correct" in aa:
                metrics = {"accuracy": cluster_bootstrap(np.asarray(bb["correct"]) - aa["correct"], ids, seed, samples)}
            else:
                if aa["reward"] != bb["reward"]:
                    raise ValueError("Reward assignments differ")
                reward, sa, sb, da, db = map(np.asarray, (aa["reward"], aa["score"], bb["score"], aa["go"], bb["go"]))
                def delta(idx):
                    ma, mb = reward_metrics(reward[idx], sa[idx], da[idx]), reward_metrics(reward[idx], sb[idx], db[idx])
                    return {k: mb[k] - ma[k] if ma[k] is not None and mb[k] is not None else None for k in ma}
                point = delta(np.arange(len(ids)))
                clusters = [np.flatnonzero(ids == i) for i in np.unique(ids)]
                rng = np.random.default_rng(seed)
                draws = {k: [] for k in point}
                for _ in range(samples):
                    idx = np.concatenate([clusters[i] for i in rng.integers(len(clusters), size=len(clusters))])
                    for k, value in delta(idx).items():
                        if value is not None:
                            draws[k].append(value)
                metrics = {k: {"mean": value, "ci95": np.quantile(draws[k], [.025, .975]).tolist() if draws[k] else None}
                           for k, value in point.items()}
        output.append({"condition": dict(zip(keys, condition)), "candidate_minus_baseline": metrics})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(compare(args.run, args.baseline, args.candidate, args.samples), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
