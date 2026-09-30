"""Plot saved trajectories; never re-simulate or interpolate runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apiaviz.mbant.io_utils import load_ant_data, prepare_route


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--memories", nargs="+", default=["population", "sequential"])
    p.add_argument("--output", type=Path, default=Path("apiaviz/output/openloop/development-trajectories.svg"))
    args = p.parse_args()
    rows = []
    for run in args.runs:
        manifest = json.loads((run / "manifest.json").read_text())
        settings = manifest["settings"]
        for line in (run / "results.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["memory"] in args.memories:
                rows.append((run, settings, row))
    jobs = sorted({(r["ant"], r["route"]) for _, _, r in rows})
    roles = sorted({settings.get("role", "development") for _, settings, _ in rows})
    role_label = "/".join(roles)
    if not jobs:
        raise ValueError("No matching trajectories")
    ants = load_ant_data(str(Path(rows[0][1]["world_dir"]) / "AntData.mat"))
    fig, axes = plt.subplots(1, len(jobs), figsize=(5 * len(jobs), 7), squeeze=False)
    for ax, (ant, route) in zip(axes[0], jobs):
        positions, _, _ = prepare_route(ants[f"Ant{ant}"]["routes"][f"Route{route}"])
        ax.plot(*positions.T, "k--", linewidth=2, label="Taught route")
        ax.scatter(*positions[0], marker="s", color="black", zorder=4)
        ax.scatter(*positions[-1], marker="*", s=150, color="black", zorder=4)
        for run, settings, row in rows:
            if (row["ant"], row["route"]) != (ant, route):
                continue
            trace = json.loads((run / row["trace"]).read_text())["trajectory"]
            if not trace:
                continue
            _, headings, _ = prepare_route(ants[f"Ant{ant}"]["routes"][f"Route{route}"])
            radians = np.radians(headings[0])
            initial = positions[0] + settings.get("start_lateral", 0.) * np.array([-np.sin(radians), np.cos(radians)])
            points = np.array([initial.tolist()] + [x["position"] for x in trace])
            encoder = settings.get("encoder", "adaptive")
            if encoder == "checkpoint":
                encoder = "adaptive"
            teaching = settings.get("convergence", "future") + " convergence" if row["lookahead_m"] else "parallel"
            result = "homed" if row["reached_nest"] else "failed"
            label = f"{encoder}, {row['memory']}, {teaching}: {result}"
            label += (f"\nrelease {settings.get('start_lateral', 0):g} m / {settings.get('start_heading', 0):g}°; "
                      f"bin/delay {settings.get('time_bin', 0):g}/{settings.get('inhibition_delay', 0):g} ms")
            if settings.get("current_gain", 1.) != 1.:
                label += f"; current gain {settings['current_gain']:g}"
            ax.plot(*points.T, linewidth=1.2, label=label)
            ax.scatter(*points[-1], s=25)
        ax.set(title=f"Ant {ant}, route {route}", xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.12))
    fig.suptitle(f"Spiking navigation without corrective resets — {role_label}")
    fig.tight_layout(rect=(0, .06, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
