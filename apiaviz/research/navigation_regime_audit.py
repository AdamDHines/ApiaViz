"""Audit teaching coverage and run a no-vision control on the saved grassland task.

This reads existing trajectories; it does not retrain models or render images.
Run from the repository root with ``python -m apiaviz.research.navigation_regime_audit``.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from apiaviz.mbant.config import NavigationConfig
from .mechanisms import polyline_distance
from .navigation import evaluate_route


LABELS = {"linear_colour": "ApiaViz · linear colour",
          "sobel_colour": "Sobel + colour", "ardin_input": "Ardin-style input"}
COLOURS = {"linear_colour": "#15796d", "sobel_colour": "#bc7722",
           "ardin_input": "#5870a5"}


class NoVision:
    """All candidates tie; the existing controller preserves its current heading."""
    def __init__(self, nav):
        self.world = SimpleNamespace(nav=nav)

    def __call__(self, position, headings):
        return np.zeros(len(headings))


def audit(environment, results, destination):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".gitignore").write_text("!*.png\n")
    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return data.decode()

    protocol = json.loads(read(environment / "protocol.json"))
    acquisition = json.loads(read(environment / "acquisition.json"))
    rows = [json.loads(line) for line in read(results / "results.jsonl").splitlines()]
    route, headings = np.array(protocol["route"]), np.array(protocol["headings"])
    teaching = np.array(acquisition["positions"])
    teaching_headings = np.array(acquisition["headings"])
    nav = NavigationConfig(**protocol["nav"])
    blind = evaluate_route(route, headings, NoVision(nav), "free", max_steps=protocol["max_steps"])
    blind_trace = blind.pop("trace")
    blind_positions = np.array([step["position"] for step in blind_trace])
    assert all(step["heading"] == headings[0] and step["reset_to"] is None for step in blind_trace)
    expected = route[0] + np.arange(1, len(blind_trace) + 1)[:, None] * nav.step_size * np.array(
        [np.cos(np.radians(headings[0])), np.sin(np.radians(headings[0]))])
    np.testing.assert_allclose(blind_positions, expected, atol=1e-12, rtol=0)
    blind["polyline_mean_m"] = float(polyline_distance(blind_positions, route).mean())

    groups = []
    traces = {}
    for resolution, mode, method in sorted({(r["resolution"], r["mode"], r["preprocessing"]) for r in rows}):
        selected = sorted([r for r in rows if (r["resolution"], r["mode"], r["preprocessing"]) ==
                           (resolution, mode, method)], key=lambda r: r["seed"])
        distances, yaw_errors, turns, corridor = [], [], [], []
        for row in selected:
            saved = json.loads(read(results / row["trace"]))
            trajectory, decisions = saved["trajectory"], saved["decisions"]
            assert len(trajectory) == len(decisions) == row["steps"]
            positions = np.array([d["position"] for d in decisions])
            after = np.array([s["position"] for s in trajectory])
            np.testing.assert_allclose(positions, np.vstack([route[0], after[:-1]]), atol=1e-12, rtol=0)
            heading_after = np.array([s["heading"] for s in trajectory])
            heading_before = np.r_[headings[0], heading_after[:-1]]
            d = np.linalg.norm(positions[:, None, :] - teaching[None, :, :], axis=2)
            nearest = d.argmin(axis=1)
            distances.extend(d[np.arange(len(d)), nearest])
            yaw_errors.extend(abs((heading_before - teaching_headings[nearest] + 180) % 360 - 180))
            turns.extend((heading_after - heading_before + 180) % 360 - 180)
            corridor.extend(polyline_distance(positions, route) <= protocol["acquisition"]["width"])
            traces[resolution, mode, method, row["seed"]] = after
        groups.append(dict(resolution=resolution, mode=mode, method=method,
                           seeds=[r["seed"] for r in selected], decisions=len(distances),
                           mean_deviation_cm=float(np.mean([r["route_deviation_mean_m"] for r in selected]) * 100),
                           per_seed_deviation_cm=[r["route_deviation_mean_m"] * 100 for r in selected],
                           nearest_teaching_xy_cm=dict(zip(("median", "p95", "max"),
                               (np.quantile(distances, [.5, .95, 1]) * 100).tolist())),
                           nearest_teaching_heading_error_deg=dict(zip(("median", "p95"),
                               np.quantile(yaw_errors, [.5, .95]).tolist())),
                           fraction_queries_within_20cm_of_route=float(np.mean(corridor)),
                           fraction_steps_without_turn=float(np.mean(np.isclose(turns, 0))),
                           nearest_teaching_xy_distances_cm=(np.array(distances) * 100).tolist()))

    delta = route[-1] - route[0]
    for name in ("navigation.py", "mechanisms.py", "navigation_regime_audit.py"):
        read(Path(__file__).parent / name)
    summary = dict(
        scope="One saved world and route; descriptive diagnostics, not independent biological replicates.",
        coverage_definition="Pre-step XY distance to closest acquisition pose; heading error uses that same pose. Quantiles pool decisions across seeds.",
        no_vision_policy="Constant candidate scores; existing tie policy preserves initial heading. No images, memory, route or endpoint enter the scorer.",
        endpoint_used_by_evaluator_only=True,
        training_views=len(teaching), route_stations=len(route),
        initial_heading_deg=float(headings[0]),
        direct_endpoint_bearing_deg=float(np.degrees(np.arctan2(delta[1], delta[0]))),
        heading_range_deg=[float(headings.min()), float(headings.max())],
        no_vision=blind, groups=groups, input_sha256=sources)
    (destination / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "no-vision-trajectory.json").write_text(json.dumps(blind_trace, indent=2) + "\n")
    make_figure(destination, route, headings, teaching, blind_positions, blind, groups, traces)
    return summary


def make_figure(destination, route, headings, teaching, blind_positions, blind, groups, traces):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(12, 8.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax = fig.add_subplot(grid[0, :])
    angle = np.radians(headings[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    def xy(points):
        return ((np.array(points) - route[0]) @ rotation).T
    ax.scatter(*xy(teaching), s=5, c="#d8dddf", label="693 teaching positions")
    ax.plot(*xy(route), color="#222b32", lw=1.6, label="Taught route")
    for method in LABELS:
        for i, seed in enumerate((19, 31, 43)):
            ax.plot(*xy(np.vstack([route[0], traces["74x18", "pixels", method, seed]])),
                    color=COLOURS[method], lw=1.2, alpha=.65, label=LABELS[method] if i == 0 else None)
    ax.plot(*xy(np.vstack([route[0], blind_positions])), color="#ad3d43", lw=2, ls="--", label="No vision: keep heading")
    ax.add_patch(Circle(xy(route[-1]), .2, facecolor="#e8e8e8", edgecolor="#606060", alpha=.7))
    ax.scatter(*xy(route[0]), s=35, color="#222b32", zorder=5)
    ax.set(title="A  Both visual guidance and a straight walk reach the endpoint",
           xlabel="Distance along the starting direction (m)", ylabel="Lateral position (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)

    ax = fig.add_subplot(grid[1, 0])
    common = [next(g for g in groups if (g["resolution"], g["mode"], g["method"]) ==
                   ("74x18", "pixels", m)) for m in LABELS]
    values = [g["mean_deviation_cm"] for g in common] + [blind["route_deviation_mean_m"] * 100]
    ax.bar(range(4), values, color=list(COLOURS.values()) + ["#ad3d43"], alpha=.85, width=.6)
    for x, g in enumerate(common):
        ax.scatter(x + np.linspace(-.12, .12, 3), g["per_seed_deviation_cm"], c="#20282b", s=18)
    for x, value in enumerate(values):
        ax.text(x, max(common[x]["per_seed_deviation_cm"]) + .6 if x < 3 else value + .6,
                f"{value:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(4), ["ApiaViz", "Sobel\n+ colour", "Ardin-style", "No vision"])
    ax.set(title="B  Vision substantially improves route fidelity",
           ylabel="Mean distance to taught route points (cm)", ylim=(0, 31))
    ax.grid(axis="y", alpha=.15)

    ax = fig.add_subplot(grid[1, 1])
    for resolution, mode, linestyle, label in [
        ("74x18", "pixels", "--", "ApiaViz · 74 × 18"),
        ("199x51", "angles", "-", "ApiaViz · 199 × 51, fixed degrees")]:
        g = next(g for g in groups if (g["resolution"], g["mode"], g["method"]) ==
                 (resolution, mode, "linear_colour"))
        distances = np.sort(g["nearest_teaching_xy_distances_cm"])
        ax.step(distances, np.arange(1, len(distances) + 1) / len(distances), where="post",
                color=COLOURS["linear_colour"], ls=linestyle, lw=2, label=label)
    ax.axhline(.95, color="#747c80", lw=.8, ls=":")
    ax.set(title="C  The tested views stay close to teaching positions",
           xlabel="Distance to nearest teaching position (cm)",
           ylabel="Fraction of pre-step positions", xlim=(0, 6), ylim=(0, 1.04))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(alpha=.15)
    fig.suptitle("What the grassland smoke test establishes", fontsize=17, weight="bold")
    fig.savefig(destination / "regime-audit.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path("apiaviz/output/grassland-smoke"))
    parser.add_argument("--results", type=Path, default=Path("apiaviz/output/grassland-resolution"))
    parser.add_argument("--output", type=Path, default=Path("docs/navigation-regime"))
    args = parser.parse_args()
    result = audit(args.environment, args.results, args.output)
    print(json.dumps({"no_vision": result["no_vision"], "groups": len(result["groups"]),
                      "output": str(args.output)}, indent=2))
