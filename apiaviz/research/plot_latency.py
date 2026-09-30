"""Show KC event timing for a development view under explicit circuit settings."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .latency import LatencyEncoder, LatencyConfig, latency_race
from .navigation import World


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("apiaviz/output/openloop/kc-spike-timing.svg"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    world = World(Path("apiaviz/mbant/data/antview"), seed=99)
    positions, headings = world.route(1, 1)
    view = world.render(positions[:1], headings[:1])
    encoder = LatencyEncoder()
    drives = encoder.currents(view)
    cases = [("Ideal reference", LatencyConfig(), 200),
             ("1 ms bins / delay, gain 1", LatencyConfig(time_bin_ms=1, inhibition_delay_ms=1), 29),
             ("1 ms bins / delay, gain 0.1", LatencyConfig(time_bin_ms=1, inhibition_delay_ms=1, current_gain=.1), 113)]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    for ax, (label, config, k) in zip(axes, cases):
        count = 0
        for stream, drive in enumerate(drives):
            result = latency_race(drive, k, config)
            times = result["spike_times_ms"][0]
            indices = torch.where(torch.isfinite(times))[0]
            ax.scatter(times[indices].numpy(), indices.numpy() + stream * 4000,
                       s=5, alpha=.65, label=("Form KCs", "Colour KCs")[stream])
            count += len(indices)
        ax.set(title=f"{label}: {count}/8000 cells spiked ({count / 80:.2f}%)", ylabel="KC index")
        ax.set_xlim(0, 50)
        ax.set_ylim(-100, 8100)
        ax.grid(alpha=.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("First-spike time after presentation (ms)")
    fig.suptitle("Deterministic LIF KC events — Ant 1, Route 1, first development view")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
