"""Development experiments for navigation without corrective resets.

The policy receives only rendered views and its previous heading. Route geometry
is used during acquisition and by the evaluator, never for online steering.
Run with ``python -m apiaviz.research.openloop --help``.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from .circuit import CircuitConfig
from .encoders import EncoderConfig, VisualEncoder
from .navigation import World, Scorer, evaluate_route, make_memory, choose_heading
from .lateralized import LateralizedMemory, LateralizedScorer, rotated_acquisition
from .sequential import SequentialMemory
from .latency import LatencyEncoder, LatencyConfig
from .prototypes import PrototypeMemory
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json
from apiaviz.nav.torch_route import route_corridor_bank, corridor_offsets


def acquisition_views(positions, headings, viewpoints=9, width=.2, lookahead=0., convergence="future"):
    """Teach parallel headings, or headings converging on a future route point.

    Convergent headings are an explicit acquisition intervention, not an online
    lookahead controller. No coordinates or headings are retained in the memory.
    """
    pos, head = route_corridor_bank(positions, headings, corridor_offsets(viewpoints, width))
    if lookahead < 0 or width < 0 or viewpoints < 1 or convergence not in ("future", "tangent"):
        raise ValueError("Acquisition dimensions must be nonnegative and viewpoints positive")
    if lookahead:
        if convergence == "tangent":
            radians = np.radians(headings)
            targets = positions[:len(headings)] + lookahead * np.column_stack([np.cos(radians), np.sin(radians)])
        else:
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
            indices = np.minimum(np.searchsorted(arc, arc[:len(headings)] + lookahead), len(positions) - 1)
            targets = positions[indices]
        delta = np.repeat(targets, viewpoints, axis=0) - pos
        head = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    return pos, head


def restoring_probe(world, scorer, positions, headings, stride=10, lateral=.2):
    """Does a heading choice point towards a future route point off the route?

    Ground-truth geometry is used only to score this offline diagnostic.
    Tangent error alone incorrectly penalizes helpful inward steering.
    """
    rows = []
    offsets = np.arange(60., -61., -10.)
    for idx in range(0, len(headings), stride):
        for side in (-lateral, lateral):
            rad = math.radians(headings[idx])
            pos = positions[idx] + side * np.array([-math.sin(rad), math.cos(rad)])
            if hasattr(scorer, "steer"):
                turn, silent = scorer.steer(pos, float(headings[idx]))
                scores = []
            else:
                scores = scorer(pos, headings[idx] + offsets)
                winner, silent = choose_heading(scores, offsets)
                turn = float(offsets[winner])
            rows.append({"index": idx, "lateral_m": side, "turn_deg": turn,
                         "restoring": bool(not silent and side * math.sin(math.radians(turn)) < 0),
                         "no_evidence": silent,
                         "scores": [float(x) if np.isfinite(x) else None for x in scores]})
    return {"restoring_fraction": float(np.mean([r["restoring"] for r in rows])), "probes": rows}


def load_encoder(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    meta = checkpoint.get("metadata", checkpoint)
    if meta.get("model_class") == "LatencyEncoder" or "time_bin_ms" in meta["circuit"] or "inhibition_delay_ms" in meta["circuit"]:
        config = meta["encoder"]
        encoder = LatencyEncoder(config["code_dim"], config["sparsity"], config["seed"], LatencyConfig(**meta["circuit"]))
    else:
        encoder = VisualEncoder(EncoderConfig(**meta["encoder"]), CircuitConfig(**meta["circuit"]))
        encoder.calibration = meta.get("calibration")
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    encoder.eval()
    if isinstance(encoder, VisualEncoder) and encoder.config.mode not in ("threshold", "feedback", "adaptive"):
        raise ValueError("This experiment requires a spiking encoder checkpoint")
    return encoder


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--encoder", choices=("checkpoint", "latency"), default="checkpoint")
    p.add_argument("--code-dim", type=int, default=8000, help="TOTAL cells for the latency reference")
    p.add_argument("--sparsity", type=float, default=.05)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--inhibition-delay", type=float, default=0., help="ms, latency reference only")
    p.add_argument("--time-bin", type=float, default=0., help="Spike timestamp resolution in ms; 0 uses exact event times")
    p.add_argument("--current-gain", type=float, default=1., help="Scale feature drive above KC rheobase; latency reference only")
    p.add_argument("--ants", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--routes", nargs="+", type=int, default=[1])
    p.add_argument("--role", choices=("development", "evaluation"), default="development")
    p.add_argument("--world-dir", type=Path, default=Path("apiaviz/mbant/data/antview"))
    p.add_argument("--environment-seed", type=int, default=99)
    p.add_argument("--viewpoints", type=int, default=9)
    p.add_argument("--width", type=float, default=.2)
    p.add_argument("--lookaheads", nargs="+", type=float, default=[0., .5])
    p.add_argument("--convergence", choices=("future", "tangent"), default="future",
                   help="Future route-point target or local tangent target (preserves centreline heading)")
    p.add_argument("--memories", nargs="+", choices=("population", "cosine", "spike_overlap", "lateralized", "sequential", "prototype"), default=["population", "cosine"])
    p.add_argument("--memory-code", choices=("auto", "binary", "graded"), default="auto",
                   help="Input to 'population' memory only; other memories use their native code contract")
    p.add_argument("--segments", type=int, default=80)
    p.add_argument("--rotations", nargs="+", type=float, default=[-40., -20., 0., 20., 40.])
    p.add_argument("--turn-gain", type=float, default=180.)
    p.add_argument("--sequence-ahead", type=int, default=3)
    p.add_argument("--sequence-behind", type=int, default=1)
    p.add_argument("--start-lateral", type=float, default=0.)
    p.add_argument("--start-heading", type=float, default=0., help="Heading error at release, degrees")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("apiaviz/output/openloop"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.role == "development" and not set(args.ants) <= {1, 2, 3}:
        raise ValueError("Use ants 1–3 for development; label other runs explicitly as evaluation")
    if min(args.viewpoints, args.segments, args.max_steps, args.threads) < 1:
        raise ValueError("Counts must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    if args.encoder == "checkpoint" and args.checkpoint is None:
        raise ValueError("Supply --checkpoint or choose --encoder latency")
    if args.encoder == "latency" and args.checkpoint is not None:
        raise ValueError("The latency reference is generated from its seed; omit --checkpoint")
    encoder = (load_encoder(args.checkpoint) if args.encoder == "checkpoint" else
               LatencyEncoder(args.code_dim, args.sparsity, args.seed,
                              LatencyConfig(inhibition_delay_ms=args.inhibition_delay, time_bin_ms=args.time_bin,
                                            current_gain=args.current_gain)))
    before = fingerprint(encoder)
    sources = [p for p in sorted(Path("apiaviz").rglob("*.py"))
               if "output" not in p.parts and "data" not in p.parts]
    manifest = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "encoder_fingerprint": before,
                "checkpoint_sha256": file_hash(args.checkpoint) if args.checkpoint else None,
                "encoder": asdict(encoder.config), "circuit": asdict(encoder.circuit_config),
                "source_sha256": {str(p): file_hash(p) for p in sources},
                "dataset_sha256": {p: file_hash(args.world_dir / p) for p in ("AntData.mat", "world5000_gray.mat")},
                "status": "running", "torch": torch.__version__, "numpy": np.__version__}
    write_json(out / "manifest.json", manifest)
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path))
    torch.save({"state_dict": encoder.state_dict(), "metadata": {
                "model_class": type(encoder).__name__, "encoder": asdict(encoder.config),
                "circuit": asdict(encoder.circuit_config),
                "calibration": getattr(encoder, "calibration", None)}}, out / "encoder.pt")
    print(f"Output: {out}", flush=True)
    world = World(args.world_dir, seed=args.environment_seed)
    rows = []
    for ant, route in world.jobs(args.ants, args.routes):
        positions, headings = world.route(ant, route)
        for lookahead in args.lookaheads:
            started = time.perf_counter()
            pos, head = acquisition_views(positions, headings, args.viewpoints, args.width, lookahead, args.convergence)
            ordinary_codes = None
            for kind in args.memories:
                if kind == "lateralized":
                    teach_pos, teach_head, rotations, groups = rotated_acquisition(pos, head, args.rotations, args.segments)
                    codes = encode(encoder, world.render(teach_pos, teach_head))
                    memory = LateralizedMemory(codes, rotations, groups)
                    scorer = LateralizedScorer(world, encoder, memory, encode, args.turn_gain)
                else:
                    if ordinary_codes is None:
                        ordinary_codes = encode(encoder, world.render(pos, head))
                    codes = ordinary_codes
                    graded = (encoder.config.readout == "count" if args.memory_code == "auto"
                              else args.memory_code == "graded")
                    if kind == "sequential":
                        memory = SequentialMemory(codes, args.segments, args.sequence_ahead, args.sequence_behind)
                    elif kind == "prototype":
                        memory = PrototypeMemory(codes, args.segments)
                    elif kind == "spike_overlap":
                        memory = SpikeOverlapMemory(codes)
                    else:
                        memory = make_memory(codes, kind, args.segments, graded)
                    scorer = Scorer(world, encoder, memory, encode, "clean", 0., args.environment_seed)
                # Isolated off-route probes cannot test a stateful sequence gate
                # without an acquisition-consistent history. Do not inject a
                # ground-truth segment index just to score these probes.
                probe = ({"restoring_fraction": None, "probes": []} if kind == "sequential" else
                         restoring_probe(world, scorer, positions, headings))
                angle = math.radians(headings[0])
                initial = positions[0] + args.start_lateral * np.array([-math.sin(angle), math.cos(angle)])
                result = {} if args.probe_only else evaluate_route(
                    positions, headings, scorer, "free", args.max_steps,
                    initial_position=initial, initial_heading=headings[0] + args.start_heading)
                name = f"ant-{ant}-route-{route}-{kind}-lookahead-{lookahead:g}"
                write_json(out / f"{name}.json", {"probes": probe.pop("probes"), "trajectory": result.pop("trace", []),
                                               "decisions": getattr(scorer, "decisions", [])})
                row = {"ant": ant, "route": route, "memory": kind, "lookahead_m": lookahead,
                       "role": args.role, "trace": f"{name}.json", "memory_views": len(codes),
                       "memory_active_fraction": float((codes > 0).float().mean()),
                       **probe, **result, "elapsed_s": time.perf_counter() - started}
                rows.append(row)
                with (out / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
    if fingerprint(encoder) != before:
        raise RuntimeError("Encoder changed during evaluation")
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", rows)


if __name__ == "__main__":
    main()
