"""Paired open-loop comparison: change image processing only.

All methods share the SAME trained-view bank, projection weights, spike timing,
inhibition trigger, memory rule and controller. There is no per-method tuning.
The Ardin condition adapts the published preprocessing, not the full 2016 SNN.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
import torch
from torch.nn import functional as F

from .latency import LatencyEncoder, LatencyConfig
from .navigation import World, Scorer, evaluate_route
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize

METHODS = ("apiaviz", "gray", "raw_colour", "opponent", "sobel_colour", "ardin_input")
DEFINITIONS = {
    "apiaviz": "Unchanged ApiaViz: hexagonal averaging, local adaptation, centre-surround form, opponent colour.",
    "gray": "L=(G+B)/2; replicated into all five feature planes. Common pooling performs downsampling.",
    "raw_colour": "Form planes [L,G,B]; second stream [G,B]. No learned or spatial filters.",
    "opponent": "Form planes [L,L,L]; colour planes [relu(G-B),relu(B-G)].",
    "sobel_colour": "Gaussian sigma=1 pixel (5x5), then Sobel x/y/magnitude of L; simple opponent colour planes.",
    "ardin_input": "Invert L, CLAHE at native resolution, bicubic downsample to 10x36, L2 normalize; replicate into five planes before common pooling. Ardin-style preprocessing only, using scikit-image rather than MATLAB.",
}


def feature_maps(images, method, backbone):
    """Return three form and two auxiliary planes, with no permanently empty pool."""
    if method not in METHODS:
        raise ValueError(method)
    gb = images[:, -2:]
    lum = gb.mean(1, keepdim=True)
    if method == "apiaviz":
        maps = backbone(gb * 2 - 1, return_maps=True)
        return maps["contrast_features"], maps["chromatic_feature"][:, 1:]
    if method == "ardin_input":
        rows = []
        for view in lum[:, 0].cpu().numpy():
            equalized = equalize_adapthist(np.clip(1 - view, 0, 1), clip_limit=.01, nbins=256)
            small = resize(equalized, (10, 36), order=3, anti_aliasing=True, preserve_range=True)
            small = small / max(float(np.linalg.norm(small)), 1e-12)
            rows.append(small)
        lum = torch.as_tensor(np.asarray(rows), dtype=images.dtype, device=images.device)[:, None]
    if method in ("gray", "ardin_input"):
        return lum.repeat(1, 3, 1, 1), lum.repeat(1, 2, 1, 1)
    if method == "raw_colour":
        return torch.cat([lum, gb], 1), gb
    difference = gb[:, :1] - gb[:, 1:]
    colour = torch.cat([difference.relu(), (-difference).relu()], 1)
    if method == "opponent":
        return lum.repeat(1, 3, 1, 1), colour
    axis = torch.arange(-2, 3, dtype=images.dtype, device=images.device)
    gaussian = torch.exp(-axis.square() / 2)
    kernel = gaussian[:, None] * gaussian[None, :]
    kernel = (kernel / kernel.sum())[None, None]
    smooth = F.conv2d(F.pad(lum, (2, 2, 2, 2), mode="reflect"), kernel)
    sx = images.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    gradients = F.conv2d(F.pad(smooth, (1, 1, 1, 1), mode="reflect"), torch.stack([sx, sx.T])[:, None])
    magnitude = torch.linalg.vector_norm(gradients, dim=1, keepdim=True)
    return torch.cat([gradients, magnitude], 1), colour


class FrontendEncoder(LatencyEncoder):
    def __init__(self, method="apiaviz", code_dim=8000, sparsity=.02825, seed=7, circuit=None):
        super().__init__(code_dim, sparsity, seed, circuit or LatencyConfig(
            current_gain=.1, time_bin_ms=1., inhibition_delay_ms=1.))
        if method not in METHODS:
            raise ValueError(method)
        self.method = method

    @torch.no_grad()
    def currents(self, images):
        if images.ndim != 4 or images.shape[1] not in (2, 3) or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Expected finite GB/RGB images in [0,1]")
        drives = []
        maps = feature_maps(images, self.method, self.features.backbone)
        for feature, projection in zip(maps, self.features.legacy):
            flat = adaptive_avg_pool2d_anysize(feature, projection.pool_hw).flatten(1)
            flat = (flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-6)
            drives.append(F.relu(flat @ projection.connection))
        return drives


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ants", nargs="+", type=int, default=[5, 8, 11, 14])
    p.add_argument("--routes", nargs="+", type=int, default=[2])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--world-dir", type=Path, default=Path("apiaviz/mbant/data/antview"))
    p.add_argument("--output", type=Path, default=Path("apiaviz/output/paper-baselines"))
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    model = FrontendEncoder()
    frozen = fingerprint(model)
    sources = [f for f in sorted(Path("apiaviz").rglob("*.py")) if "output" not in f.parts and "data" not in f.parts]
    shared = {"world_dir": str(args.world_dir), "ants": args.ants, "routes": args.routes,
              "max_steps": 200, "start_lateral": 0., "start_heading": 0.,
              "environment_seed": 99, "viewpoints": 9, "width": .2, "lookahead": 0.,
              "memory": "spike_overlap", "role": "paired-comparison", "threads": args.threads,
              "methods": args.methods, "preprocessing_definitions": DEFINITIONS,
              "shared_parameters": "No baseline-specific calibration or tuning."}
    manifest = {"settings": shared, "encoder": asdict(model.config), "circuit": asdict(model.circuit_config),
                "encoder_fingerprint": frozen, "status": "running", "torch": str(torch.__version__),
                "numpy": np.__version__, "dataset_sha256": {n: file_hash(args.world_dir / n) for n in ("AntData.mat", "world5000_gray.mat")},
                "source_sha256": {str(f): file_hash(f) for f in sources}}
    write_json(out / "manifest.json", manifest)
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sources: archive.write(f, str(f))
    torch.save({"state_dict": model.state_dict(), "metadata": {"model_class": "FrontendEncoder",
                "encoder": asdict(model.config), "circuit": asdict(model.circuit_config),
                "preprocessing_definitions": DEFINITIONS}}, out / "encoder.pt")
    print(f"Output: {out}", flush=True)
    world = World(args.world_dir, seed=99)
    if set(world.jobs(args.ants, args.routes)) != {(a, r) for a in args.ants for r in args.routes}:
        raise ValueError("Requested route unavailable")
    rows = []
    for ant, route in world.jobs(args.ants, args.routes):
        positions, headings = world.route(ant, route)
        teach_pos, teach_head = acquisition_views(positions, headings, 9, .2, 0.)
        images = world.render(teach_pos, teach_head)
        bank_hash = hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        acquisition_hash = hashlib.sha256(teach_pos.tobytes() + teach_head.tobytes()).hexdigest()
        for method in args.methods:
            started = time.perf_counter()
            model.method = method
            if fingerprint(model) != frozen:
                raise RuntimeError("Fixed encoder weights changed")
            codes = encode(model, images)
            memory = SpikeOverlapMemory(codes)
            memory_hash = fingerprint(memory)
            scorer = Scorer(world, model, memory, encode, "clean", 0., 99)
            result = evaluate_route(positions, headings, scorer, "free", max_steps=200)
            if fingerprint(memory) != memory_hash or fingerprint(model) != frozen:
                raise RuntimeError("Encoder or route memory changed during recall")
            name = f"ant-{ant}-route-{route}-{method}.json"
            write_json(out / name, {"trajectory": result.pop("trace"), "decisions": scorer.decisions, "probes": []})
            row = {"ant": ant, "route": route, "preprocessing": method, "memory": "spike_overlap",
                   "role": "paired-comparison", "trace": name, "memory_views": len(codes),
                   "training_images_sha256": bank_hash, "acquisition_sha256": acquisition_hash,
                   "memory_fingerprint": memory_hash, "memory_active_fraction": float((codes > 0).float().mean()),
                   "stream_activity": [float((x > 0).float().mean()) for x in codes.chunk(2, 1)],
                   **result, "elapsed_s": time.perf_counter() - started}
            rows.append(row)
            with (out / "results.jsonl").open("a") as h: h.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", rows)


if __name__ == "__main__":
    main()
