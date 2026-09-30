"""Explicit, untrained frontend candidates; historical defaults are unchanged.

This is a development study on previously inspected routes, not a held-out test.
The candidates isolate filter gain, channel balance, oriented features and
the placement of colour nonlinearities. No parameters are selected from results.
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
import torch
from torch.nn import functional as F

from .paper_baselines import FrontendEncoder, feature_maps, DEFINITIONS as BASELINE_DEFINITIONS
from .fast_render import accelerate_world
from .mechanisms import code_statistics, polyline_distance, probe_summary
from .navigation import World, Scorer, evaluate_route, choose_heading
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize

DEFINITIONS = {
    "apiaviz": "Historical ApiaViz; exact reproduction check.",
    "sobel_colour": "Historical Sobel plus simple colour; exact reproduction check.",
    "unit_dc": "ApiaViz with only its low-pass filter divided by its kernel sum (unit DC gain rather than unit L2 norm).",
    "balanced_form": "ApiaViz with per-plane RMS balance AFTER form pooling, before the unchanged stream standardisation; RMS floor is 5% of shared form RMS.",
    "oriented_form": "Retain ApiaViz hex averaging and local adaptation; replace the complete form bank with Gaussian sigma=1 then signed Sobel x/y/magnitude. Retain Apia colour.",
    "linear_colour": "Retain Apia form; colour is rectified G-B/B-G after hex averaging, without pre-opponency tanh. No change to form or wiring.",
    "balanced_linear": "Combine balanced_form and linear_colour, specified before candidate outcome inspection.",
    "oriented_linear": "Combine oriented_form and linear_colour exactly; no other change to features, wiring or navigation.",
    "ardin_input": BASELINE_DEFINITIONS["ardin_input"],
}
METHODS = tuple(DEFINITIONS)


def refinement_maps(images, method, backbone):
    if method not in METHODS:
        raise ValueError(method)
    if method in ("apiaviz", "sobel_colour", "ardin_input"):
        return feature_maps(images, method, backbone)
    maps = backbone(images[:, -2:] * 2 - 1, return_maps=True)
    form, colour = maps["contrast_features"], maps["chromatic_feature"][:, 1:]
    if method == "unit_dc":
        gain = backbone.contrast_filter.filters.weight[2].sum()
        form = torch.cat([form[:, :2], form[:, 2:] / gain], 1)
    if method in ("oriented_form", "oriented_linear"):
        # feature_maps' Sobel branch uses luminance and is valid for signed
        # adapted signals too; no input range check is performed in that helper.
        form = feature_maps(maps["adapted_luminance"].repeat(1, 2, 1, 1), "sobel_colour", backbone)[0]
    if method in ("linear_colour", "balanced_linear", "oriented_linear"):
        sampled = backbone.spatial_sampler(images[:, -2:] * 2 - 1)
        diff = .5 * (sampled[:, :1] - sampled[:, 1:])
        colour = torch.cat([diff.relu(), (-diff).relu()], 1)
    return form, colour


class RefinementEncoder(FrontendEncoder):
    def __init__(self, method="apiaviz", seed=7, code_dim=8000):
        super().__init__("apiaviz", seed=seed, code_dim=code_dim)
        if method not in METHODS:
            raise ValueError(method)
        self.method = method

    @torch.no_grad()
    def pooled_features(self, images):
        if images.ndim != 4 or images.shape[1] not in (2, 3) or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Expected finite GB/RGB images in [0,1]")
        result = []
        for i, (feature, projection) in enumerate(zip(refinement_maps(images, self.method, self.features.backbone), self.features.legacy)):
            pooled = adaptive_avg_pool2d_anysize(feature, projection.pool_hw)
            if i == 0 and self.method in ("balanced_form", "balanced_linear"):
                rms = pooled.square().mean((2, 3), keepdim=True).sqrt()
                floor = .05 * pooled.square().mean((1, 2, 3), keepdim=True).sqrt()
                pooled = pooled / torch.maximum(rms, floor).clamp_min(1e-6)
            result.append(pooled)
        return result

    @torch.no_grad()
    def currents(self, images):
        result = []
        for pooled, projection in zip(self.pooled_features(images), self.features.legacy):
            flat = pooled.flatten(1)
            flat = (flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-6)
            result.append(F.relu(flat @ projection.connection))
        return result


def stream_probes(model, codes, images, records, offsets):
    """Counterfactual readout of each stream, without rerunning navigation."""
    query = encode(model, images)
    output = {}
    for stream, train, test in zip(("form", "colour"), codes.chunk(2, 1), query.chunk(2, 1)):
        train = F.normalize((train > 0).float(), dim=1)
        test = F.normalize((test > 0).float(), dim=1)
        similarity = test @ train.T
        values, indices = similarity.max(1)
        scores = (1 - values).numpy().reshape(len(records), len(offsets))
        matched = indices.numpy().reshape(len(records), len(offsets)) // 9
        turns, margins, jumps = [], [], []
        for record, score, index in zip(records, scores, matched):
            if record["lateral_m"] != 0:
                continue
            winner, silent = choose_heading(score, offsets)
            turns.append(180. if silent else abs(float(offsets[winner])))
            margins.append(float(score[abs(offsets) >= 20].min() - score[offsets == 0][0]))
            jumps.append(abs(float(index[winner]) - (record["route_index"] + .5)) * .1)
        output[stream + "_heading_error_deg"] = float(np.mean(turns))
        output[stream + "_tangent_margin"] = float(np.mean(margins))
        output[stream + "_matched_progress_error_m"] = float(np.mean(jumps))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ants", type=int, nargs="+", required=True)
    p.add_argument("--protocol", type=Path, default=Path("docs/frontend-deep-dive/protocol.json"))
    p.add_argument("--output", type=Path, default=Path("apiaviz/output/frontend-refinements"))
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    assert set(args.ants) <= set(protocol["ants"])
    methods = protocol["methods"]
    if not methods or len(methods) != len(set(methods)) or not set(methods) <= set(METHODS):
        raise ValueError("Protocol must specify unique supported methods")
    if protocol["environment_seed"] != 99:
        raise ValueError("Saved teaching images require environment seed 99")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = args.output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}")
    out.mkdir(parents=True)
    sources = [f for f in sorted(Path("apiaviz").rglob("*.py")) if "output" not in f.parts and "data" not in f.parts]
    previous = json.loads(Path("docs/mechanism-study/runs.json").read_text())["runs"]
    manifest = dict(status="running", protocol=protocol, protocol_sha256=file_hash(args.protocol),
                    settings=dict(ants=args.ants, seeds=protocol["wiring_seeds"], methods=methods),
                    definitions=DEFINITIONS, source_sha256={str(f): file_hash(f) for f in sources},
                    stimulus_sources={}, encoders={})
    with zipfile.ZipFile(out / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sources:
            archive.write(f, str(f))
    write_json(out / "manifest.json", manifest)
    print(f"Output: {out}", flush=True)
    world = accelerate_world(World(Path("apiaviz/mbant/data/antview"), seed=99))
    for ant in args.ants:
        matches = [Path(r) for r in previous if (Path(r) / f"ant-{ant}-images.pt").exists()]
        assert len(matches) == 1
        source = matches[0]
        bank = source / f"ant-{ant}-images.pt"
        geo_file = source / f"ant-{ant}-geometry.json"
        manifest["stimulus_sources"][str(ant)] = dict(path=str(source), images_sha256=file_hash(bank), geometry_sha256=file_hash(geo_file))
        images = torch.load(bank, weights_only=True, map_location="cpu")
        geo = json.loads(geo_file.read_text())
        positions, headings = world.route(ant, 2)
        np.testing.assert_array_equal(positions, geo["route"])
        offsets = np.asarray(geo["scan_offsets"])
        old = {(r["seed"], r["preprocessing"]): r for r in map(json.loads, (source / "results.jsonl").read_text().splitlines()) if r["ant"] == ant}
        for seed in protocol["wiring_seeds"]:
            model = RefinementEncoder(seed=seed)
            checkpoint = source / f"encoder-{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"])
            frozen = fingerprint(model)
            manifest["encoders"][str(seed)] = dict(fingerprint=frozen, encoder=asdict(model.config), circuit=asdict(model.circuit_config), source=str(checkpoint), sha256=file_hash(checkpoint))
            for method in methods:
                started = time.perf_counter()
                model.method = method
                codes = encode(model, images["training"])
                memory = SpikeOverlapMemory(codes)
                memory_hash = fingerprint(memory)
                stats = code_statistics(codes)
                probes, probe_rows = probe_summary(model, memory, images["probes"], geo["probes"], offsets)
                stream = stream_probes(model, codes, images["probes"], geo["probes"], offsets)
                scorer = Scorer(world, model, memory, encode, "clean", 0., 99)
                result = evaluate_route(positions, headings, scorer, "free", max_steps=200)
                trace = result.pop("trace")
                distance = polyline_distance([r["position"] for r in trace], positions)
                result.update(polyline_mean_m=float(distance.mean()), first50_polyline_mean_m=float(distance[:50].mean()), deviation_median_m=float(np.median([r["deviation_m"] for r in trace])))
                assert fingerprint(model) == frozen and fingerprint(memory) == memory_hash
                if method in ("apiaviz", "sobel_colour", "ardin_input"):
                    prior = old[seed, method]
                    assert memory_hash == prior["memory_fingerprint"]
                    for key in ("route_deviation_mean_m", "steps", "reached_nest"):
                        assert result[key] == prior[key], (ant, seed, method, key)
                    prior_trace = json.loads((source / prior["trace"]).read_text())
                    assert trace == prior_trace["trajectory"]
                    assert scorer.decisions == prior_trace["decisions"]
                name = f"ant-{ant}-seed-{seed}-{method}.json"
                write_json(out / name, dict(trajectory=trace, decisions=scorer.decisions, probes=probe_rows))
                row = dict(ant=ant, route=2, seed=seed, preprocessing=method, trace=name,
                           memory_fingerprint=memory_hash, **result, **stats, **probes, **stream,
                           elapsed_s=time.perf_counter()-started)
                with (out / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps({k: row[k] for k in ("ant", "seed", "preprocessing", "reached_nest", "route_deviation_mean_m")}), flush=True)
        write_json(out / "manifest.json", manifest)
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
