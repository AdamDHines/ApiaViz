"""Reproducible study runner. Invoke with python -m apiaviz.research --help."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .circuit import CircuitConfig
from .encoders import VisualEncoder, DeepEncoder, EncoderConfig, MODES, REPRESENTATIONS
from .metrics import code_metrics, cluster_bootstrap, reward_bootstrap
from .navigation import World, Scorer, evaluate_route, memory_bank, make_memory
from .stimuli import CONDITIONS, flower_views, perturb
from apiaviz.nav.retino_kc import RewardMBON, AntiHebbianMBON
from apiaviz.src.flowers import affine_scan, choose_rewarded, stratified_folds


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def encode(encoder, images, batch=16, measured=None):
    chunks = []
    spiking = isinstance(encoder, VisualEncoder) and encoder.config.mode in ("threshold", "feedback", "adaptive")
    for i in range(0, len(images), batch):
        view = images[i:i + batch]
        if measured is not None and spiking:
            trace = encoder.diagnostics(view)
            chunks.append(trace["codes"])
            measured["views"] = measured.get("views", 0) + len(view)
            for field, output in (("kc_spikes", "counts"), ("pn_spikes", "pn_counts")):
                measured[field] = measured.get(field, 0.) + sum(float(s[output].sum()) for s in trace["streams"])
        else:
            chunks.append(encoder(view))
    return torch.cat(chunks)


def specifications(args):
    for seed, rep in itertools.product(args.seeds, args.representations):
        if rep in ("mobilenet", "dinov2"):
            for pooling, sensory in itertools.product(args.deep_pooling, args.deep_sensory):
                yield {"representation": rep, "seed": seed, "pooling": pooling, "sensory": sensory}
            continue
        for mode, sparsity in itertools.product(args.modes, args.sparsities):
            spiking = mode in ("threshold", "feedback", "adaptive")
            durations = args.durations if spiking else [50.]
            perturbations = args.perturbations if spiking else ["none"]
            for duration, alteration in itertools.product(durations, perturbations):
                config = EncoderConfig(rep, mode, args.code_dim, tuple(args.pool), args.fan_in,
                                       sparsity, seed, args.spike_readout, alteration, args.ablate)
                circuit = CircuitConfig(duration_ms=duration, dt_ms=args.dt)
                yield {"encoder": asdict(config), "circuit": asdict(circuit), "seed": seed,
                       "representation": rep, "calibration_mode": args.calibration_mode}


def make_encoder(spec, development, args, outdir):
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    if "encoder" in spec:
        config = EncoderConfig(**spec["encoder"])
        circuit = CircuitConfig(**spec["circuit"])
        encoder = VisualEncoder(config, circuit).to(args.device)
        if config.perturbation != "none" and args.calibration_mode == "fixed":
            base = VisualEncoder(replace(config, perturbation="none"), circuit).to(args.device)
            base.calibrate(development)
            for target, source in zip(encoder.streams, base.streams):
                target.threshold.copy_(source.threshold)
            encoder.calibration = {"kind": "inherited-unperturbed", "reference": base.calibration}
        else:
            encoder.calibrate(development)
        metadata = encoder.metadata()
        torch.save({"metadata": metadata, "state_dict": encoder.state_dict()}, outdir / f"encoder-{identity}.pt")
        if args.record_spikes and config.mode in ("threshold", "feedback", "adaptive"):
            trace = encoder.diagnostics(development[:2], record=True)
            torch.save(trace, outdir / f"spikes-{identity}.pt")
        if encoder.calibration.get("kind") == "development-only":
            achieved = [round(s["achieved"], 4) for s in encoder.calibration["streams"]]
            print(f"  calibration {identity}: target={config.sparsity:g}, achieved per stream={achieved}", flush=True)
    else:
        checkpoint = getattr(args, spec["representation"] + "_checkpoint")
        if checkpoint is None:
            raise ValueError(f"--{spec['representation']}-checkpoint is required")
        encoder = DeepEncoder(spec["representation"], checkpoint, spec["pooling"], spec["sensory"],
                              args.dino_repo).to(args.device)
        metadata = {**spec, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint)}
        if spec["representation"] == "dinov2":
            metadata["implementation_revision"] = subprocess.check_output(
                ["git", "-C", str(args.dino_repo), "rev-parse", "HEAD"], text=True).strip()
    encoder.eval()
    # Warm up before profiling; timing includes preprocessing and spike simulation.
    encode(encoder, development[:1], args.batch)
    synchronize(args.device)
    started = time.perf_counter()
    codes = encode(encoder, development, args.batch)
    synchronize(args.device)
    metadata.update(id=identity, fingerprint=fingerprint(encoder),
                    development_ms_per_view=1000 * (time.perf_counter() - started) / len(development),
                    stored_tensor_bytes=sum(t.numel() * t.element_size() for t in encoder.state_dict().values()),
                    development_code_metrics=code_metrics(codes))
    if isinstance(encoder, VisualEncoder) and encoder.config.mode in ("threshold", "feedback", "adaptive"):
        trace = encoder.diagnostics(development[:min(args.batch, len(development))])
        metadata["development_kc_spikes_per_view"] = sum(float(s["counts"].sum(1).mean()) for s in trace["streams"])
        metadata["development_pn_spikes_per_view"] = sum(float(s["pn_counts"].sum(1).float().mean()) for s in trace["streams"])
    write_json(outdir / f"encoder-{identity}.json", metadata)
    return identity, encoder, metadata


def condition_grid(args):
    for condition in args.conditions:
        for severity in ([0.] if condition in ("clean", "gray") else args.severities):
            yield condition, severity


def append_result(outdir, result):
    with (outdir / "results.jsonl").open("a") as handle:
        handle.write(json.dumps(result, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("code_metrics", "perturbation_metrics")}), flush=True)


def run_navigation(args, outdir, specs):
    if set(args.ants) & set(args.development_ants):
        raise ValueError("Evaluation ants and development ants must be disjoint")
    results = []
    dataset_hashes = {name: file_hash(args.world_dir / name) for name in ("AntData.mat", "world5000_gray.mat")}
    write_json(outdir / "dataset.json", dataset_hashes)
    for environment_seed in args.environment_seeds:
        world = World(args.world_dir, args.device, environment_seed, args.landmark_fraction, args.chroma)
        dev_jobs = world.jobs(args.development_ants)
        jobs = world.jobs(args.ants, args.routes)
        if args.max_jobs:
            jobs = jobs[:args.max_jobs]
        if not dev_jobs or not jobs:
            raise ValueError("No development/evaluation routes match the requested ants")
        development, development_ids = [], []
        # Sample equally across development ants, without behavioural labels.
        for ant in args.development_ants:
            candidates = [job for job in dev_jobs if job[0] == ant]
            if not candidates:
                continue
            a, r = candidates[0]
            pos, heading = world.route(a, r)
            indices = np.linspace(0, len(heading) - 1,
                                  max(1, args.calibration_views // len(args.development_ants)), dtype=int)
            development.append(world.render(pos[indices], heading[indices]))
            development_ids.extend([[a, r, int(i)] for i in indices])
        development = torch.cat(development)
        envdir = outdir / f"world-{environment_seed}"
        envdir.mkdir()
        write_json(envdir / "split.json", {"development": development_ids, "evaluation": jobs})
        for spec in specs:
            identity, encoder, metadata = make_encoder(spec, development, args, envdir)
            measured = {}
            encode_fn = lambda model, images: encode(model, images, args.batch, measured)
            for ant, route in jobs:
                positions, headings = world.route(ant, route)
                for viewpoints in args.viewpoints:
                    raw, memory_positions = memory_bank(world, positions, headings, viewpoints, args.corridor_width)
                    measured.clear()
                    codes = encode_fn(encoder, raw)
                    diagnostics = code_metrics(codes, memory_positions)
                    if measured:
                        diagnostics.update({f"memory_{key}_per_view": measured[key] / measured["views"]
                                            for key in ("kc_spikes", "pn_spikes")})
                    for readout in args.readouts:
                        # Graded readout for continuous deep descriptors / count experiments.
                        graded = "encoder" not in spec or args.spike_readout == "count"
                        memory = make_memory(codes, readout, args.segments, graded)
                        for (condition, severity), protocol in itertools.product(condition_grid(args), args.protocols):
                            scorer = Scorer(world, encoder, memory, encode_fn, condition, severity, environment_seed)
                            measured.clear()
                            summary = evaluate_route(positions, headings, scorer, protocol, args.max_steps, args.offline_stride)
                            if measured:
                                summary.update({f"test_{key}_per_view": measured[key] / measured["views"]
                                                for key in ("kc_spikes", "pn_spikes")})
                            row = {"task": "nav", "encoder": identity, "ant": ant, "route": route,
                                   "environment_seed": environment_seed, "wiring_seed": spec["seed"],
                                   "readout": readout, "protocol": protocol, "viewpoints": viewpoints,
                                   "condition": condition, "severity": severity, "code_metrics": diagnostics}
                            trace = summary.pop("trace")
                            trace_id = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]
                            write_json(envdir / f"trace-{trace_id}.json", trace)
                            row.update(summary, trace=str((envdir / f"trace-{trace_id}.json").relative_to(outdir)))
                            if scorer.perturbations:
                                row["perturbation_metrics"] = {k: float(np.mean([x[k] for x in scorer.perturbations]))
                                                               for k in scorer.perturbations[0]}
                            append_result(outdir, row)
                            results.append(row)
            if metadata["fingerprint"] != fingerprint(encoder):
                raise RuntimeError("Encoder state changed during evaluation")
    groups = {}
    for row in results:
        key = tuple(row[k] for k in ("encoder", "readout", "protocol", "viewpoints", "condition", "severity"))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        metrics = [k for k in rows[0] if k in ("reached_nest", "heading_error_deg", "final_nest_distance_m",
                   "route_deviation_mean_m", "path_length_m", "corrective_resets", "familiarity_margin")]
        summary.append({"condition_key": list(key), "metrics": {
            metric: cluster_bootstrap([float(r[metric]) for r in rows], [r["ant"] for r in rows], samples=args.bootstrap)
            for metric in metrics}})
    write_json(outdir / "summary.json", summary)


def flower_split(args):
    rng = np.random.default_rng(args.split_seed)
    paths, labels, development, class_names = [], [], [], []
    for directory in sorted(args.flower_dir.iterdir()):
        images = sorted(directory.glob("*.jpg")) if directory.is_dir() else []
        if not images:
            continue
        if len(images) < args.folds + 2:
            raise ValueError(f"Insufficient images in {directory}")
        images = [images[i] for i in rng.permutation(len(images))]
        ndev = max(1, min(len(images) // 5, args.development_per_class))
        development.extend(images[:ndev])
        selected = images[ndev:ndev + args.per_class]
        if len(selected) < args.folds:
            raise ValueError("per-class must provide at least one image per fold")
        labels.extend([len(class_names)] * len(selected))
        paths.extend(selected)
        class_names.append(directory.name)
    if len(class_names) < 2:
        raise ValueError("Need at least two flower class directories")
    return paths, np.asarray(labels), development, class_names


def run_flowers(args, outdir, specs):
    paths, labels, dev_paths, classes = flower_split(args)
    write_json(outdir / "split.json", {"development": list(map(str, dev_paths)),
              "evaluation": list(map(str, paths)), "labels": labels.tolist(), "classes": classes,
              "sha256": {str(p): file_hash(p) for p in paths + dev_paths}})
    dev_indices = np.linspace(0, len(dev_paths) - 1, min(len(dev_paths), args.calibration_views), dtype=int)
    calibration_paths = [dev_paths[i] for i in dev_indices]
    write_json(outdir / "calibration-images.json", list(map(str, calibration_paths)))
    development = flower_views(calibration_paths, args.object_size, args.canvas_size).to(args.device)
    base = flower_views(paths, args.object_size, args.canvas_size).to(args.device)
    rewarded = choose_rewarded(len(classes), args.n_rewarded, args.reward_seed)
    reward = np.isin(labels, rewarded)
    folds = stratified_folds(labels, args.folds, args.split_seed, args.repeats)
    write_json(outdir / "folds.json", {"rewarded_classes": [classes[i] for i in rewarded],
                "folds": [{"train": tr.tolist(), "test": te.tolist()} for tr, te in folds]})
    for spec in specs:
        identity, encoder, metadata = make_encoder(spec, development, args, outdir)
        # Memory acquisition always uses clean views, independently of test corruption and K.
        train_codes = torch.stack([encode(encoder, affine_scan(base, fix), args.batch).cpu()
                                   for fix in range(args.training_fixations)])
        for condition, severity in condition_grid(args):
            stack, corruption_metrics, spike_metrics = [], [], []
            sequence_state = None
            if args.persist_fixations and (not isinstance(encoder, VisualEncoder) or encoder.config.mode not in
                                           ("threshold", "feedback", "adaptive")):
                raise ValueError("--persist-fixations requires exclusively spiking encoders")
            for fix in range(max(args.fixations)):
                images, measured = perturb(affine_scan(base, fix), condition, severity,
                                           args.reward_seed + fix * 10000)
                corruption_metrics.append(measured)
                if args.persist_fixations:
                    trace = encoder.diagnostics(images, state=sequence_state)
                    sequence_state = [s["state"] for s in trace["streams"]]
                    code = trace["codes"]
                    measured = {"views": len(images),
                                "kc_spikes": sum(float(s["counts"].sum()) for s in trace["streams"]),
                                "pn_spikes": sum(float(s["pn_counts"].sum()) for s in trace["streams"])}
                else:
                    measured = {}
                    code = encode(encoder, images, args.batch, measured)
                spike_metrics.append(measured)
                stack.append(code.cpu())
            stack = torch.stack(stack)
            for shots in args.shots:
                for count in args.fixations:
                    rewards, scores, decisions, ids, correct = [], [], [], [], []
                    for fold_id, (train, test) in enumerate(folds):
                        if shots:
                            rng = np.random.default_rng(args.split_seed + fold_id)
                            train = np.concatenate([rng.permutation(train[labels[train] == cl])[:shots]
                                                    for cl in range(len(classes))])
                        fit = train_codes[:, train].reshape(-1, train_codes.shape[-1])
                        graded = "encoder" not in spec or args.spike_readout == "count"
                        if args.flower_task == "reward":
                            mbon = RewardMBON(fit.shape[-1], graded=graded)
                            mbon.store(fit, np.tile(reward[train], args.training_fixations))
                            training_scores = mbon(fit).reshape(args.training_fixations, -1).mean(0).numpy()
                            threshold = .5 * (training_scores[reward[train]].mean() + training_scores[~reward[train]].mean())
                            scale = max(float(training_scores.std()), 1e-6)
                            evidence = mbon(stack[:count, test].reshape(-1, fit.shape[-1])).reshape(count, -1).mean(0).numpy()
                            rewards.extend(reward[test].tolist())
                            scores.extend(((evidence - threshold) / scale).tolist())
                            decisions.extend((evidence > threshold).tolist())
                        else:
                            novelty = []
                            for cl in range(len(classes)):
                                memory = AntiHebbianMBON(fit.shape[-1], graded=graded)
                                memory.store(train_codes[:, train[labels[train] == cl]].reshape(-1, fit.shape[-1]))
                                novelty.append(memory(stack[:count, test].reshape(-1, fit.shape[-1])).reshape(count, -1).mean(0))
                            prediction = torch.stack(novelty, 1).argmin(1).numpy()
                            correct.extend((prediction == labels[test]).astype(float).tolist())
                        ids.extend(test.tolist())
                    if args.flower_task == "reward":
                        metrics = reward_bootstrap(rewards, scores, decisions, ids, samples=args.bootstrap)
                        predictions = {"image_id": ids, "reward": rewards, "score": scores, "go": decisions}
                    else:
                        metrics = {"accuracy": cluster_bootstrap(correct, ids, samples=args.bootstrap)}
                        predictions = {"image_id": ids, "correct": correct}
                    record = {"task": "flowers", "flower_task": args.flower_task, "encoder": identity,
                              "condition": condition, "severity": severity, "shots_per_class": shots or "all",
                              "fixations": count, "metrics": metrics, "persist_fixations": args.persist_fixations,
                              "code_metrics": code_metrics(stack[:count].flatten(0, 1)),
                              "perturbation_metrics": {k: float(np.mean([m[k] for m in corruption_metrics[:count]]))
                                                       for k in corruption_metrics[0]}}
                    pred_id = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:16]
                    if spike_metrics[0]:
                        record.update({f"test_{key}_per_view": sum(s[key] for s in spike_metrics[:count]) /
                                       sum(s["views"] for s in spike_metrics[:count]) for key in ("kc_spikes", "pn_spikes")})
                    write_json(outdir / f"predictions-{pred_id}.json", predictions)
                    record["predictions"] = f"predictions-{pred_id}.json"
                    append_result(outdir, record)
        if metadata["fingerprint"] != fingerprint(encoder):
            raise RuntimeError("Encoder state changed during flower evaluation")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("nav", "flowers"), default="nav")
    p.add_argument("--representations", nargs="+", choices=(*REPRESENTATIONS, "mobilenet", "dinov2"), default=["gray", "colour", "opponent", "apiaviz"])
    p.add_argument("--modes", nargs="+", choices=MODES, default=["kwta", "threshold", "feedback", "adaptive"])
    p.add_argument("--code-dim", type=int, default=4000, help="TOTAL KCs, divided equally across streams")
    p.add_argument("--pool", nargs=2, type=int, default=[8, 64])
    p.add_argument("--fan-in", type=int, default=10)
    p.add_argument("--sparsities", nargs="+", type=float, default=[.05])
    p.add_argument("--durations", nargs="+", type=float, default=[50.])
    p.add_argument("--dt", type=float, default=1.)
    p.add_argument("--spike-readout", choices=("binary", "count", "early"), default="binary")
    p.add_argument("--perturbations", nargs="+", choices=("none", "no-inhibition", "no-adaptation", "shuffle-times"), default=["none"])
    p.add_argument("--calibration-mode", choices=("matched", "fixed"), default="matched")
    p.add_argument("--calibration-views", type=int, default=24)
    p.add_argument("--ablate", choices=("none", "hex", "adapt", "opponency", "dog"), default="none")
    p.add_argument("--seeds", nargs="+", type=int, default=[7])
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=["clean"])
    p.add_argument("--severities", nargs="+", type=float, default=[.5, 1.])
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output", type=Path, default=Path("apiaviz/output/studies"))
    p.add_argument("--list", action="store_true", help="Print the encoder matrix without loading data or models")
    p.add_argument("--record-spikes", action="store_true")
    p.add_argument("--world-dir", type=Path, default=Path("apiaviz/mbant/data/antview"))
    p.add_argument("--ants", nargs="+", type=int, default=list(range(4, 16)))
    p.add_argument("--development-ants", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--routes", nargs="*", type=int, default=[])
    p.add_argument("--max-jobs", type=int, default=0, help="Explicit route cap for smoke tests; 0 means all")
    p.add_argument("--environment-seeds", nargs="+", type=int, default=[99])
    p.add_argument("--landmark-fraction", type=float, default=.5)
    p.add_argument("--chroma", type=float, default=60.)
    p.add_argument("--viewpoints", nargs="+", type=int, default=[1, 9])
    p.add_argument("--corridor-width", type=float, default=.2)
    p.add_argument("--readouts", nargs="+", choices=("cosine", "single", "population"), default=["cosine", "single", "population"])
    p.add_argument("--segments", type=int, default=16)
    p.add_argument("--protocols", nargs="+", choices=("offline", "reset", "free"), default=["offline", "reset", "free"])
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--offline-stride", type=int, default=10)
    p.add_argument("--flower-dir", type=Path, default=Path("apiaviz/dataset/17flowers"))
    p.add_argument("--flower-task", choices=("reward", "classify"), default="reward")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--development-per-class", type=int, default=4)
    p.add_argument("--object-size", type=int, default=75)
    p.add_argument("--canvas-size", type=int, default=135)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--reward-seed", type=int, default=7)
    p.add_argument("--n-rewarded", type=int, default=5)
    p.add_argument("--shots", nargs="+", type=int, default=[0], help="Examples per class; 0 uses all training images")
    p.add_argument("--fixations", nargs="+", type=int, default=[1, 8])
    p.add_argument("--training-fixations", type=int, default=1)
    p.add_argument("--persist-fixations", action="store_true")
    p.add_argument("--deep-pooling", nargs="+", choices=("global", "spatial"), default=["global", "spatial"])
    p.add_argument("--deep-sensory", nargs="+", choices=("gb", "rgb"), default=["gb", "rgb"])
    p.add_argument("--mobilenet-checkpoint", type=Path)
    p.add_argument("--dinov2-checkpoint", type=Path)
    p.add_argument("--dino-repo", type=Path)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    positive = ("code_dim", "fan_in", "calibration_views", "threads", "batch", "bootstrap", "segments",
                "max_steps", "offline_stride", "per_class", "development_per_class", "folds", "repeats", "training_fixations")
    if any(getattr(args, name) < 1 for name in positive) or min(args.fixations + args.viewpoints) < 1:
        p.error("Counts, batches, fixations and viewpoints must be positive")
    if args.folds < 2 or min(args.shots + args.severities) < 0 or args.max_jobs < 0:
        p.error("folds must be >=2; shots, severities and max-jobs must be nonnegative")
    specs = list(specifications(args))
    if args.persist_fixations and any(spec.get("encoder", {}).get("mode") not in
                                     ("threshold", "feedback", "adaptive") for spec in specs):
        p.error("--persist-fixations requires exclusively spiking encoders")
    if args.list:
        print(json.dumps(specs, indent=2))
        return
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    # Process ID disambiguates concurrent runs started within the same microsecond.
    outdir = args.output / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}"
    outdir.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"settings": settings, "specifications": specs, "torch": torch.__version__,
                "numpy": np.__version__, "status": "running",
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())}
    manifest["source_sha256"] = {str(path): file_hash(path) for path in sorted(Path("apiaviz").rglob("*.py"))
                                 if "output" not in path.parts and "data" not in path.parts}
    manifest["packages"] = {name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "Pillow")}
    write_json(outdir / "manifest.json", manifest)
    args.device = torch.device(args.device)
    print(f"Study output: {outdir}", flush=True)
    try:
        (run_navigation if args.task == "nav" else run_flowers)(args, outdir, specs)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(outdir / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    write_json(outdir / "manifest.json", manifest)
    print(f"Completed: {outdir}", flush=True)
