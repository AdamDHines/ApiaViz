"""Recompute navigation metrics and check provenance in saved open-loop runs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from apiaviz.mbant.io_utils import load_ant_data, prepare_route
from .study import file_hash, write_json


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    settings = manifest["settings"]
    root = Path(settings["world_dir"])
    for name, expected in manifest["dataset_sha256"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Dataset mismatch: {name}")
    archived = (run / "source.zip").exists()
    if archived:
        with zipfile.ZipFile(run / "source.zip") as archive:
            for name, expected in manifest["source_sha256"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Archived source mismatch: {name}")
    checkpoint_path = run / "encoder.pt"
    if not checkpoint_path.exists() and settings.get("checkpoint"):
        checkpoint_path = Path(settings["checkpoint"])
        if file_hash(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise ValueError("Original checkpoint file mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", checkpoint)
    for key in ("encoder", "circuit"):
        if json.dumps(metadata[key], sort_keys=True) != json.dumps(manifest[key], sort_keys=True):
            raise ValueError(f"Checkpoint {key} configuration mismatch")
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint["state_dict"].items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != manifest["encoder_fingerprint"]:
        raise ValueError("Encoder fingerprint mismatch")
    ants = load_ant_data(str(root / "AntData.mat"))
    results = run / "results.jsonl"
    rows = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    checked, controller_steps = 0, 0
    for row in rows:
        if "steps" not in row:  # Probe-only runs have no navigation outcome.
            continue
        route, headings, _ = prepare_route(ants[f"Ant{row['ant']}"]["routes"][f"Route{row['route']}"])
        saved = json.loads((run / row["trace"]).read_text())
        trace = saved["trajectory"]
        angle = np.radians(headings[0])
        position = route[0] + settings.get("start_lateral", 0.) * np.array([-np.sin(angle), np.cos(angle)])
        heading = float(headings[0]) + settings.get("start_heading", 0.)
        decisions = saved.get("decisions", [])[len(saved.get("probes", [])):]
        if decisions and len(decisions) != len(trace):
            raise ValueError("Decision/trajectory length mismatch")
        deviations = []
        reached = np.linalg.norm(position - route[-1]) <= .2
        for step, entry in enumerate(trace, 1):
            if reached or entry["reset_to"] is not None or entry["step"] != step:
                raise ValueError(f"Invalid free-navigation trajectory: {row['trace']}")
            decision = decisions[step - 1] if decisions else {}
            if "headings" in decision:
                # Reconstruct the scan decision from saved scores, independently
                # of evaluate_route/choose_heading. Older traces without scan
                # scores and the distinct lateralized policy are not covered.
                offsets = np.arange(60., -61., -10.)
                np.testing.assert_allclose(decision["position"], position, atol=1e-9, rtol=0)
                np.testing.assert_allclose(decision["headings"], heading + offsets, atol=1e-9, rtol=0)
                scores = np.array([np.inf if s is None else s for s in decision["scores"]])
                if scores.shape != offsets.shape:
                    raise ValueError("Scan score/heading count mismatch")
                valid = np.isfinite(scores)
                silent = not valid.any() or np.ptp(scores[valid]) <= 1e-6
                if silent:
                    selected = int(np.argmin(np.abs(offsets)))
                else:
                    candidates = np.flatnonzero(valid & np.isclose(scores, scores[valid].min(), rtol=0, atol=1e-6))
                    selected = candidates[np.argmin(np.abs(offsets[candidates]))]
                np.testing.assert_allclose(entry["heading"], heading + offsets[selected], atol=1e-9, rtol=0)
                if bool(entry["no_evidence"]) != bool(silent):
                    raise ValueError("Decision silence flag mismatch")
                controller_steps += 1
            heading = entry["heading"]
            angle = np.radians(entry["heading"])
            position = position + .1 * np.array([np.cos(angle), np.sin(angle)])
            np.testing.assert_allclose(entry["position"], position, atol=1e-9, rtol=0)
            deviation = float(np.linalg.norm(route - position, axis=1).min())
            np.testing.assert_allclose(entry["deviation_m"], deviation, atol=1e-9, rtol=0)
            deviations.append(deviation)
            reached = np.linalg.norm(position - route[-1]) <= .2
        expected = {"steps": len(trace), "path_length_m": .1 * len(trace),
                    "corrective_resets": 0, "reached_nest": reached,
                    "silent_scans": sum(t["no_evidence"] for t in trace),
                    "final_nest_distance_m": np.linalg.norm(position - route[-1]),
                    "route_deviation_mean_m": np.mean(deviations) if deviations else 0.,
                    "route_deviation_max_m": max(deviations, default=0.)}
        for key, value in expected.items():
            np.testing.assert_allclose(row[key], value, atol=1e-9, rtol=0,
                                       err_msg=f"{row['trace']}: {key}")
        if not reached and len(trace) != settings["max_steps"]:
            raise ValueError("Unexplained early termination")
        probes = saved.get("probes", [])
        if probes:
            fraction = np.mean([not p["no_evidence"] and
                                p["lateral_m"] * np.sin(np.radians(p["turn_deg"])) < 0
                                for p in probes])
            np.testing.assert_allclose(row["restoring_fraction"], fraction, atol=1e-9, rtol=0)
        checked += 1
    return {"run": str(run), "status": manifest["status"], "result_rows": len(rows),
            "trajectories_checked": checked, "source_archive_verified": archived,
            "scan_controller_steps_verified": controller_steps,
            "encoder_fingerprint_verified": True, "encoder_config_verified": True,
            "checkpoint": str(checkpoint_path),
            "dataset_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("apiaviz/output/openloop/audit.json"))
    args = parser.parse_args()
    results, errors = [], []
    for run in args.runs:
        try:
            results.append(audit(run))
        except Exception as exc:
            errors.append({"run": str(run), "error": f"{type(exc).__name__}: {exc}"})
    report = {"runs": results, "errors": errors,
              "trajectories_checked": sum(r["trajectories_checked"] for r in results),
              "scan_controller_steps_verified": sum(r["scan_controller_steps_verified"] for r in results),
              "scope": "Saved trajectory metrics, free-motion kinematics, available scan decisions, checkpoints, datasets and available source archives; not an independent renderer/encoder reproduction. Lateralized-policy decisions and older missing decision records are not reconstructed."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({"trajectories_checked": report["trajectories_checked"], "errors": errors}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
