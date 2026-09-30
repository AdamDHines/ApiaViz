"""Independent checks of stored study provenance, decisions and motion."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research.study import file_hash

HERE=Path(__file__).parent


def hash_tensor(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def selection(scores,offsets):
    scores=np.asarray([np.inf if s is None else s for s in scores]);valid=np.isfinite(scores)
    silent=not valid.any() or np.ptp(scores[valid])<=1e-6
    if silent:return int(np.argmin(abs(offsets))),True
    candidates=np.flatnonzero(valid & np.isclose(scores,scores[valid].min(),rtol=0,atol=1e-6))
    return int(candidates[np.argmin(abs(offsets[candidates]))]),False


def main():
    report=dict(runs=[],trajectories=0,steering_decisions=0,probe_decisions=0,
                errors=[],scope="Stored source/checkpoint/stimulus hashes; shared settings and banks; independent reconstruction of scan choices, free-motion trajectories, endpoint metrics and common-pose probe summaries. Not a complete independent rerun of every neural computation.")
    for name in json.loads((HERE/"runs.json").read_text())["runs"]:
        run=ROOT/name;manifest=json.loads((run/"manifest.json").read_text())
        assert manifest["status"]=="complete",name
        assert file_hash(HERE/"protocol.json")==manifest["protocol_sha256"]
        for path,h in manifest["dataset_sha256"].items():assert file_hash(ROOT/manifest["settings"]["world_dir"]/path)==h
        with zipfile.ZipFile(run/"source.zip") as archive:
            for path,h in manifest["source_sha256"].items():assert hashlib.sha256(archive.read(path)).hexdigest()==h
        for seed,meta in manifest["encoders"].items():
            checkpoint=torch.load(run/f"encoder-{seed}.pt",weights_only=True,map_location="cpu")
            assert json.dumps(checkpoint["metadata"],sort_keys=True)==json.dumps(meta,sort_keys=True)
            digest=hashlib.sha256()
            for key,value in sorted(checkpoint["state_dict"].items()):
                digest.update(key.encode());digest.update(value.contiguous().numpy().tobytes())
            assert digest.hexdigest()==meta["fingerprint"]
        geometry={};stimuli={}
        for ant in manifest["settings"]["ants"]:
            geo=json.loads((run/f"ant-{ant}-geometry.json").read_text());geometry[ant]=geo
            images=torch.load(run/f"ant-{ant}-images.pt",weights_only=True,map_location="cpu")
            # route_corridor_bank explicitly stores both acquisition arrays as float32.
            digest=hashlib.sha256(np.asarray(geo["teaching_positions"],dtype=np.float32).tobytes()+np.asarray(geo["teaching_headings"],dtype=np.float32).tobytes()).hexdigest()
            stimuli[ant]=dict(training_images_sha256=hash_tensor(images["training"]),probe_images_sha256=hash_tensor(images["probes"]),acquisition_sha256=digest)
        count=0
        for line in (run/"results.jsonl").read_text().splitlines():
            row=json.loads(line);geo=geometry[row["ant"]]
            for key,h in stimuli[row["ant"]].items():assert row[key]==h,(row["trace"],key)
            route=np.array(geo["route"]);position=route[0].copy();heading=geo["headings"][0]
            saved=json.loads((run/row["trace"]).read_text());trace=saved["trajectory"];decisions=saved["decisions"]
            assert len(trace)==len(decisions)==row["steps"]
            offsets=np.arange(60.,-61.,-10.);errors=[];continuous=[]
            for index,(entry,decision) in enumerate(zip(trace,decisions),1):
                assert np.linalg.norm(position-route[-1])>.2
                assert entry["step"]==index and entry["reset_to"] is None
                np.testing.assert_allclose(decision["position"],position,atol=1e-9,rtol=0)
                np.testing.assert_allclose(decision["headings"],heading+offsets,atol=1e-9,rtol=0)
                winner,silent=selection(decision["scores"],offsets)
                heading+=offsets[winner]
                np.testing.assert_allclose(entry["heading"],heading,atol=1e-9,rtol=0)
                assert entry["no_evidence"]==silent
                position+=.1*np.array([np.cos(np.radians(heading)),np.sin(np.radians(heading))])
                np.testing.assert_allclose(entry["position"],position,atol=1e-9,rtol=0)
                distance=np.linalg.norm(route-position,axis=1).min();errors.append(distance)
                np.testing.assert_allclose(entry["deviation_m"],distance,atol=1e-9,rtol=0)
                segments=np.diff(route,axis=0);delta=position-route[:-1]
                t=np.clip(np.einsum("ij,ij->i",delta,segments)/np.maximum(np.einsum("ij,ij->i",segments,segments),1e-20),0,1)
                continuous.append(np.linalg.norm(delta-t[:,None]*segments,axis=1).min())
            expected=dict(route_deviation_mean_m=np.mean(errors),route_deviation_max_m=max(errors),
                          deviation_median_m=np.median(errors),polyline_mean_m=np.mean(continuous),
                          first50_polyline_mean_m=np.mean(continuous[:50]),path_length_m=.1*len(trace),
                          final_nest_distance_m=np.linalg.norm(position-route[-1]))
            for key,value in expected.items():np.testing.assert_allclose(row[key],value,atol=1e-9,rtol=0)
            assert row["reached_nest"]==bool(expected["final_nest_distance_m"]<=.2)
            assert row["corrective_resets"]==0
            assert row["reached_nest"] or len(trace)==200
            assert row["silent_scans"]==sum(r["no_evidence"] for r in trace)
            conditions={}
            for condition in ("clean","brightness"):
                probes=[p for p in saved["probes"] if p["condition"]==condition]
                assert len(probes)==len(geo["probes"])
                conditions[condition]=probes
                for probe,planned in zip(probes,geo["probes"]):
                    for key,value in planned.items():assert probe[key]==value
                    winner,silent=selection(probe["scores"],offsets)
                    assert probe["turn_deg"]==offsets[winner] and probe["no_evidence"]==silent
                    assert probe["restoring"]==bool(not silent and probe["lateral_m"]*np.sin(np.radians(offsets[winner]))<0)
                    # Stored scores come from float32 Torch outputs; preserve
                    # that arithmetic when reconstructing their subtraction.
                    scores=np.array(probe["scores"],dtype=np.float32)
                    np.testing.assert_allclose(probe["tangent_margin"],scores[abs(offsets)>=20].min()-scores[6],rtol=0,atol=1e-9)
                centre=[p for p in probes if p["lateral_m"]==0]
                off=[p for p in probes if p["lateral_m"]!=0]
                quantities={"heading_error_deg":np.mean([180 if p["no_evidence"] else abs(p["turn_deg"]) for p in centre]),
                            "restoring_fraction":np.mean([p["restoring"] for p in off]),
                            "tangent_margin":np.mean([p["tangent_margin"] for p in centre])}
                for key,value in quantities.items():np.testing.assert_allclose(row[f"{condition}_{key}"],value,atol=1e-9,rtol=0)
            change=np.mean([abs(a["turn_deg"]-b["turn_deg"]) for a,b in zip(conditions["clean"],conditions["brightness"])])
            np.testing.assert_allclose(row["brightness_heading_change_deg"],change,atol=1e-9,rtol=0)
            report["trajectories"]+=1;report["steering_decisions"]+=len(trace);report["probe_decisions"]+=len(saved["probes"]);count+=1
        report["runs"].append(dict(path=name,trials=count,provenance_verified=True))
    assert report["trajectories"]==json.loads((HERE/"protocol.json").read_text())["navigation_trials"]
    (HERE/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":main()
