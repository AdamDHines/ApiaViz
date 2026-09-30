"""Independent reconstruction of all candidate decisions and reference reruns."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from apiaviz.research.study import file_hash
HERE=Path(__file__).parent


def choice(scores,offsets):
    score=np.array([np.inf if x is None else x for x in scores]);valid=np.isfinite(score)
    silent=not valid.any() or np.ptp(score[valid])<=1e-6
    if silent:return int(np.argmin(abs(offsets))),True
    candidates=np.flatnonzero(valid & np.isclose(score,score[valid].min(),rtol=0,atol=1e-6))
    return int(candidates[np.argmin(abs(offsets[candidates]))]),False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,default=HERE)
    args=parser.parse_args()
    here=args.study_dir
    protocol=json.loads((here/"protocol.json").read_text())
    observed=[]
    report=dict(trajectories=0,steering_decisions=0,probe_decisions=0,exact_reference_reruns=0,errors=[])
    for name in json.loads((here/"runs.json").read_text())["runs"]:
        run=ROOT/name;m=json.loads((run/"manifest.json").read_text());assert m["status"]=="complete"
        assert file_hash(here/"protocol.json")==m["protocol_sha256"]
        with zipfile.ZipFile(run/"source.zip") as z:
            for path,h in m["source_sha256"].items():assert hashlib.sha256(z.read(path)).hexdigest()==h
        for enc in m["encoders"].values():assert file_hash(ROOT/enc["source"])==enc["sha256"]
        geometry={};previous={}
        for ant,bank in m["stimulus_sources"].items():
            source=ROOT/bank["path"]
            assert file_hash(source/f"ant-{ant}-images.pt")==bank["images_sha256"]
            assert file_hash(source/f"ant-{ant}-geometry.json")==bank["geometry_sha256"]
            old_manifest=json.loads((source/"manifest.json").read_text())
            for f,h in old_manifest["dataset_sha256"].items():assert file_hash(ROOT/old_manifest["settings"]["world_dir"]/f)==h
            geometry[int(ant)]=json.loads((source/f"ant-{ant}-geometry.json").read_text())
            previous[int(ant)]={(r["seed"],r["preprocessing"]):(source,r) for r in map(json.loads,(source/"results.jsonl").read_text().splitlines()) if r["ant"]==int(ant)}
        for r in map(json.loads,(run/"results.jsonl").read_text().splitlines()):
            observed.append((r["ant"],r["seed"],r["preprocessing"]))
            saved=json.loads((run/r["trace"]).read_text());geo=geometry[r["ant"]]
            if r["preprocessing"] in ("apiaviz","sobel_colour","ardin_input"):
                source,old=previous[r["ant"]][r["seed"],r["preprocessing"]]
                old_trace=json.loads((source/old["trace"]).read_text())
                assert saved==old_trace
                for key in old:
                    if key in r and key not in ("elapsed_s",):assert r[key]==old[key],key
                report["exact_reference_reruns"]+=1
            route=np.asarray(geo["route"]);pos=route[0].copy();heading=geo["headings"][0];offsets=np.asarray(geo["scan_offsets"])
            trace=saved["trajectory"];decisions=saved["decisions"]
            assert len(trace)==len(decisions)==r["steps"]
            errors=[];poly=[]
            for i,(step,decision) in enumerate(zip(trace,decisions),1):
                assert np.linalg.norm(pos-route[-1])>.2
                np.testing.assert_allclose(decision["position"],pos,rtol=0,atol=1e-9)
                np.testing.assert_allclose(decision["headings"],heading+offsets,rtol=0,atol=1e-9)
                winner,silent=choice(decision["scores"],offsets);heading+=offsets[winner]
                pos+=.1*np.array([np.cos(np.radians(heading)),np.sin(np.radians(heading))])
                np.testing.assert_allclose(step["position"],pos,rtol=0,atol=1e-9)
                assert step["step"]==i and step["reset_to"] is None and step["heading"]==heading and step["no_evidence"]==silent
                error=np.linalg.norm(route-pos,axis=1).min();errors.append(error)
                np.testing.assert_allclose(step["deviation_m"],error,rtol=0,atol=1e-9)
                delta=np.diff(route,axis=0);v=pos-route[:-1]
                t=np.clip((v*delta).sum(1)/np.maximum((delta*delta).sum(1),1e-20),0,1)
                poly.append(np.linalg.norm(v-t[:,None]*delta,axis=1).min())
            for key,value in dict(route_deviation_mean_m=np.mean(errors),route_deviation_max_m=max(errors),
                 polyline_mean_m=np.mean(poly),first50_polyline_mean_m=np.mean(poly[:50]),deviation_median_m=np.median(errors),
                 path_length_m=.1*len(trace),final_nest_distance_m=np.linalg.norm(pos-route[-1])).items():
                np.testing.assert_allclose(r[key],value,rtol=0,atol=1e-9)
            assert r["reached_nest"]==bool(np.linalg.norm(pos-route[-1])<=.2)
            assert r["reached_nest"] or len(trace)==200
            for p in saved["probes"]:
                winner,silent=choice(p["scores"],offsets)
                assert p["turn_deg"]==offsets[winner] and p["no_evidence"]==silent
                assert p["restoring"]==bool(not silent and p["lateral_m"]*np.sin(np.radians(p["turn_deg"]))<0)
            for condition in ("clean","brightness"):
                probes=[p for p in saved["probes"] if p["condition"]==condition]
                centre=[p for p in probes if p["lateral_m"]==0];off=[p for p in probes if p["lateral_m"]!=0]
                assert len(centre)==8 and len(off)==32
                for key,val in dict(heading_error_deg=np.mean([180 if p["no_evidence"] else abs(p["turn_deg"]) for p in centre]),
                    restoring_fraction=np.mean([p["restoring"] for p in off]),tangent_margin=np.mean([p["tangent_margin"] for p in centre])).items():
                    np.testing.assert_allclose(r[condition+"_"+key],val,rtol=0,atol=1e-9)
            report["trajectories"]+=1;report["steering_decisions"]+=len(trace);report["probe_decisions"]+=len(saved["probes"])
    expected={(a,s,m) for a in protocol["ants"] for s in protocol["wiring_seeds"] for m in protocol["methods"]}
    assert len(observed)==len(expected) and set(observed)==expected
    assert report["trajectories"]==protocol["navigation_trials"]
    expected_references=len(protocol["ants"])*len(protocol["wiring_seeds"])*len(set(protocol["methods"]) & {"apiaviz","sobel_colour","ardin_input"})
    assert report["exact_reference_reruns"]==expected_references
    report["scope"]="Source, stimulus, checkpoint and dataset hashes; reconstructed navigation and probes; exact stored-score, probe and trajectory reproduction for all reference runs. Not a full independent recalculation of all neural computations."
    (here/"audit.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
