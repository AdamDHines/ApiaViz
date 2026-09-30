"""Fixed, paired navigation and common-pose mechanistic experiments.

Read docs/mechanism-study/protocol.json before interpreting these experiments.
This module never tunes a frontend, circuit, memory or controller.
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

from .paper_baselines import FrontendEncoder, METHODS, feature_maps, DEFINITIONS
from .fast_render import accelerate_world
from .navigation import World, Scorer, evaluate_route, choose_heading
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode, fingerprint, file_hash, write_json
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize

ABLATIONS = ("no_hex", "no_adapt", "no_dog", "apia_simple_colour", "sobel_apia_colour")
ALL_METHODS = METHODS + ABLATIONS
ABLATION_DEFINITIONS = {
    "no_hex": "Bypass hexagonal neighbourhood averaging in both ApiaViz streams.",
    "no_adapt": "Bypass local luminance and chromatic adaptation, retain the other ApiaViz stages.",
    "no_dog": "Bypass the entire form filter bank (DoG ON/OFF AND Gaussian low-pass); use pointwise ON/OFF and adapted luminance. This is not an isolated DoG-surround lesion.",
    "apia_simple_colour": "ApiaViz form planes with raw rectified G-B/B-G; replace only the processed colour pathway.",
    "sobel_apia_colour": "Sobel form planes with ApiaViz processed colour planes; reciprocal pathway swap.",
}


def mechanism_maps(images, method, backbone):
    if method in METHODS:
        return feature_maps(images, method, backbone)
    if method in ("no_hex", "no_adapt", "no_dog"):
        maps = backbone(images[:, -2:] * 2-1, return_maps=True, ablate=method[3:])
        return maps["contrast_features"], maps["chromatic_feature"][:, 1:]
    apia = feature_maps(images, "apiaviz", backbone)
    if method == "apia_simple_colour":
        return apia[0], feature_maps(images, "opponent", backbone)[1]
    if method == "sobel_apia_colour":
        return feature_maps(images, "sobel_colour", backbone)[0], apia[1]
    raise ValueError(method)


class MechanismEncoder(FrontendEncoder):
    def __init__(self, method="apiaviz", seed=7, code_dim=8000):
        super().__init__("apiaviz", seed=seed, code_dim=code_dim)
        if method not in ALL_METHODS: raise ValueError(method)
        self.method = method

    @torch.no_grad()
    def currents(self, images):
        if images.ndim != 4 or images.shape[1] not in (2,3) or not torch.isfinite(images).all() or images.min()<0 or images.max()>1:
            raise ValueError("Expected GB/RGB images in [0,1]")
        drives=[]
        for feature, projection in zip(mechanism_maps(images,self.method,self.features.backbone),self.features.legacy):
            flat=adaptive_avg_pool2d_anysize(feature,projection.pool_hw).flatten(1)
            flat=(flat-flat.mean(1,keepdim=True))/(flat.std(1,keepdim=True)+1e-6)
            drives.append(F.relu(flat @ projection.connection))
        return drives


def polyline_distance(points, route):
    points,route=np.asarray(points),np.asarray(route)
    start,delta=route[:-1],np.diff(route,axis=0)
    t=np.sum((points[:,None]-start)*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=1),1e-20)
    closest=start+np.clip(t,0,1)[...,None]*delta
    return np.linalg.norm(points[:,None]-closest,axis=2).min(axis=1)


def probe_grid(positions, headings):
    # Midpoints prevent exact reuse of a centreline learning photograph.
    indices=np.unique(np.linspace(1,len(positions)-3,8).astype(int))
    offsets=np.arange(60.,-61.,-10.)
    records=[]; pp=[]; hh=[]
    for idx in indices:
        centre=(positions[idx]+positions[idx+1])/2
        direction=positions[idx+1]-positions[idx]
        heading=np.degrees(np.arctan2(direction[1],direction[0]))
        radians=np.radians(heading)
        for lateral in (-.3,-.1,0.,.1,.3):
            pos=centre+lateral*np.array([-np.sin(radians),np.cos(radians)])
            records.append(dict(route_index=int(idx),lateral_m=lateral,position=pos.tolist(),heading=float(heading)))
            pp.extend([pos]*len(offsets)); hh.extend(heading+offsets)
    return np.asarray(pp),np.asarray(hh),records,offsets


def code_statistics(codes):
    binary=(codes>0).float()
    # Centreline teaching views only, one for each route location.
    centres=binary[4::9]
    norm=F.normalize(centres,dim=1)
    overlap=norm @ norm.T
    n=len(centres)
    distance=(torch.arange(n)[:,None]-torch.arange(n)[None,:]).abs()
    near=overlap[distance==1].mean()
    far=overlap[distance>=max(2,n//2)].mean()
    centred=centres-centres.mean(0,keepdim=True)
    gram=centred @ centred.T
    participation=gram.trace().square()/gram.square().sum().clamp_min(1e-12)
    lifetime=binary.mean(0)
    return dict(memory_active_fraction=float(binary.mean()),
                stream_activity=[float(x.mean()) for x in binary.chunk(2,1)],
                neighbour_overlap=float(near),distant_overlap=float(far),
                local_distinctiveness=float(near-far),effective_rank=float(participation),
                lifetime_recruited_fraction=float((lifetime>0).float().mean()),
                lifetime_frequency_sd=float(lifetime.std()))


def probe_summary(model,memory,images,records,offsets):
    all_rows=[]; summary={}; selections={}
    for condition,gain in (("clean",1.),("brightness",.6)):
        codes=encode(model,images*gain)
        values=memory(codes).detach().numpy().reshape(len(records),len(offsets))
        rows=[]
        for record,scores in zip(records,values):
            winner,silent=choose_heading(scores,offsets)
            turn=float(offsets[winner]); lateral=record["lateral_m"]
            # Positive means the correct tangent view beats all turns >=20 degrees.
            margin=float(scores[np.abs(offsets)>=20].min()-scores[offsets==0][0])
            rows.append(dict(**record,condition=condition,turn_deg=turn,no_evidence=silent,
                             tangent_margin=margin,
                             restoring=bool(not silent and lateral*np.sin(np.radians(turn))<0),
                             scores=scores.tolist()))
        selections[condition]=np.array([r["turn_deg"] for r in rows])
        centre=[r for r in rows if r["lateral_m"]==0]
        off=[r for r in rows if r["lateral_m"]!=0]
        summary.update({f"{condition}_heading_error_deg":float(np.mean([180. if r["no_evidence"] else abs(r["turn_deg"]) for r in centre])),
                        f"{condition}_tangent_margin":float(np.mean([r["tangent_margin"] for r in centre])),
                        f"{condition}_restoring_fraction":float(np.mean([r["restoring"] for r in off]))})
        all_rows.extend(rows)
    summary["brightness_heading_change_deg"]=float(np.mean(np.abs(selections["brightness"]-selections["clean"])))
    return summary,all_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ants",type=int,nargs="+",required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[19,31,43,59,71])
    p.add_argument("--methods",nargs="+",choices=ALL_METHODS,default=list(ALL_METHODS))
    p.add_argument("--protocol",type=Path,default=Path("docs/mechanism-study/protocol.json"))
    p.add_argument("--world-dir",type=Path,default=Path("apiaviz/mbant/data/antview"))
    p.add_argument("--output",type=Path,default=Path("apiaviz/output/mechanisms"))
    p.add_argument("--threads",type=int,default=1)
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if not set(args.ants)<=set(protocol["ants"]) or not set(args.seeds)<=set(protocol["wiring_seeds"]):
        raise ValueError("Cases must be in the fixed protocol")
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    out=args.output/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+f"-{os.getpid()}")
    out.mkdir(parents=True)
    sources=[f for f in sorted(Path("apiaviz").rglob("*.py")) if "output" not in f.parts and "data" not in f.parts]
    manifest=dict(status="running",protocol=protocol,protocol_sha256=file_hash(args.protocol),
                  settings=dict(ants=args.ants,seeds=args.seeds,methods=args.methods,threads=args.threads,
                                world_dir=str(args.world_dir),max_steps=200,environment_seed=99),
                  torch=str(torch.__version__),numpy=np.__version__,
                  source_sha256={str(f):file_hash(f) for f in sources},
                  dataset_sha256={n:file_hash(args.world_dir/n) for n in ("AntData.mat","world5000_gray.mat")},
                  definitions={**DEFINITIONS,**ABLATION_DEFINITIONS},encoders={})
    with zipfile.ZipFile(out/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sources:archive.write(f,str(f))
    write_json(out/"manifest.json",manifest);print(f"Output: {out}",flush=True)
    models={seed:MechanismEncoder(seed=seed) for seed in args.seeds}
    for seed,model in models.items():
        manifest["encoders"][str(seed)]=dict(encoder=asdict(model.config),circuit=asdict(model.circuit_config),fingerprint=fingerprint(model))
        torch.save(dict(state_dict=model.state_dict(),metadata=manifest["encoders"][str(seed)]),out/f"encoder-{seed}.pt")
    write_json(out/"manifest.json",manifest)
    world=accelerate_world(World(args.world_dir,seed=99))
    for ant in args.ants:
        positions,headings=world.route(ant,2)
        teach_pos,teach_head=acquisition_views(positions,headings,9,.2,0.)
        images=world.render(teach_pos,teach_head)
        training_hash=hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest()
        acquisition_hash=hashlib.sha256(teach_pos.tobytes()+teach_head.tobytes()).hexdigest()
        pp,hh,records,offsets=probe_grid(positions,headings)
        probe_images=world.render(pp,hh)
        probe_hash=hashlib.sha256(probe_images.contiguous().numpy().tobytes()).hexdigest()
        # Archive shared stimuli and geometry so mechanism calculations can be reproduced.
        torch.save(dict(training=images,probes=probe_images),out/f"ant-{ant}-images.pt")
        write_json(out/f"ant-{ant}-geometry.json",dict(route=positions.tolist(),headings=headings.tolist(),
                    teaching_positions=teach_pos.tolist(),teaching_headings=teach_head.tolist(),probes=records,scan_offsets=offsets.tolist()))
        for seed,model in models.items():
            frozen=manifest["encoders"][str(seed)]["fingerprint"]
            for method in args.methods:
                started=time.perf_counter();model.method=method
                codes=encode(model,images); memory=SpikeOverlapMemory(codes)
                memory_hash=fingerprint(memory)
                stats=code_statistics(codes)
                diagnostics,probe_rows=probe_summary(model,memory,probe_images,records,offsets)
                scorer=Scorer(world,model,memory,encode,"clean",0.,99)
                result=evaluate_route(positions,headings,scorer,"free",max_steps=200)
                trace=result.pop("trace")
                errors=polyline_distance([r["position"] for r in trace],positions)
                result.update(polyline_mean_m=float(errors.mean()),first50_polyline_mean_m=float(errors[:50].mean()),
                              deviation_median_m=float(np.median([r["deviation_m"] for r in trace])))
                if fingerprint(model)!=frozen or fingerprint(memory)!=memory_hash:
                    raise RuntimeError("Frozen encoder or memory changed during recall")
                name=f"ant-{ant}-seed-{seed}-{method}.json"
                write_json(out/name,dict(trajectory=trace,decisions=scorer.decisions,probes=probe_rows))
                row=dict(ant=ant,route=2,seed=seed,preprocessing=method,trace=name,
                         training_images_sha256=training_hash,acquisition_sha256=acquisition_hash,
                         probe_images_sha256=probe_hash,memory_fingerprint=memory_hash,memory_views=len(codes),
                         **result,**stats,**diagnostics,elapsed_s=time.perf_counter()-started)
                with (out/"results.jsonl").open("a") as handle:handle.write(json.dumps(row,allow_nan=False)+"\n")
                print(json.dumps({k:row[k] for k in ("ant","seed","preprocessing","reached_nest","route_deviation_mean_m","elapsed_s")}),flush=True)
    manifest["status"]="complete";write_json(out/"manifest.json",manifest)


if __name__=="__main__":main()
