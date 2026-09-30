"""Verify sparse rendering pixels and reproduce the four recorded pilot paths."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .navigation import World,Scorer,evaluate_route
from .fast_render import accelerate_world
from .paper_baselines import FrontendEncoder
from .openloop import acquisition_views
from .spike_overlap import SpikeOverlapMemory
from .study import encode,write_json

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    world=World(Path("apiaviz/mbant/data/antview"));original=world.renderer
    accelerate_world(world);rng=np.random.default_rng(20260929)
    count=0;slow=fast=0.
    for ant in range(1,16):
        positions,headings=world.route(ant,2)
        for _ in range(8):
            idx=int(rng.integers(len(positions)));pos=positions[idx]+rng.uniform(-.7,.7,2)
            heading=float(headings[idx]+rng.uniform(-180,180))
            t=time.perf_counter();a=original.render_single(*pos,world.ic.eye_height,heading);slow+=time.perf_counter()-t
            t=time.perf_counter();b=world.renderer.render_single(*pos,world.ic.eye_height,heading);fast+=time.perf_counter()-t
            torch.testing.assert_close(a,b,atol=0,rtol=0);count+=1
    report=dict(images=count,pixel_mismatches=0,original_seconds=slow,sparse_seconds=fast,
                speedup=slow/fast,pilot_paths=[])
    model=FrontendEncoder()
    for root in ("20260929T014349.800124Z-62986","20260929T014349.800138Z-62988","20260929T014349.800161Z-62987","20260929T014349.800124Z-62989"):
        run=Path("apiaviz/output/paper-baselines")/root
        row=next(json.loads(x) for x in (run/"results.jsonl").read_text().splitlines() if json.loads(x)["preprocessing"]=="apiaviz")
        positions,headings=world.route(row["ant"],2)
        pp,hh=acquisition_views(positions,headings,9,.2,0.)
        memory=SpikeOverlapMemory(encode(model,world.render(pp,hh)))
        scorer=Scorer(world,model,memory,encode,"clean",0.,99)
        result=evaluate_route(positions,headings,scorer,"free",max_steps=200)
        saved=json.loads((run/row["trace"]).read_text())
        assert result["trace"]==saved["trajectory"],row["ant"]
        score_error=max(float(np.max(np.abs(np.array(a["scores"])-np.array(b["scores"])))) for a,b in zip(scorer.decisions,saved["decisions"]))
        report["pilot_paths"].append(dict(ant=row["ant"],steps=result["steps"],trajectory_exact=True,max_score_difference=score_error))
        print(report["pilot_paths"][-1],flush=True)
    write_json(Path("docs/mechanism-study/renderer-validation.json"),report)
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
