"""Check the most displaced chosen view in every new trial against the old renderer."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from apiaviz.research.navigation import World
from apiaviz.research.fast_render import accelerate_world
HERE=Path(__file__).parent

def main():
    torch.set_num_threads(1)
    world=World(ROOT/"apiaviz/mbant/data/antview");original=world.renderer;accelerate_world(world)
    checked=[];mismatches=[]
    for name in json.loads((HERE/"runs.json").read_text())["runs"]:
        run=ROOT/name;assert json.loads((run/"manifest.json").read_text())["status"]=="complete"
        for line in (run/"results.jsonl").read_text().splitlines():
            row=json.loads(line);data=json.loads((run/row["trace"]).read_text())
            idx=int(np.argmax([t["deviation_m"] for t in data["trajectory"]]))
            entry,decision=data["trajectory"][idx],data["decisions"][idx]
            args=(*decision["position"],world.ic.eye_height,entry["heading"])
            a,b=original.render_single(*args),world.renderer.render_single(*args)
            item=dict(ant=row["ant"],seed=row["seed"],method=row["preprocessing"],trace=row["trace"],step=idx+1,
                      deviation_m=entry["deviation_m"],pixels_different=int((a!=b).sum()))
            checked.append(item)
            if item["pixels_different"]:mismatches.append(item)
    report=dict(views_checked=len(checked),mismatches=mismatches,checks=checked,
                selection="For every trial: query position and chosen heading at the step with maximum post-movement route deviation.")
    (HERE/"study-image-validation.json").write_text(json.dumps(report,indent=2)+"\n")
    assert len(checked)==440 and not mismatches
    print(json.dumps({k:report[k] for k in ("views_checked","mismatches")}))

if __name__=="__main__":main()
