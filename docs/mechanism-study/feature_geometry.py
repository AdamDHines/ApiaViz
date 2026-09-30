"""Inspect representation geometry before random projection, without fitting."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from apiaviz.research.mechanisms import MechanismEncoder,mechanism_maps,ALL_METHODS
from apiaviz.nav.retino_kc import adaptive_avg_pool2d_anysize
from apiaviz.research.study import file_hash
HERE=Path(__file__).parent

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model=MechanismEncoder(code_dim=128);rows=[]
    for name in json.loads((HERE/"runs.json").read_text())["runs"]:
        run=ROOT/name;manifest=json.loads((run/"manifest.json").read_text())
        for ant in manifest["settings"]["ants"]:
            images=torch.load(run/f"ant-{ant}-images.pt",weights_only=True,map_location="cpu")["training"][4::9]
            with torch.no_grad():
                for method in ALL_METHODS:
                    for stream,(maps,projection) in enumerate(zip(mechanism_maps(images,method,model.features.backbone),model.features.legacy)):
                        flat=adaptive_avg_pool2d_anysize(maps,projection.pool_hw).flatten(1)
                        flat=(flat-flat.mean(1,keepdim=True))/(flat.std(1,keepdim=True)+1e-6)
                        norm=F.normalize(flat,dim=1);sim=norm @ norm.T
                        distance=(torch.arange(len(flat))[:,None]-torch.arange(len(flat))[None,:]).abs()
                        centred=flat-flat.mean(0,keepdim=True);gram=centred @ centred.T
                        energy=flat.reshape(len(flat),maps.shape[1],-1).square().sum(2)
                        shares=(energy/energy.sum(1,keepdim=True).clamp_min(1e-12)).mean(0)
                        rows.append(dict(ant=ant,method=method,stream="form" if stream==0 else "colour",
                                         neighbour_cosine=float(sim[distance==1].mean()),
                                         distant_cosine=float(sim[distance>=max(2,len(flat)//2)].mean()),
                                         effective_rank=float(gram.trace().square()/gram.square().sum().clamp_min(1e-12)),
                                         plane_energy_shares=shares.tolist()))
    (HERE/"feature-geometry.json").write_text(json.dumps(dict(rows=rows,script_sha256=file_hash(Path(__file__)),
       scope="Additional explanatory analysis; fixed pre-projection maps on centreline teaching images. No wiring seed replication because these maps are seed-independent."),indent=2)+"\n")
    print(f"Recorded {len(rows)} route/method/stream representations")

if __name__=="__main__":main()
