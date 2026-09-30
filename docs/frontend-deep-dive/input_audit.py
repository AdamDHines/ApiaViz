"""Measure filter gains, channel energy, and context dependence of colour."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from apiaviz.research.frontend_refinements import RefinementEncoder, refinement_maps

HERE = Path(__file__).parent
METHODS = tuple(json.loads((HERE / "protocol.json").read_text())["methods"])


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = RefinementEncoder(code_dim=128)
    paths = json.loads((ROOT / "docs/mechanism-study/runs.json").read_text())["runs"]
    rows = []
    context_rows = []
    example = None
    for name in paths:
        run = ROOT / name
        for ant in json.loads((run / "manifest.json").read_text())["settings"]["ants"]:
            images = torch.load(run / f"ant-{ant}-images.pt", weights_only=True, map_location="cpu")["training"][4::9]
            if ant == 4:
                example = images[len(images)//2:len(images)//2+1]
            for method in METHODS:
                model.method = method
                for stream, pooled in zip(("form", "colour"), model.pooled_features(images)):
                    flat = pooled.flatten(1)
                    flat = (flat-flat.mean(1,keepdim=True))/(flat.std(1,keepdim=True)+1e-6)
                    planes = flat.reshape(len(flat), pooled.shape[1], -1)
                    energy = planes.square().sum(2)
                    similarity = F.normalize(flat,dim=1) @ F.normalize(flat,dim=1).T
                    separation = (torch.arange(len(flat))[:,None]-torch.arange(len(flat))[None,:]).abs()
                    rows.append(dict(ant=ant, method=method, stream=stream,
                        energy_shares=(energy/energy.sum(1,keepdim=True).clamp_min(1e-12)).mean(0).tolist(),
                        spatially_constant_energy_fraction=float((planes.mean(2).square().sum(1)*planes.shape[2]/energy.sum(1).clamp_min(1e-12)).mean()),
                        neighbour_cosine=float(similarity[separation==1].mean()),
                        distant_cosine=float(similarity[separation>=len(flat)//2].mean())))
            # Shared spatial luminance perturbation, with headroom to avoid
            # clipping and therefore preserve the actual G-B difference.
            bounded = .2 + .6 * images
            xx = torch.arange(images.shape[-1]) * (2*np.pi/images.shape[-1])
            yy = torch.linspace(-1,1,images.shape[-2])
            perturbation = .1 * yy[:,None] * xx.sin()[None,:]
            shifted = bounded + perturbation[None,None]
            for method in ("apiaviz", "sobel_colour", "linear_colour"):
                a,b = [refinement_maps(x,method,model.features.backbone)[1].flatten(1) for x in (bounded,shifted)]
                an,bn = [F.normalize(x-x.mean(1,keepdim=True),dim=1) for x in (a,b)]
                context_rows.append(dict(ant=ant,method=method,
                    colour_relative_l2_change=float(((b-a).norm(dim=1)/a.norm(dim=1).clamp_min(1e-6)).mean()),
                    centred_colour_cosine=float((an*bn).sum(1).mean())))
    weights = model.features.backbone.contrast_filter.filters.weight.detach()[:,0]
    coord = torch.arange(-2., 3.)
    r2 = coord[:,None].square()+coord[None,:].square()
    surround = torch.exp(-r2/(2*2.5**2));surround /= surround.sum()
    kernels = dict(form_kernel_sums=weights.sum((1,2)).tolist(), form_kernel_l2=weights.square().sum((1,2)).sqrt().tolist(),
                   nominal_surround_sigma_px=2.5, implemented_surround_sigma_x_px=float((surround*coord[None,:].square()).sum().sqrt()),
                   surround_mass_within_support=float(torch.exp(-r2/(2*2.5**2)).sum()/(2*np.pi*2.5**2)))
    # Central 3x3 patch has fixed G=.7/B=.3; only its achromatic surround changes.
    levels = torch.linspace(.1,.9,41)
    stimulus = levels[:,None,None,None].expand(-1,2,18,74).clone()
    stimulus[:,0,7:10,35:38] = .7;stimulus[:,1,7:10,35:38] = .3
    colour = {}
    for method in ("apiaviz", "linear_colour"):
        maps = refinement_maps(stimulus,method,model.features.backbone)[1]
        colour[method] = maps[:,0,8,36].tolist()
    result = dict(kernels=kernels, rows=rows, common_luminance_pattern=context_rows, colour_context=dict(surround_levels=levels.tolist(), centre_green=.7, centre_blue=.3, responses=colour),
                  scope="Descriptive input audit, no fitted statistics. Geometry uses eight routes without seed duplication. Fixed-colour patch varies only its achromatic surround.")
    (HERE / "input-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    out = HERE / "figures";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    selected = ("apiaviz","unit_dc","balanced_form","sobel_colour","oriented_form")
    fig,(ax,bx) = plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={"width_ratios":[1.3,1]})
    labels = ("ApiaViz","Unit-gain low-pass","Balanced form","Sobel + colour","Oriented Apia form")
    shares = np.array([np.mean([r["energy_shares"] for r in rows if r["method"]==m and r["stream"]=="form"],axis=0) for m in selected])
    for j,c in enumerate(("#087f8c","#ce7b32","#8256a3")):
        ax.barh(range(len(selected)),shares[:,j],left=shares[:,:j].sum(1),color=c,label=f"Plane {j+1}")
    ax.set(yticks=range(len(selected)),yticklabels=labels,xlabel="Fraction of standardised form-feature energy",xlim=(0,1),title="A  Most ApiaViz form energy is in its third plane")
    ax.invert_yaxis();ax.legend(loc="upper center",bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8)
    for m,label in (("apiaviz","Current colour path"),("linear_colour","Linear colour difference")):
        y=np.array(colour[m]);bx.plot(levels,y/y[20],label=label)
    bx.set(xlabel="Achromatic surround intensity",ylabel="Relative opponent response",title="B  The same colour changes with its surround")
    bx.legend(fontsize=8,loc="lower center")
    fig.tight_layout(w_pad=2)
    fig.savefig(out/"input-audit.pdf",bbox_inches="tight");fig.savefig(out/"input-audit.png",dpi=180,bbox_inches="tight");plt.close(fig)
    # Fixed illustrative case, chosen by ant/index before examining candidate outcomes.
    fig,axes=plt.subplots(3,3,figsize=(10,4.8))
    for row,method in enumerate(("apiaviz","sobel_colour","balanced_form")):
        model.method=method;form=model.pooled_features(example)[0][0].numpy()
        for j,ax in enumerate(axes[row]):
            limit=max(float(abs(form[j]).max()),1e-6)
            ax.imshow(form[j],cmap="RdBu_r",vmin=-limit,vmax=limit,aspect="auto")
            ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel({"apiaviz":"ApiaViz","sobel_colour":"Sobel","balanced_form":"Balanced ApiaViz"}[method])
            ax.set_title(("ON","OFF","Smoothed adapted contrast")[j] if method!="sobel_colour" else ("Horizontal derivative","Vertical derivative","Gradient magnitude")[j],fontsize=9)
    fig.suptitle("Same teaching image, different form maps (each panel scaled separately)",fontsize=11)
    fig.tight_layout()
    fig.savefig(out/"feature-maps.pdf",bbox_inches="tight");fig.savefig(out/"feature-maps.png",dpi=180,bbox_inches="tight")
    print(json.dumps(dict(kernels=kernels,colour_response_ranges={m:[min(v),max(v)] for m,v in colour.items()},mean_energy=shares.tolist()),indent=2))


if __name__ == "__main__":main()
