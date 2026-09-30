"""Common image contracts for matched visual representations and deep controls."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from apiaviz.src.modules import load_vision_backbone
from apiaviz.nav.retino_kc import RetinotopicKCProjection, adaptive_avg_pool2d_anysize
from apiaviz.nav.torch_route import preprocess_original_torch
from .circuit import CircuitConfig, SparseCircuit


REPRESENTATIONS = ("apiaviz", "gray", "colour", "opponent", "clahe")
MODES = ("kwta", "ann-threshold", "threshold", "feedback", "adaptive", "legacy")


@dataclass(frozen=True)
class EncoderConfig:
    representation: str = "apiaviz"
    mode: str = "adaptive"
    code_dim: int = 4000  # TOTAL across both streams
    pool_hw: tuple = (8, 64)
    fan_in: int = 10
    sparsity: float = 0.05
    seed: int = 7
    readout: str = "binary"
    perturbation: str = "none"
    ablate: str = "none"

    def __post_init__(self):
        if self.representation not in REPRESENTATIONS or self.mode not in MODES:
            raise ValueError("Unknown representation or encoder mode")
        if self.readout not in ("binary", "count", "early"):
            raise ValueError("Unknown spike readout")
        if self.perturbation not in ("none", "no-inhibition", "no-adaptation", "shuffle-times"):
            raise ValueError("Unknown circuit perturbation")
        if self.mode in ("kwta", "ann-threshold", "legacy") and self.perturbation != "none":
            raise ValueError("Circuit perturbations require a spiking mode")
        if self.code_dim < 2 or self.code_dim % 2:
            raise ValueError("code_dim is total KC count and must be positive and even")
        if not 0 < self.sparsity <= 1:
            raise ValueError("sparsity must be in (0, 1]")
        if len(self.pool_hw) != 2 or min(self.pool_hw) < 1:
            raise ValueError("pool_hw must contain two positive dimensions")


class VisualEncoder(nn.Module):
    """Input RGB or GB [B,C,H,W] in [0,1], output [B,total_KCs].

    Both streams occupy fixed three-channel slots so feature ablations and
    representations share exactly the same PN/KC connectivity for a given seed.
    """
    def __init__(self, config=None, circuit=None):
        super().__init__()
        self.config = config or EncoderConfig()
        self.circuit_config = circuit or CircuitConfig()
        c = self.config
        self.backbone = load_vision_backbone()
        self.backbone.ablate = c.ablate
        pn_dim = 6 * math.prod(c.pool_hw)  # three channels, each split by polarity
        self.streams = nn.ModuleList([SparseCircuit(pn_dim, c.code_dim // 2, c.fan_in,
                                                    c.seed + offset, self.circuit_config)
                                      for offset in (0, 16)])
        self.legacy = nn.ModuleList()
        if c.mode == "legacy":
            self.legacy.extend([RetinotopicKCProjection(channels, pool_hw=c.pool_hw,
                                code_dim=c.code_dim // 2, fan_in=c.fan_in,
                                sparsity=c.sparsity, seed=c.seed + offset)
                                for channels, offset in ((3, 0), (2, 16))])
        self.calibration = None
        self.eval()

    def maps(self, images):
        if images.ndim != 4 or images.shape[1] not in (2, 3):
            raise ValueError("Expected RGB or GB images [B,C,H,W] in [0,1]")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Images must be finite and in [0,1]")
        gb = images[:, -2:]
        gray = gb.mean(1, keepdim=True)
        zero = torch.zeros_like(gray)
        rep = self.config.representation
        if rep == "apiaviz":
            maps = self.backbone(gb * 2 - 1, return_maps=True)
            form, colour = maps["contrast_features"], maps["chromatic_feature"]
        else:
            if rep == "clahe":
                flat = preprocess_original_torch(gb.mean(1, keepdim=True) * 255, 1.,
                                                 self.config.pool_hw)
                gray = flat.reshape(-1, 1, *self.config.pool_hw)
                # CLAHE helper L2 normalizes; restore a bounded intensity range.
                gray = gray / gray.amax((2, 3), keepdim=True).clamp_min(1e-6)
                zero = torch.zeros_like(gray)
            form = torch.cat([gray, zero, zero], 1)
            if rep == "colour":
                colour = torch.cat([zero, gb[:, :1], gb[:, 1:]], 1)
            elif rep == "opponent":
                difference = gb[:, :1] - gb[:, 1:]
                colour = torch.cat([zero, difference.relu(), (-difference).relu()], 1)
            else:
                colour = torch.cat([zero, zero, zero], 1)
        return form, colour

    def pn_inputs(self, images):
        result = []
        for feature in self.maps(images):
            pooled = adaptive_avg_pool2d_anysize(feature, self.config.pool_hw).flatten(1)
            # Per-view gain control, no fitted dataset statistics. Zero stays zero.
            scale = pooled.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
            scaled = pooled / scale
            polarity = torch.cat([scaled.relu(), (-scaled).relu()], 1)
            result.append(polarity / (1 + polarity))
        return result

    @torch.no_grad()
    def diagnostics(self, images, record=False, state=None):
        c = self.config
        if c.mode == "legacy":
            form, chroma = self.maps(images)
            chunks = [F.normalize(self.legacy[0](form), dim=1),
                      F.normalize(self.legacy[1](chroma[:, 1:]), dim=1)]
            return {"codes": torch.cat(chunks, 1), "streams": []}
        inputs_by_stream = self.pn_inputs(images)
        if c.mode in ("kwta", "ann-threshold"):
            drives = [stream.project(inputs) for inputs, stream in zip(inputs_by_stream, self.streams)]
            if c.mode == "kwta":
                # One TOTAL activity budget, even when the colour stream is silent.
                drive = torch.cat(drives, 1)
                k = max(1, round(c.code_dim * c.sparsity))
                order = torch.argsort(drive, dim=1, descending=True, stable=True)[:, :k]
                codes = torch.zeros_like(drive).scatter(1, order, drive.gather(1, order))
            else:
                codes = torch.cat([torch.where(drive >= stream.threshold, drive, 0.)
                                   for drive, stream in zip(drives, self.streams)], 1)
            if c.readout != "count":
                codes = (codes > 0).float()
            return {"codes": codes, "streams": []}
        outputs = [stream.simulate(inputs, c.mode, c.readout, c.perturbation, record,
                                   None if state is None else state[i])
                   for i, (inputs, stream) in enumerate(zip(inputs_by_stream, self.streams))]
        return {"codes": torch.cat([o["codes"] for o in outputs], 1), "streams": outputs}

    def forward(self, images):
        return self.diagnostics(images)["codes"]

    @torch.no_grad()
    def calibrate(self, images, target=None, candidates=21):
        """Select a fixed threshold by closest mean activity, using development images only.

        Recurrent inhibition need not be monotonic in threshold; evaluate the
        complete fixed logarithmic grid rather than assuming binary search works.
        Silent streams are retained; achieved activity and target error are reported.
        """
        target = self.config.sparsity if target is None else float(target)
        if not 0 < target <= 1 or len(images) == 0:
            raise ValueError("Calibration needs nonempty development images and target in (0,1]")
        if self.config.mode in ("kwta", "legacy"):
            self.calibration = {"kind": "exact-topk", "target": target}
            return self.calibration
        inputs_by_stream = self.pn_inputs(images)
        if candidates < 3:
            raise ValueError("Calibration needs at least three threshold candidates")
        trials = []
        values = torch.logspace(-2, math.log10(4), candidates)
        for refinement in range(3):
            for value in values:
                achieved = []
                for inputs, stream in zip(inputs_by_stream, self.streams):
                    stream.threshold.fill_(float(value))
                    if self.config.mode == "ann-threshold":
                        active = stream.project(inputs) >= stream.threshold
                    else:
                        active = stream.simulate(inputs, self.config.mode,
                                                 perturbation=self.config.perturbation)["counts"] > 0
                    achieved.append(float(active.float().mean()))
                trials.append({"threshold": float(value), "achieved": sum(achieved) / 2,
                               "stream_activity": achieved})
            chosen = min(trials, key=lambda trial: (abs(trial["achieved"] - target), trial["threshold"]))
            ordered = sorted({t["threshold"] for t in trials})
            index = ordered.index(chosen["threshold"])
            lower, upper = ordered[max(0, index - 1)], ordered[min(len(ordered) - 1, index + 1)]
            values = torch.linspace(lower, upper, 9)[1:-1]
        for stream in self.streams:
            stream.threshold.fill_(chosen["threshold"])
        self.calibration = {"kind": "development-only", "target": target,
                            "achieved": chosen["achieved"],
                            "target_error": abs(chosen["achieved"] - target),
                            "streams": [{"threshold": chosen["threshold"], "achieved": activity}
                                        for activity in chosen["stream_activity"]], "trials": trials}
        return self.calibration

    def metadata(self):
        return {"encoder": asdict(self.config), "circuit": asdict(self.circuit_config),
                "thresholds": [float(s.threshold) for s in self.streams],
                "calibration": self.calibration}


class DeepEncoder(nn.Module):
    """Frozen official checkpoints. Spatial pooling preserves the full field of view.

    Checkpoints are local and explicit: constructing an encoder never downloads
    weights or fetches executable code. DINO imports a local official checkout.
    """
    def __init__(self, name, checkpoint, pooling="spatial", sensory="gb", dino_repo=None):
        super().__init__()
        if pooling not in ("global", "spatial") or sensory not in ("gb", "rgb"):
            raise ValueError("Invalid deep pooling or sensory condition")
        self.name, self.pooling, self.sensory = name, pooling, sensory
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Supply a local official checkpoint: {checkpoint}")
        if name == "mobilenet":
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.model = model.features
            self.patch_size = 32
        elif name == "dinov2":
            if dino_repo is None or not (Path(dino_repo) / "hubconf.py").is_file():
                raise ValueError("DINOv2 requires --dino-repo pointing to the official local checkout")
            self.model = torch.hub.load(str(dino_repo), "dinov2_vits14", source="local", pretrained=False)
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.patch_size = 14
        else:
            raise ValueError(name)
        self.model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images):
        if self.sensory == "gb":
            gb = images[:, -2:]
            images = torch.cat([gb.mean(1, keepdim=True), gb], 1)
        elif images.shape[1] != 3:
            raise ValueError("Native RGB comparison requires RGB source images")
        h, w = images.shape[-2:]
        # Resize the short side; pad to patch multiples without cropping the panorama.
        shape = (round(h * 224 / min(h, w)), round(w * 224 / min(h, w)))
        x = F.interpolate(images, size=shape, mode="bilinear", align_corners=False)
        ph, pw = (-shape[0]) % self.patch_size, (-shape[1]) % self.patch_size
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        x = (x - self.mean) / self.std
        if self.name == "dinov2":
            features = self.model.forward_features(x)["x_norm_patchtokens"]
            features = features.transpose(1, 2).reshape(len(x), -1,
                         x.shape[-2] // 14, x.shape[-1] // 14)
        else:
            features = self.model(x)
        shape = (1, 1) if self.pooling == "global" else (2, 8)
        features = adaptive_avg_pool2d_anysize(features, shape).flatten(1)
        # Nonnegative contract for the shared memories; preserve both polarities.
        return F.normalize(torch.cat([features.relu(), (-features).relu()], 1), dim=1)
