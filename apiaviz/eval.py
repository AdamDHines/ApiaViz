"""Navigation evaluation: mushroom-body route following with the MBON population readout.

On the colour-landmark ant world, memory is stored on the clean world; the agent then free-navigates
(familiarity-based heading scan) while every glimpse is optionally hit by a chroma-preserving
luminance perturbation (models lighting/weather: the opponent-colour landmarks survive; grayscale
luminance does not). Three readouts are compared:

  - CLAHE                     the Ardin/Webb grayscale template (baseline)
  - MBON single (S=1)         one anti-Hebbian novelty neuron over the colour KC code
  - MBON population (S)        ours: a small population, one MBON per route segment

Reports reached-nest fraction and off-route error_rate (fraction of steps corrected back to the route).
The headline: under corruption the single MBON drifts, but the population homes as well as the template
bank and beats CLAHE -- the biologically faithful readout, no episodic per-view memory.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cv2
import torch.nn.functional as F
from PIL import Image

from apiaviz.mbant.config import ImageConfig, NavigationConfig
from apiaviz.mbant.io_utils import load_ant_data, load_world_data, prepare_route

from apiaviz.nav.retino_kc import MBONPopulation, RetinotopicKCEncoder, RetinotopicKCProjection, AntiHebbianMBON, RetinotopicKCProjection
from apiaviz.nav.torch_route import (
    TorchWorldRenderer, CosineRouteMemory, navigate_torch, preprocess_apiaviz_torch, preprocess_original_torch, select_device, render_route_corridor, corridor_offsets, nav_result_summary
)
from apiaviz.src.metrics import render_table
from apiaviz.src.modules import load_vision_backbone


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def free_navigate(img_pos_np, heading_np, scorer, nav_config, max_steps=None):
    """Open-loop route following: scan headings, step toward the most-familiar one, and repeat -- with
    NO snap-back to the route (unlike ``navigate_torch``). The honest test of the view code alone: the
    agent flies on its own familiarity gradient and either homes or drifts. Returns reached_nest and the
    final route progress (nearest route index reached / route length)."""
    device = scorer.renderer.device
    img_pos = torch.as_tensor(img_pos_np, dtype=torch.float32, device=device)
    heading = torch.as_tensor(heading_np, dtype=torch.float32, device=device)
    nest = img_pos[-1]
    pos = img_pos[0].clone()
    center = heading[0].clone()
    n_steps = int(max_steps) if max_steps else int(math.ceil((len(img_pos_np) - 1) * 10))
    scan_offsets = (nav_config.scan_range / 2.0
                    - torch.arange(nav_config.num_scan_img, dtype=torch.float32, device=device) * nav_config.scan_step)
    reached = False
    for _ in range(n_steps):
        scan_headings = center + scan_offsets
        en = scorer.score(pos, scan_headings, device)["en"].detach()
        center = scan_headings[int(torch.argmin(en).item())]
        rad = torch.deg2rad(center)
        pos = pos + torch.stack([torch.cos(rad), torch.sin(rad)]) * nav_config.step_size
        if torch.linalg.norm(nest - pos) <= nav_config.dis_threshold:
            reached = True
            break
    nearest = int(torch.linalg.norm(img_pos - pos, dim=1).argmin().item())
    return {"reached_nest": reached, "final_route_progress": nearest / max(len(img_pos_np) - 1, 1)}


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


def build_landmark_color(triangle_grey: torch.Tensor, fraction: float, chroma: float, seed: int, device: torch.device) -> torch.Tensor:
    """[n_tri,3] RGB: grey baseline for all, an isoluminant G-B tint on a `fraction` of triangles.

    Landmark i: RGB = (L, L+chroma*s, L-chroma*s), s in {+1,-1}. mean(RGB)=L (luminance held at the grey
    baseline -> invisible to a grayscale system) but G-B = 2*chroma*s (vivid to the opponent channels).
    """
    n = triangle_grey.numel()
    gen = torch.Generator().manual_seed(int(seed))
    L = triangle_grey
    tc = L[:, None].repeat(1, 3)
    is_lm = torch.rand(n, generator=gen) < fraction
    sign = torch.where(torch.rand(n, generator=gen) < 0.5, 1.0, -1.0)
    # put onto torch.device
    L = L.to(device)
    is_lm = is_lm.to(device)
    sign = sign.to(device)
    tc[:, 1] = torch.where(is_lm, (L + chroma * sign).clamp(0, 255), L)  # G
    tc[:, 2] = torch.where(is_lm, (L - chroma * sign).clamp(0, 255), L)  # B
    return tc


def luminance_corrupt(raw: torch.Tensor, severity: float, seed: int, device: torch.device) -> torch.Tensor:
    """Add a multi-scale luminance shift to all channels equally (G-B preserved)."""
    if severity <= 0:
        return raw
    n, _, h, w = raw.shape
    gen = torch.Generator().manual_seed(int(seed))
    field = torch.zeros(n, 1, h, w)
    for scale in (2, 4, 8, 16):
        small = torch.randn(n, 1, max(1, h // scale), max(1, w // scale), generator=gen)
        field = field + F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(n, 1, 1, 1)
    # place all on device
    raw = raw.to(device)
    field = field.to(device)
    return (raw + severity * 90.0 * field).clamp(0.0, 255.0)


class _PerturbScorer:
    """Renders a heading scan and (optionally) applies the luminance perturbation before encoding."""

    def __init__(self, renderer, memory, ic, severity, seed0):
        self.renderer, self.memory, self.ic = renderer, memory, ic
        self.severity, self.seed0, self._calls = float(severity), int(seed0), 0

    def _raw(self, position, headings, device):
        positions = position.view(1, 2).expand(headings.numel(), -1)
        raw = self.renderer.render_batch(positions, headings, self.ic.eye_height)
        if self.severity > 0:
            self._calls += 1
            raw = luminance_corrupt(raw, self.severity, self.seed0 + self._calls, device)
        return raw


class _KCScorer(_PerturbScorer):
    def __init__(self, renderer, backbone, code_fn, memory, ic, severity, seed0):
        super().__init__(renderer, memory, ic, severity, seed0)
        self.backbone, self.code_fn = backbone, code_fn

    @torch.no_grad()
    def score(self, position, headings, device):
        raw = self._raw(position, headings, device)
        maps = self.backbone(preprocess_apiaviz_torch(raw).to(self.renderer.device), return_maps=True)
        code = self.code_fn(maps)
        return {"en": self.memory(code), "codes": code}


class _CLAHEScorer(_PerturbScorer):
    @torch.no_grad()
    def score(self, position, headings, device):
        raw = self._raw(position, headings, device)
        code = preprocess_original_torch(raw, 1.0, self.ic.resize_shape).to(self.renderer.device)
        return {"en": self.memory(code), "codes": code}


class NavEval:
    """Evaluate familiarity-based route navigation with the MBON population readout vs CLAHE."""

    def __init__(self, args, logger, outdir):
        for key in vars(args):
            setattr(self, key, getattr(args, key))
        self.logger = logger
        self.outdir = Path(outdir)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        antData = "apiaviz/mbant/data/antview/AntData.mat"
        worldData = "apiaviz/mbant/data/antview/world5000_gray.mat"

        # check for data, if not exist then download from HF source
        if not os.path.exists(antData) or not os.path.exists(worldData):
            hfURL = "https://huggingface.co/datasets/AdamHines/MB_Ant"
            # clone into apiaviz/mbant directory
            subprocess.run(
                ["git", "clone", hfURL, "./apiaviz/mbant/data/"],
                check=True
            )

        self.ants = load_ant_data(antData)
        world = load_world_data(worldData)
        self.ic = ImageConfig(resolution=4.0, hfov=296.0)

        base = TorchWorldRenderer(world, device=self.device, hfov=self.ic.hfov,
                                  resolution=self.ic.resolution, chunk_size=512, color=True)
        landmark_tc = build_landmark_color(base.triangle_grey, self.landmark_fraction, self.chroma, seed=99, device=self.device)
        self.renderer = TorchWorldRenderer(world, device=self.device, hfov=self.ic.hfov,
                                           resolution=self.ic.resolution, chunk_size=512,
                                           color=True, triangle_color=landmark_tc)
        backbone = load_vision_backbone(self.device, logger=self.logger)
        encoder = RetinotopicKCEncoder(backbone=backbone, tap="contrast_features",
                                       code_dim=self.code_dim, seed=self.seed).to(self.device)
        self.backbone, self.ach_proj = encoder.backbone, encoder.projection
        self.opp_proj = RetinotopicKCProjection(in_channels=2, code_dim=self.code_dim, seed=self.seed + 16).to(self.device)
        self.mem_kwargs = dict(mode="max_cosine", topk=5, softmax_temperature=0.05)
        self.freenav = bool(getattr(self, "freenav", False))
        self.viewpoints = max(1, int(getattr(self, "viewpoints", 1)))
        self.corridor_width = float(getattr(self, "corridor_width", 0.2))
        loop = "open-loop free navigation" if self.freenav else "corrected route-following"
        # The route-corridor memory (multiple laterally-shifted viewpoints per route index) is what
        # rescues open-loop navigation, so it is only engaged in the freenav paradigm.
        self.memory_viewpoints = self.viewpoints if self.freenav else 1
        memory = (f"route corridor ({self.memory_viewpoints} viewpoints, +-{self.corridor_width:g} m)"
                  if self.memory_viewpoints > 1 else "single centreline viewpoint")
        self.logger.info(f"Device: {self.device}  |  loop: {loop}  |  memory: {memory}")
        self.logger.info("")

    def _combined_code(self, maps):
        """Achromatic contrast KC code (+) opponent-colour KC code -- the navigation code the MBON reads."""
        return torch.cat([_l2(self.ach_proj(maps["contrast_features"])),
                          _l2(self.opp_proj(maps["chromatic_feature"][:, 1:3]))], dim=1)

    def run(self):
        nav_config = NavigationConfig(step_size=0.1, scan_range=120.0, scan_step=10.0, dis_threshold=0.2,
                                      eye_height=self.ic.eye_height, resolution=self.ic.resolution, hfov=self.ic.hfov)
        pop_name = f"MBON population (S={self.segments})"
        methods = ["CLAHE", pop_name]
        conditions = [("clean", 0.0), ("corruption", float(self.severity))]
        agg = {(c, m): [] for c in ("clean", "corruption") for m in methods}

        for route in self.routes:
            rd = self.ants[f"Ant{self.ant}"]["routes"][f"Route{route}"]
            img_pos, heading, _ = prepare_route(rd, img_separation=self.ic.img_separation)
            offsets = corridor_offsets(self.memory_viewpoints, self.corridor_width)
            raw_route = render_route_corridor(self.renderer, img_pos, heading, offsets, self.ic.eye_height)
            with torch.no_grad():
                maps = self.backbone(preprocess_apiaviz_torch(raw_route).to(self.device), return_maps=True)
                combined_route = self._combined_code(maps).detach()
            clahe_route = preprocess_original_torch(raw_route, 1.0, self.ic.resize_shape).detach()

            memories = {
                "CLAHE": CosineRouteMemory(clahe_route, **self.mem_kwargs).to(self.device),
                pop_name: MBONPopulation(combined_route, n_mbons=self.segments).to(self.device),
            }

            for cond, severity in conditions:
                for method in methods:
                    if method == "CLAHE":
                        scorer = _CLAHEScorer(self.renderer, memories[method], self.ic, severity, seed0=5000)
                    else:
                        scorer = _KCScorer(self.renderer, self.backbone, self._combined_code,
                                           memories[method], self.ic, severity, seed0=5000)
                    if self.freenav:
                        summary = free_navigate(img_pos, heading, scorer, nav_config, max_steps=self.max_steps)
                        metric = float(summary["final_route_progress"])
                    else:
                        nav = navigate_torch(img_pos, heading, scorer, nav_config)
                        nav.setdefault("trained_route", img_pos)
                        summary = nav_result_summary(nav, nav_config)
                        metric = float(summary.get("error_rate"))
                    agg[(cond, method)].append((bool(summary.get("reached_nest")), metric))
                    mname = "progress" if self.freenav else "error_rate"
                    self.logger.info(f"Route{route:<2d} {cond:<11s} {method:<22s} "
                                     f"reached={summary.get('reached_nest')} {mname}={metric:.3f}")

        self._report(methods, agg)

    def _report(self, methods, agg):
        n = len(self.routes)
        metric_name = "progress" if self.freenav else "error_rate"
        loop_kind = "open-loop free navigation" if self.freenav else "corrected route-following"
        for cond in ("clean", "corruption"):
            rows = []
            for method in methods:
                results = agg[(cond, method)]
                reached = np.mean([1.0 if r else 0.0 for r, _ in results]) if results else float("nan")
                metric = np.mean([m for _, m in results]) if results else float("nan")
                note = "ours" if method.startswith("MBON population") else ("baseline" if method == "CLAHE" else "")
                rows.append([method, f"{reached:.2f}", f"{metric:.3f}", note])
            table = render_table(
                ["readout", "reached", metric_name, "note"], rows,
                title=f"\nNavigation under {cond}  ({loop_kind}; Ant{self.ant}, {n} route{'s' if n != 1 else ''}; "
                      f"reached = fraction of nests homed)",
            )
            self.logger.info("")
            for line in table.splitlines():
                self.logger.info(line)
        self.logger.info("")
        self.logger.info("Under corruption the single MBON drifts; the MBON population homes as often as the "
                         "colour code allows and beats the colour-blind CLAHE baseline.")

class EvalVision:
    """Evaluate the frozen VisionBackbone on a downstream identification task."""

    def __init__(self, args, logger, outdir):
        for key in vars(args):
            setattr(self, key, getattr(args, key))
        self.logger = logger
        self.outdir = Path(outdir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(self.models_dir, f"{self.vision_model}.pth")
        self.backbone = load_vision_backbone(model_path, self.device,
                                             untrained=getattr(self, "untrained", False), logger=self.logger)
        self.logger.info(f"Device: {self.device}")
        self.logger.info("")

    def eval(self) -> None:
        if self.eval_dataset in ("flowers", "17flowers"):
            self._eval_flowers()
        else:
            raise ValueError(
                f"eval_dataset '{self.eval_dataset}' is not supported here; try 'flowers' "
                "(navigation is a separate task: `pixi run nav`)."
            )

    # ------------------------------------------------------------------ flower identification
    def _make_views(self, paths, offset=(14, 14), canvas=64, obj=36) -> torch.Tensor:
        """Paste each flower's (G,B) channels on a neutral canvas -> [N,2,canvas,canvas] in [-1,1]."""
        oy, ox = offset
        views = []
        for path in paths:
            arr = np.asarray(Image.open(path).convert("RGB").resize((obj, obj)), dtype=np.float32) / 255.0
            canvas_arr = np.full((canvas, canvas, 2), 0.5, dtype=np.float32)
            canvas_arr[oy:oy + obj, ox:ox + obj, :] = arr[:, :, 1:3]  # (G, B)
            views.append((torch.from_numpy(canvas_arr).permute(2, 0, 1) - 0.5) / 0.5)
        return torch.stack(views).to(self.device)

    def _jitter(self, views, seed, max_rot=15.0, max_trans=0.10, max_scale=0.1) -> torch.Tensor:
        """One simulated fixation: a small random affine (rotate/translate/scale)."""
        n = views.size(0)
        gen = torch.Generator().manual_seed(int(seed))
        ang = (torch.rand(n, generator=gen) * 2 - 1) * (max_rot * np.pi / 180.0)
        scl = 1.0 + (torch.rand(n, generator=gen) * 2 - 1) * max_scale
        tx = (torch.rand(n, generator=gen) * 2 - 1) * max_trans
        ty = (torch.rand(n, generator=gen) * 2 - 1) * max_trans
        cos, sin = torch.cos(ang), torch.sin(ang)
        theta = torch.zeros(n, 2, 3)
        theta[:, 0, 0] = cos / scl; theta[:, 0, 1] = -sin / scl; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin / scl; theta[:, 1, 1] = cos / scl; theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, list(views.size()), align_corners=False)
        return F.grid_sample(views.cpu(), grid, align_corners=False, padding_mode="border").to(views.device)

    def _clahe_map(self, views, size=24) -> torch.Tensor:
        """Ardin/Webb CLAHE luminance map (grayscale -> invert -> CLAHE clip 2.0/8x8) -> [N,1,size,size].

        Returned as a feature map (not a flat template) so it can be lifted through the SAME kind of
        KC projection as the chromatic map. Both representations then read out through the MBON, so the
        comparison isolates the representation (opponent colour vs grayscale luminance), not the readout.
        """
        gray = ((views + 1.0) / 2.0).mean(dim=1).cpu().numpy()
        maps = []
        for img in gray:
            resized = cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
            inverted = np.clip(1.0 - resized, 0.0, 1.0)
            eq = _CLAHE.apply(np.round(inverted * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
            maps.append(eq)
        arr = np.stack(maps, axis=0)[:, None, :, :]
        return torch.from_numpy(arr).to(self.device)

    @torch.no_grad()
    def _encode(self, proj_chroma, proj_clahe, views, batch=128) -> dict:
        """One fixation -> {representation: [N,D] KC code}. Both reps: feature map -> KC projection."""
        chunks = {"CLAHE": [], "chromatic_kc": []}
        for i in range(0, views.size(0), batch):
            v = views[i:i + batch]
            maps = self.backbone(v, return_maps=True)
            chunks["CLAHE"].append(proj_clahe(self._clahe_map(v)).cpu().numpy())
            chunks["chromatic_kc"].append(proj_chroma(maps["chromatic_feature"]).cpu().numpy())
        return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}

    @staticmethod
    def _mbon_accuracy(feats, labels, train_idx, test_idx, depression=0.5) -> float:
        """Per-class MBON population: one anti-Hebbian MBON stores each class's KC codes; a test
        view is assigned to the most-familiar class (lowest novelty). The nav readout, per-class."""
        codes = torch.from_numpy(feats).float()
        classes = np.unique(labels[train_idx])
        novelty = np.zeros((len(test_idx), len(classes)), dtype=np.float64)
        for j, c in enumerate(classes):
            mbon = AntiHebbianMBON(code_dim=feats.shape[1], depression=depression)
            mbon.store(codes[train_idx][labels[train_idx] == c])
            novelty[:, j] = mbon(codes[test_idx]).numpy()
        preds = classes[novelty.argmin(axis=1)]
        return float((preds == labels[test_idx]).mean())

    def _eval_flowers(self) -> None:
        reps = ["CLAHE", "chromatic_kc"]
        ks = list(self.ks)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        root = Path(self.dataset_dir) / "17flowers"
        class_dirs = sorted([d for d in root.iterdir() if d.is_dir() and any(d.glob("*.jpg"))])
        if not class_dirs:
            raise FileNotFoundError(f"No flower classes under {root} (run `pixi run get_evaldata`?)")
        paths, labels = [], []
        for ci, d in enumerate(class_dirs):
            imgs = sorted(d.glob("*.jpg"))
            chosen = rng.choice(imgs, size=min(self.per_class, len(imgs)), replace=False)
            paths += [Path(p) for p in chosen]
            labels += [ci] * len(chosen)
        labels = np.asarray(labels)
        chance = 1.0 / len(class_dirs)
        self.logger.info(f"17flowers: {len(paths)} images, {len(class_dirs)} classes (chance {chance:.3f})")
        self.logger.info(f"Scan-accumulating chromatic code over up to K={max(ks)} fixations...")

        proj_chroma = RetinotopicKCProjection(in_channels=3, code_dim=self.code_dim, seed=23).to(self.device)
        proj_clahe = RetinotopicKCProjection(in_channels=1, code_dim=self.code_dim, seed=29).to(self.device)
        base = self._make_views(paths)
        running, snapshots = None, {}
        for fix in range(max(ks)):
            views = base if fix == 0 else self._jitter(base, seed=1000 + fix)
            feats = self._encode(proj_chroma, proj_clahe, views)
            running = ({r: feats[r].astype(np.float64) for r in reps} if running is None
                       else {r: running[r] + feats[r] for r in reps})
            if fix + 1 in ks:
                snapshots[fix + 1] = {r: (running[r] / (fix + 1)).astype(np.float32) for r in reps}

        idx = np.arange(len(paths)); rng.shuffle(idx)
        split = int(0.7 * len(idx))
        train_idx, test_idx = idx[:split], idx[split:]

        rows = []
        for r in reps:
            accs = [self._mbon_accuracy(snapshots[k][r], labels, train_idx, test_idx) for k in ks]
            note = "ours" if r == "chromatic_kc" else "baseline"
            rows.append([r, "MBON", *[f"{a:.3f}" for a in accs], f"{accs[-1] / chance:.1f}x", note])

        headers = ["representation", "readout", *[f"K={k}" for k in ks], "x chance", "note"]
        table = render_table(
            headers, rows,
            title=f"\nFlower identification accuracy vs scan length K  (chance {chance:.3f})",
        )
        self.logger.info("")
        for line in table.splitlines():
            self.logger.info(line)
        self.logger.info("")
        self.logger.info("Same per-class MBON readout on both representations, so only the code differs: "
                         "the opponent-colour KC code identifies flowers; the grayscale CLAHE KC code cannot.")
