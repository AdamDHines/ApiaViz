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

from apiaviz.mbant.config import ImageConfig, NavigationConfig
from apiaviz.mbant.io_utils import load_ant_data, load_world_data, prepare_route

from apiaviz.nav.retino_kc import MBONPopulation, RetinotopicKCEncoder, RetinotopicKCProjection
from apiaviz.nav.torch_route import (
    TorchWorldRenderer, CosineRouteMemory, navigate_torch, preprocess_apiaviz_torch, preprocess_original_torch, select_device, render_route_corridor, corridor_offsets, nav_result_summary
)
from apiaviz.src.metrics import render_table
from apiaviz.src.modules import load_vision_backbone
from apiaviz.src.flowers import (
    ScanEncoder, build_views, clahe_map, evaluate_cv, stratified_folds, paired_delta,
    choose_rewarded, evaluate_gonogo_cv, corrupting_scan, affine_scan,
)


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
        backbone.ablate = str(getattr(self, "ablate", "none"))
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
        self.logger.info(f"Device: {self.device}  |  loop: {loop}  |  memory: {memory}  |  "
                         f"backbone ablation: {self.backbone.ablate}")
        self.logger.info("")

    def _combined_code(self, maps):
        """Achromatic contrast KC code (+) opponent-colour KC code -- the navigation code the MBON reads."""
        return torch.cat([_l2(self.ach_proj(maps["contrast_features"])),
                          _l2(self.opp_proj(maps["chromatic_feature"][:, 1:3]))], dim=1)

    def _resolve_jobs(self):
        """Resolve (ant, route) pairs to actually run, skipping anything absent from the data.

        Ants and routes are non-uniform (e.g. Ant6 only has 2 inward routes), so a requested
        index may not exist. Rather than crash on a KeyError, we filter to what is available and
        log a warning for each skipped selection.

        Selection rules:
            --ant  <= 0  -> every ant present in the data
            --routes empty / any value <= 0 -> every available route for that ant
        """
        available_ants = sorted(int(k[3:]) for k in self.ants)  # e.g. [1, 2, ..., 15]

        if int(self.ant) <= 0:
            ant_indices = available_ants
        elif int(self.ant) in available_ants:
            ant_indices = [int(self.ant)]
        else:
            self.logger.info(f"[skip] Ant{self.ant} is not in the data "
                             f"(available: {available_ants[0]}-{available_ants[-1]}); nothing to run.")
            ant_indices = []

        requested = [int(r) for r in self.routes]
        want_all_routes = not requested or any(r <= 0 for r in requested)

        jobs = []
        for ant_idx in ant_indices:
            available = self.ants[f"Ant{ant_idx}"]["available_routes"]
            routes = available if want_all_routes else requested
            for route in routes:
                if route in available:
                    jobs.append((ant_idx, route))
                else:
                    self.logger.info(f"[skip] Ant{ant_idx} has no Route{route} "
                                     f"(available routes: {available}).")
        return jobs

    def run(self):
        nav_config = NavigationConfig(step_size=0.1, scan_range=120.0, scan_step=10.0, dis_threshold=0.2,
                                      eye_height=self.ic.eye_height, resolution=self.ic.resolution, hfov=self.ic.hfov)
        pop_name = f"MBON population (S={self.segments})"
        methods = ["CLAHE", pop_name]
        # Clean is always evaluated; each positive severity in --severities adds a corruption
        # condition. An empty list means clean-only (no corruption).
        severities = [float(s) for s in getattr(self, "severities", []) if float(s) > 0]
        conditions = [("clean", 0.0)] + [(f"corruption(sev={s:g})", s) for s in severities]
        self._cond_labels = [c for c, _ in conditions]
        agg = {(c, m): [] for c in self._cond_labels for m in methods}

        jobs = self._resolve_jobs()
        if not jobs:
            self.logger.info("No valid (ant, route) selections to navigate; check --ant / --routes.")
            return
        self._jobs = jobs

        for ant, route in jobs:
            rd = self.ants[f"Ant{ant}"]["routes"][f"Route{route}"]
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
                    self.logger.info(f"Ant{ant:<2d} Route{route:<2d} {cond:<11s} {method:<22s} "
                                     f"reached={summary.get('reached_nest')} {mname}={metric:.3f}")

        self._report(methods, agg)

    def _report(self, methods, agg):
        jobs = getattr(self, "_jobs", [])
        n = len(jobs)
        ants_run = sorted({a for a, _ in jobs})
        if len(ants_run) == 1:
            ant_label = f"Ant{ants_run[0]}"
        elif ants_run:
            ant_label = f"{len(ants_run)} ants (Ant{ants_run[0]}-Ant{ants_run[-1]})"
        else:
            ant_label = f"Ant{self.ant}"
        metric_name = "progress" if self.freenav else "error_rate"
        loop_kind = "open-loop free navigation" if self.freenav else "corrected route-following"
        for cond in getattr(self, "_cond_labels", ["clean"]):
            rows = []
            for method in methods:
                results = agg[(cond, method)]
                reached = np.mean([1.0 if r else 0.0 for r, _ in results]) if results else float("nan")
                metric = np.mean([m for _, m in results]) if results else float("nan")
                note = "ours" if method.startswith("MBON population") else ("baseline" if method == "CLAHE" else "")
                rows.append([method, f"{reached:.2f}", f"{metric:.3f}", note])
            table = render_table(
                ["readout", "reached", metric_name, "note"], rows,
                title=f"\nNavigation under {cond}  ({loop_kind}; {ant_label}, {n} route{'s' if n != 1 else ''}; "
                      f"reached = fraction of nests homed)",
            )
            self.logger.info("")
            for line in table.splitlines():
                self.logger.info(line)
        if len(getattr(self, "_cond_labels", ["clean"])) > 1:
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

        self.backbone = load_vision_backbone(self.device, logger=self.logger)
        self.backbone.ablate = str(getattr(self, "ablate", "none"))
        self.logger.info(f"Device: {self.device}  |  backbone ablation: {self.backbone.ablate}")
        self.logger.info("")

    def eval(self) -> None:
        if self.eval_dataset in ("flowers", "17flowers"):
            if getattr(self, "task", "reward_gonogo") == "classify":
                self._eval_flowers()          # 17-way anti-Hebbian novelty (reference)
            else:
                self._eval_flowers_reward()   # reward-gated go/no-go foraging choice
        else:
            raise ValueError(
                f"eval_dataset '{self.eval_dataset}' is not supported here; try 'flowers' "
                "(navigation is a separate task: `pixi run nav`)."
            )

    def _flower_representations(self, ablation: bool = False):
        """Build the flower representation panel -> (ScanEncoder, ordered rep names).

        Every representation lifts a feature map through the SAME kind of fixed KC projection, so only
        the code differs at the shared MBON readout. The ablation panel feeds channel-masked slices of
        the chromatic feature ``[achromatic, G-B, B-G]`` through the *same* projection (seed 23), which
        isolates which sub-channel carries the discrimination and which survives corruption.
        """
        proj_chroma = RetinotopicKCProjection(in_channels=3, code_dim=self.code_dim, seed=23).to(self.device)
        proj_clahe = RetinotopicKCProjection(in_channels=1, code_dim=self.code_dim, seed=29).to(self.device)
        proj_view = RetinotopicKCProjection(in_channels=6, code_dim=self.code_dim, seed=23).to(self.device)

        def chroma(keep):
            """Feature fn: project the chromatic map with only channels in ``keep`` retained (None = all)."""
            def fn(v, maps, dev):
                cf = maps["chromatic_feature"]
                if keep is not None:
                    mask = torch.zeros(cf.shape[1], device=cf.device)
                    if keep:
                        mask[list(keep)] = 1.0
                    cf = cf * mask.view(1, -1, 1, 1)
                return proj_chroma(cf)
            return fn

        def view_fn(v, maps, dev):
            """The unified backbone view: form + colour, all stages upstream."""
            return proj_view(torch.cat([maps["contrast_features"], maps["chromatic_feature"]], dim=1))

        reps = {"CLAHE": lambda v, maps, dev: proj_clahe(clahe_map(v, dev))}
        if ablation:
            # Channel-ablation panel: dissect which chromatic sub-channel carries the signal
            # (orthogonal to the --ablate stage-removal study).
            reps["chromatic (full)"] = chroma(None)
            reps["achromatic-only"] = chroma((0,))       # keep [ach], zero the opponent channels
            reps["opponent-only"] = chroma((1, 2))       # zero [ach], keep G-B, B-G
            reps["chroma-ablated"] = chroma(())          # all zero -> chance sanity
        else:
            reps["vision_kc"] = view_fn
        return ScanEncoder(self.backbone, reps, self.device), list(reps)

    def _eval_flowers_reward(self) -> None:
        """Reward-gated go/no-go: learn to approach a rewarded subset of flowers, reject the rest."""
        ks = list(self.ks)
        kmax = max(ks)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        paths, labels, class_dirs = self._sample_dataset(rng)
        class_names = [d.name for d in class_dirs]
        n_classes = len(class_dirs)
        n_rewarded = int(getattr(self, "n_rewarded", 5))
        rewarded = choose_rewarded(n_classes, n_rewarded, self.seed)
        reward = np.isin(labels, rewarded).astype(np.float32)
        base_rate = float(reward.mean())
        n_folds = int(getattr(self, "cv_folds", 5))
        n_repeats = int(getattr(self, "cv_repeats", 1))
        corruption = str(getattr(self, "corruption", "none"))
        ablation = bool(getattr(self, "ablation", False))
        lr = float(getattr(self, "reward_lr", 1.0))
        self.logger.info(f"17flowers go/no-go{' [ablation]' if ablation else ''}: {len(paths)} images; "
                         f"reward {len(rewarded)}/{n_classes} classes "
                         f"[{', '.join(class_names[i] for i in rewarded)}]  P(reward)={base_rate:.3f}")

        encoder, reps = self._flower_representations(ablation=ablation)
        base = build_views(paths, self.device, obj=int(getattr(self, "patch_size", 75)),
                           canvas=int(getattr(self, "canvas_size", 135)))
        folds = stratified_folds(reward, n_folds, self.seed, repeats=n_repeats)

        if corruption == "none":
            self.logger.info(f"Accumulating approach evidence over up to K={kmax} fixations; "
                             f"{n_folds}-fold CV x{n_repeats} (balanced accuracy / AUC, chance 0.5)...")
            codes_by_fix = encoder.encode_scan_stack(base, kmax)
            metrics = evaluate_gonogo_cv(codes_by_fix, reward, folds, reps, ks, lr=lr)
            if ablation:
                self._report_ablation(reps, kmax, metrics, base_rate, len(folds))
            else:
                self._report_gonogo(reps, ks, kmax, metrics, base_rate, len(folds))
            return

        # Corruption battery: each fixation is independently corrupted, so accumulating looks denoises.
        severities = [0.0] + [float(s) for s in getattr(self, "severities", [1.0]) if float(s) > 0]
        self.logger.info(f"Corruption '{corruption}' sweep {severities[1:]}; accumulating up to K={kmax} "
                         f"fixations; {n_folds}-fold CV x{n_repeats} (AUC, chance 0.5)...")
        grid = {}
        for sev in severities:
            policy = affine_scan if sev == 0.0 else corrupting_scan(corruption, sev)
            stack = encoder.encode_scan_stack(base, kmax, scan_policy=policy)
            grid[sev] = evaluate_gonogo_cv(stack, reward, folds, reps, ks, lr=lr)
        if ablation:
            self._report_ablation_corruption(reps, kmax, grid, corruption, base_rate, len(folds))
        else:
            self._report_gonogo_corruption(reps, ks, kmax, grid, corruption, base_rate, len(folds))

    def _report_gonogo_corruption(self, reps, ks, kmax, grid, corruption, base_rate, n_folds) -> None:
        # AUC at the shortest and longest scan for each rep, across corruption severities.
        rows = []
        for sev in sorted(grid):
            m = grid[sev]
            rows.append([
                f"{sev:.1f}",
                f"{m[('CLAHE', 1)]['auc']:.3f}", f"{m[('CLAHE', kmax)]['auc']:.3f}",
                f"{m[('vision_kc', 1)]['auc']:.3f}", f"{m[('vision_kc', kmax)]['auc']:.3f}",
                f"{m[('vision_kc', kmax)]['auc'] - m[('CLAHE', kmax)]['auc']:+.3f}",
            ])
        self._log_table(render_table(
            ["severity", "CLAHE K=1", f"CLAHE K={kmax}", "chroma K=1", f"chroma K={kmax}", f"colour-CLAHE@K{kmax}"],
            rows,
            title=f"\nReward go/no-go AUC under '{corruption}' corruption  ({n_folds}-fold CV; "
                  f"P(reward)={base_rate:.3f}, chance 0.5)"))

        # Isolate the scan benefit: AUC gained by accumulating K=1 -> K=max at each severity.
        grows = []
        for sev in sorted(grid):
            m = grid[sev]
            grows.append([f"{sev:.1f}",
                          f"{m[('CLAHE', kmax)]['auc'] - m[('CLAHE', 1)]['auc']:+.3f}",
                          f"{m[('vision_kc', kmax)]['auc'] - m[('vision_kc', 1)]['auc']:+.3f}"])
        self._log_table(render_table(
            ["severity", f"CLAHE dAUC(K1->K{kmax})", f"chroma dAUC(K1->K{kmax})"], grows,
            title="\nScan benefit: AUC recovered by accumulating fixations"))
        self.logger.info("")
        self.logger.info("Under isoluminant corruption the opponent-colour code holds while grayscale CLAHE "
                         "falls; accumulating fixations recovers AUC as independent corruption averages out.")

    def _report_ablation(self, reps, kmax, metrics, base_rate, n_folds) -> None:
        # Which sub-channel of the chromatic feature carries the reward discrimination (clean).
        rows = [[r, f"{metrics[(r, 1)]['auc']:.3f}+-{metrics[(r, 1)]['ci_auc']:.3f}",
                 f"{metrics[(r, kmax)]['auc']:.3f}+-{metrics[(r, kmax)]['ci_auc']:.3f}"] for r in reps]
        self._log_table(render_table(
            ["representation", "K=1 AUC+-ci", f"K={kmax} AUC+-ci"], rows,
            title=f"\nChannel ablation: reward go/no-go AUC  ({n_folds}-fold CV; "
                  f"P(reward)={base_rate:.3f}, chance 0.5)"))
        self.logger.info("")
        self.logger.info("Opponent-only tracks the full chromatic code; achromatic-only sits near CLAHE; "
                         "zeroing chroma (ablate_chromatic) falls to chance -- the signal is the opponent colour.")

    def _report_ablation_corruption(self, reps, kmax, grid, corruption, base_rate, n_folds) -> None:
        # Each sub-channel's robustness: AUC at the longest scan across corruption severities.
        sevs = sorted(grid)
        rows = [[r, *[f"{grid[s][(r, kmax)]['auc']:.3f}" for s in sevs]] for r in reps]
        self._log_table(render_table(
            ["representation", *[f"sev {s:.1f}" for s in sevs]], rows,
            title=f"\nChannel ablation under '{corruption}' corruption: AUC@K{kmax}  ({n_folds}-fold CV; "
                  f"P(reward)={base_rate:.3f}, chance 0.5)"))
        self.logger.info("")
        self.logger.info("The opponent-only channel survives isoluminant corruption (G-B is spared); the "
                         "achromatic-only channel collapses with CLAHE -- the robustness is specifically the colour opponency.")

    def _report_gonogo(self, reps, ks, kmax, metrics, base_rate, n_folds) -> None:
        rows = []
        for r in reps:
            note = "ours" if r == "vision_kc" else "baseline"
            auc_cells = [f"{metrics[(r, k)]['auc']:.3f}+-{metrics[(r, k)]['ci_auc']:.3f}" for k in ks]
            m = metrics[(r, kmax)]
            rows.append([r, *auc_cells, f"{m['bacc']:.3f}+-{m['ci_bacc']:.3f}",
                         f"{m['hit']:.2f}", f"{m['fa']:.2f}", note])
        headers = ["representation", *[f"K={k} AUC+-ci" for k in ks],
                   f"bal.acc@K{kmax}", f"hit@K{kmax}", f"FA@K{kmax}", "note"]
        self._log_table(render_table(
            headers, rows,
            title=f"\nReward go/no-go vs scan length K  ({n_folds}-fold CV, mean+-95%CI; "
                  f"P(reward)={base_rate:.3f}, chance AUC/bal.acc = 0.5)"))

        prows = []
        for k in ks:
            d = paired_delta(metrics[("vision_kc", k)]["fold_auc"], metrics[("CLAHE", k)]["fold_auc"])
            has_p = d["p"] == d["p"]
            prows.append([f"K={k}", f"{d['mean']:+.3f}+-{d['ci']:.3f}",
                          f"{d['p']:.1e}" if has_p else "n/a", "yes" if has_p and d["p"] < 0.05 else "no"])
        self._log_table(render_table(
            ["scan", "vision_kc - CLAHE (AUC)", "paired-t p", "p<0.05"], prows,
            title="\nPaired advantage over CLAHE (aligned folds)"))
        self.logger.info("")
        self.logger.info("Reward-gated approach MBON on both representations, so only the code differs: the "
                         "opponent-colour code learns which flowers pay; the grayscale CLAHE code cannot.")

    # ------------------------------------------------------------------ flower identification
    def _sample_dataset(self, rng):
        """Sample up to ``per_class`` jpgs from each 17flowers class dir -> (paths, labels, class_dirs)."""
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
        return paths, np.asarray(labels), class_dirs

    def _eval_flowers(self) -> None:
        reps = ["CLAHE", "vision_kc"]
        ks = list(self.ks)
        kmax = max(ks)
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        paths, labels, class_dirs = self._sample_dataset(rng)
        class_names = [d.name for d in class_dirs]
        n_classes = len(class_dirs)
        chance = 1.0 / n_classes
        n_folds = int(getattr(self, "cv_folds", 5))
        n_repeats = int(getattr(self, "cv_repeats", 1))
        depression = float(getattr(self, "depression", 0.5))
        self.logger.info(f"17flowers: {len(paths)} images, {n_classes} classes (chance {chance:.3f})")
        self.logger.info(f"Scan-accumulating the full vision code over up to K={kmax} fixations; "
                         f"{n_folds}-fold CV x{n_repeats} (MBON depression {depression})...")

        proj_view = RetinotopicKCProjection(in_channels=6, code_dim=self.code_dim, seed=23).to(self.device)
        proj_clahe = RetinotopicKCProjection(in_channels=1, code_dim=self.code_dim, seed=29).to(self.device)
        # Both representations lift a feature map through the same kind of fixed KC projection, so only
        # the code differs at the (shared) MBON readout. ``vision_kc`` is the unified backbone view
        # [ON, OFF, luminance | achromatic, G-B, B-G] -- every stage (hex, adapt, DoG, opponency) is
        # upstream of it -- versus the grayscale CLAHE luminance template.
        representations = {
            "CLAHE": lambda v, maps, dev: proj_clahe(clahe_map(v, dev)),
            "vision_kc": lambda v, maps, dev: proj_view(
                torch.cat([maps["contrast_features"], maps["chromatic_feature"]], dim=1)),
        }
        encoder = ScanEncoder(self.backbone, representations, self.device)
        base = build_views(paths, self.device, obj=int(getattr(self, "patch_size", 75)),
                           canvas=int(getattr(self, "canvas_size", 135)))
        snapshots = encoder.scan(base, ks)

        folds = stratified_folds(labels, n_folds, self.seed, repeats=n_repeats)
        metrics = evaluate_cv(snapshots, labels, folds, reps, ks, depression=depression, topk=3)
        self._report_flowers(reps, ks, kmax, metrics, chance, class_names, len(folds))
        if bool(getattr(self, "mbon_sweep", False)):
            self._report_mbon_sweep(snapshots, labels, folds, kmax, chance)

    def _report_flowers(self, reps, ks, kmax, metrics, chance, class_names, n_folds) -> None:
        # Headline: top-1 (mean +- 95% CI) at each scan length, plus top-3 and x chance at the longest scan.
        rows = []
        for r in reps:
            note = "ours" if r == "vision_kc" else "baseline"
            cells = [f"{metrics[(r, k)]['top1']:.3f}+-{metrics[(r, k)]['ci']:.3f}" for k in ks]
            m = metrics[(r, kmax)]
            rows.append([r, "MBON", *cells, f"{m['topk']:.3f}", f"{m['top1'] / chance:.1f}x", note])
        headers = ["representation", "readout", *[f"K={k} top1+-ci" for k in ks],
                   f"top3@K{kmax}", "x chance", "note"]
        self._log_table(render_table(
            headers, rows,
            title=f"\nFlower identification vs scan length K  ({n_folds}-fold CV, mean+-95%CI; "
                  f"chance {chance:.3f})"))

        # Paired advantage of the colour code over CLAHE on aligned folds.
        prows = []
        for k in ks:
            d = paired_delta(metrics[("vision_kc", k)]["fold_top1"], metrics[("CLAHE", k)]["fold_top1"])
            has_p = d["p"] == d["p"]  # not NaN
            sig = "yes" if has_p and d["p"] < 0.05 else "no"
            prows.append([f"K={k}", f"{d['mean']:+.3f}+-{d['ci']:.3f}",
                          f"{d['p']:.1e}" if has_p else "n/a", sig])
        self._log_table(render_table(
            ["scan", "vision_kc - CLAHE (top1)", "paired-t p", "p<0.05"], prows,
            title="\nPaired advantage over CLAHE (aligned folds)"))

        # Per-class recall of the colour code at the longest scan + its dominant off-diagonal confusion.
        conf = metrics[("vision_kc", kmax)]["confusion"]
        pca = metrics[("vision_kc", kmax)]["per_class_acc"]
        off = conf.copy(); np.fill_diagonal(off, 0)
        crows = []
        for ci, name in enumerate(class_names):
            worst = int(off[ci].argmax())
            conf_note = f"{class_names[worst]} ({off[ci, worst]})" if off[ci, worst] > 0 else "-"
            crows.append([name, f"{pca[ci]:.3f}", conf_note])
        self._log_table(render_table(
            ["class", f"recall@K{kmax}", "most confused with (n)"], crows,
            title=f"\nPer-class accuracy (vision_kc, K={kmax})"))

        # Persist the full confusion matrices (rows = true, cols = predicted) for offline inspection.
        for r in reps:
            path = self.outdir / f"confusion_{r}_K{kmax}.csv"
            np.savetxt(path, metrics[(r, kmax)]["confusion"], fmt="%d", delimiter=",",
                       header="true\\pred," + ",".join(class_names), comments="")
            self.logger.info(f"saved {path}")
        self.logger.info("")
        self.logger.info("Same per-class MBON readout on both representations, so only the code differs: "
                         "the full vision KC code (form + colour, all backbone stages) identifies flowers; "
                         "the grayscale CLAHE KC code cannot.")

    def _report_mbon_sweep(self, snapshots, labels, folds, kmax, chance) -> None:
        """Readout-robustness check: CV top-1 at the longest scan across MBON depression / graded settings."""
        reps = ["CLAHE", "vision_kc"]
        rows = []
        for depression in (0.25, 0.5, 0.75, 1.0):
            for graded in (False, True):
                m = evaluate_cv(snapshots, labels, folds, reps, [kmax],
                                depression=depression, topk=3, graded=graded)
                rows.append([f"{depression:.2f}", "yes" if graded else "no",
                             f"{m[('CLAHE', kmax)]['top1']:.3f}+-{m[('CLAHE', kmax)]['ci']:.3f}",
                             f"{m[('vision_kc', kmax)]['top1']:.3f}+-{m[('vision_kc', kmax)]['ci']:.3f}"])
        self._log_table(render_table(
            ["depression", "graded", "CLAHE top1", "vision_kc top1"], rows,
            title=f"\nMBON readout robustness sweep (K={kmax}, {len(folds)} folds; chance {chance:.3f})"))

    def _log_table(self, table) -> None:
        self.logger.info("")
        for line in table.splitlines():
            self.logger.info(line)
