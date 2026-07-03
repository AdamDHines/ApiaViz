"""Flower-identification evaluation components.

Phase-0 refactor of ``EvalVision._eval_flowers`` into three composable pieces so later phases
can sweep any axis (metrics, corruptions, scan policies) with CLAHE always paired against the
chromatic KC code:

  * view building + (future) corruptions        -> ``build_views`` / ``jitter`` / ``affine_scan``
  * representation encoding + scan integration   -> ``ScanEncoder`` (+ ``RunningMean``)
  * per-class MBON readout + cross-validated eval -> ``mbon_novelty`` / ``evaluate_cv``

This module is behaviour-preserving: with the default affine scan policy, running-mean
integration and a single train/test split it reproduces the original numbers exactly. The seams
(``scan_policy``, ``integration``, ``splits``, the representation registry) are where the later
phases plug in without touching the encode/readout core.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from apiaviz.nav.retino_kc import AntiHebbianMBON, RewardMBON

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# --------------------------------------------------------------------- views + scan policy
def build_views(paths, device, obj=75, canvas=135) -> torch.Tensor:
    """Paste each flower's (G,B) channels, resized to ``obj``x``obj`` and centred, on a neutral
    ``canvas``x``canvas`` grey field -> [N,2,canvas,canvas] in [-1,1].

    The object is centred automatically (equal margins), leaving surround for the scan jitter to move
    over. Default 75x75 object on a 135x135 field -- the earlier 36x36 patch was too small to resolve.
    """
    if not 0 < obj <= canvas:
        raise ValueError(f"obj must be in (0, canvas]; got obj={obj}, canvas={canvas}")
    oy = ox = (canvas - obj) // 2
    views = []
    for path in paths:
        arr = np.asarray(Image.open(path).convert("RGB").resize((obj, obj)), dtype=np.float32) / 255.0
        canvas_arr = np.full((canvas, canvas, 2), 0.5, dtype=np.float32)
        canvas_arr[oy:oy + obj, ox:ox + obj, :] = arr[:, :, 1:3]  # (G, B)
        views.append((torch.from_numpy(canvas_arr).permute(2, 0, 1) - 0.5) / 0.5)
    return torch.stack(views).to(device)


def jitter(views, seed, max_rot=15.0, max_trans=0.10, max_scale=0.1) -> torch.Tensor:
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


def affine_scan(base_views: torch.Tensor, fix: int) -> torch.Tensor:
    """Default scan policy: fixation 0 is the base view, later fixations are small affine jitters."""
    return base_views if fix == 0 else jitter(base_views, seed=1000 + fix)


# --------------------------------------------------------------------- test-time corruptions
def corrupt_luminance(views: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    """Multi-scale luminance shift added *equally* to both (G,B) channels of ``[N,2,H,W]`` views.

    The shared additive field moves luminance (mean of G,B) but leaves G-B untouched, so the opponent
    colour signal survives while grayscale luminance is corrupted -- the isoluminant lighting/weather
    model, the flower analogue of nav's ``luminance_corrupt``.
    """
    if severity <= 0:
        return views
    n, _, h, w = views.shape
    gen = torch.Generator().manual_seed(int(seed))
    field = torch.zeros(n, 1, h, w)
    for scale in (2, 4, 8, 16):
        small = torch.randn(n, 1, max(1, h // scale), max(1, w // scale), generator=gen)
        field = field + F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    field = field / field.flatten(1).std(dim=1).clamp_min(1e-6).view(n, 1, 1, 1)
    return (views + severity * 0.4 * field.to(views.device)).clamp(-1.0, 1.0)


def corrupt_noise(views: torch.Tensor, severity: float, seed: int) -> torch.Tensor:
    """Independent per-pixel Gaussian sensor noise (hits both channels) -> ``[N,2,H,W]`` in [-1,1]."""
    if severity <= 0:
        return views
    gen = torch.Generator().manual_seed(int(seed))
    noise = torch.randn(views.shape, generator=gen).to(views.device)
    return (views + severity * 0.3 * noise).clamp(-1.0, 1.0)


_CORRUPTIONS = {"luminance": corrupt_luminance, "noise": corrupt_noise}


def corrupting_scan(corruption: str, severity: float, seed0: int = 5000,
                    base_policy: Callable = affine_scan) -> Callable:
    """Scan policy that applies ``corruption`` independently per fixation on top of ``base_policy``.

    Per-fixation independence (seed varies with ``fix``) is what lets evidence accumulation across the
    scan denoise the decision -- more looks average out uncorrelated corruption.
    """
    fn = _CORRUPTIONS.get(corruption)
    if fn is None:
        return base_policy

    def policy(base_views, fix):
        return fn(base_policy(base_views, fix), severity, seed0 + fix)

    return policy


def clahe_map(views, device, size=24) -> torch.Tensor:
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
    return torch.from_numpy(arr).to(device)


# --------------------------------------------------------------------- scan integration
class RunningMean:
    """Accumulate per-fixation KC codes as a running sum; snapshot the mean over the first K fixations.

    Sparse codes are k-WTA'd per fixation, then averaged -- the original scan-accumulation scheme.
    Alternative integrations (sum-then-resparsify, readout-level evidence accumulation) will implement
    the same ``accumulate`` / ``snapshot`` interface in a later phase.
    """

    def accumulate(self, running: dict | None, feats: dict, reps: list) -> dict:
        if running is None:
            return {r: feats[r].astype(np.float64) for r in reps}
        return {r: running[r] + feats[r] for r in reps}

    def snapshot(self, running: dict, k: int, reps: list) -> dict:
        return {r: (running[r] / k).astype(np.float32) for r in reps}


# --------------------------------------------------------------------- encoder
class ScanEncoder:
    """Encode scanned fixations into accumulated KC codes: one code matrix per representation per K.

    ``representations`` maps a rep name to ``fn(views, maps, device) -> [N, code_dim]`` KC code,
    evaluated once per fixation. The backbone is run once per batch and its map dict passed to every
    representation fn, so reps that read the backbone (chromatic) and reps that do not (CLAHE) share a
    single forward pass. ``scan_policy(base_views, fix)`` supplies the view for each fixation and
    ``integration`` accumulates codes across fixations, snapshotting at each requested K.
    """

    def __init__(self, backbone, representations: dict[str, Callable], device, batch: int = 128):
        self.backbone = backbone
        self.representations = representations
        self.device = device
        self.batch = int(batch)

    @torch.no_grad()
    def encode_fixation(self, views) -> dict:
        """One fixation -> {rep: [N, code_dim] KC code}."""
        chunks = {r: [] for r in self.representations}
        for i in range(0, views.size(0), self.batch):
            v = views[i:i + self.batch]
            maps = self.backbone(v, return_maps=True)
            for r, fn in self.representations.items():
                chunks[r].append(fn(v, maps, self.device).cpu().numpy())
        return {r: np.concatenate(c, axis=0) for r, c in chunks.items()}

    def scan(self, base_views, ks, scan_policy: Callable = affine_scan, integration: RunningMean | None = None) -> dict:
        """Scan ``max(ks)`` fixations; return ``{k: {rep: [N, code_dim] accumulated code}}`` for k in ks."""
        integration = integration if integration is not None else RunningMean()
        reps = list(self.representations)
        running, snapshots = None, {}
        for fix in range(max(ks)):
            views = scan_policy(base_views, fix)
            feats = self.encode_fixation(views)
            running = integration.accumulate(running, feats, reps)
            if fix + 1 in ks:
                snapshots[fix + 1] = integration.snapshot(running, fix + 1, reps)
        return snapshots

    def encode_scan_stack(self, base_views, n_fix: int, scan_policy: Callable = affine_scan) -> dict:
        """Encode ``n_fix`` fixations WITHOUT integrating -> ``{rep: [n_fix, N, code_dim]}``.

        The reward go/no-go path accumulates *evidence* (the scalar approach score) across fixations
        rather than averaging the sparse codes, so it needs the per-fixation codes kept separate.
        """
        per = {r: [] for r in self.representations}
        for fix in range(int(n_fix)):
            feats = self.encode_fixation(scan_policy(base_views, fix))
            for r in self.representations:
                per[r].append(feats[r])
        return {r: np.stack(v, axis=0) for r, v in per.items()}


# --------------------------------------------------------------------- MBON readout
def mbon_novelty(feats, labels, train_idx, test_idx, depression: float = 0.5, graded: bool = False):
    """Per-class anti-Hebbian MBON novelty for each test view: ``[n_test, n_classes]`` (lower = familiar).

    One MBON stores each class's training KC codes; a test view's novelty against every class is the
    per-class readout used for identification (argmin -> predicted class). Returns ``(novelty, classes)``.
    """
    codes = torch.from_numpy(feats).float()
    classes = np.unique(labels[train_idx])
    novelty = np.zeros((len(test_idx), len(classes)), dtype=np.float64)
    for j, c in enumerate(classes):
        mbon = AntiHebbianMBON(code_dim=feats.shape[1], depression=depression, graded=graded)
        mbon.store(codes[train_idx][labels[train_idx] == c])
        novelty[:, j] = mbon(codes[test_idx]).numpy()
    return novelty, classes


def _topk_hit(novelty, classes, y_true, k: int) -> np.ndarray:
    """Boolean ``[n_test]``: is the true class among the k most-familiar (lowest-novelty) classes?"""
    k = min(int(k), novelty.shape[1])
    topk = classes[np.argsort(novelty, axis=1)[:, :k]]  # [n_test, k]
    return (topk == y_true[:, None]).any(axis=1)


# --------------------------------------------------------------------- cross-validated evaluation
def _ci95(values) -> float:
    """Half-width of the 95% CI of the mean (Student-t); 0 for fewer than 2 samples."""
    from scipy import stats
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return 0.0
    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def stratified_folds(labels, n_folds: int, seed: int, repeats: int = 1):
    """Stratified k-fold (optionally repeated with reshuffled folds) -> list of ``(train_idx, test_idx)``.

    Each repeat reshuffles the fold assignment with a fresh seed, giving more split samples for the CI
    at no extra encoding cost. Every sample is a test exactly once per repeat, so predictions pool into
    a full confusion matrix over all samples.
    """
    from sklearn.model_selection import StratifiedKFold
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for stratified CV, got {n_folds}")
    x = np.zeros(len(labels))
    folds = []
    for r in range(max(1, int(repeats))):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed) + r)
        folds += [(tr, te) for tr, te in skf.split(x, labels)]
    return folds


def evaluate_cv(snapshots, labels, folds, reps, ks, depression: float = 0.5,
                topk: int = 3, graded: bool = False) -> dict:
    """Cross-validated per-class-MBON metrics for every ``(rep, K)``.

    Returns ``{(rep, k): {top1, ci, topk, fold_top1, confusion, per_class_acc}}`` where ``top1``/``ci``
    are the mean and 95% CI half-width of top-1 accuracy across folds, ``fold_top1`` the per-fold
    accuracies (aligned across reps by fold index, for paired tests), and ``confusion`` the ``[C,C]``
    matrix (rows = true, cols = predicted) pooled over all folds.
    """
    n_classes = len(np.unique(labels))
    metrics = {}
    for r in reps:
        for k in ks:
            feats = snapshots[k][r]
            fold_top1, fold_topk = [], []
            confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
            for tr, te in folds:
                novelty, classes = mbon_novelty(feats, labels, tr, te, depression, graded)
                y_true = labels[te]
                preds = classes[novelty.argmin(axis=1)]
                fold_top1.append(float((preds == y_true).mean()))
                fold_topk.append(float(_topk_hit(novelty, classes, y_true, topk).mean()))
                np.add.at(confusion, (y_true, preds), 1)
            support = confusion.sum(axis=1)
            per_class_acc = np.divide(np.diag(confusion), support,
                                      out=np.zeros(n_classes), where=support > 0)
            metrics[(r, k)] = {
                "top1": float(np.mean(fold_top1)),
                "ci": _ci95(fold_top1),
                "topk": float(np.mean(fold_topk)),
                "fold_top1": np.asarray(fold_top1, dtype=np.float64),
                "confusion": confusion,
                "per_class_acc": per_class_acc,
            }
    return metrics


def paired_delta(fold_a, fold_b) -> dict:
    """Paired chromatic-vs-baseline comparison over aligned folds: mean delta, 95% CI, paired-t p."""
    from scipy import stats
    a = np.asarray(fold_a, dtype=np.float64)
    b = np.asarray(fold_b, dtype=np.float64)
    delta = a - b
    p = float(stats.ttest_rel(a, b).pvalue) if delta.size >= 2 and np.any(delta != 0) else float("nan")
    return {"mean": float(delta.mean()), "ci": _ci95(delta), "p": p}


# --------------------------------------------------------------------- reward go/no-go
def choose_rewarded(n_classes: int, n_rewarded: int, seed: int) -> np.ndarray:
    """Pick which class indices are rewarded (deterministic from ``seed``); sorted."""
    rng = np.random.default_rng(seed)
    n_rewarded = int(np.clip(n_rewarded, 1, n_classes - 1))
    return np.sort(rng.choice(n_classes, size=n_rewarded, replace=False))


def _gonogo_stats(score, reward, thr) -> dict:
    """Go/No-Go decision stats at threshold ``thr`` plus threshold-free AUC.

    Reward is imbalanced (few rewarded classes), so raw accuracy is misleading -- balanced accuracy
    ``0.5*(hit + correct-reject)`` and AUC are the honest headline (both chance = 0.5).
    """
    from sklearn.metrics import roc_auc_score
    go = np.asarray(score) > thr
    rew = np.asarray(reward) > 0.5
    hit = float(go[rew].mean()) if rew.any() else float("nan")      # P(go | rewarded)
    fa = float(go[~rew].mean()) if (~rew).any() else float("nan")   # P(go | unrewarded)
    bacc = 0.5 * (hit + (1.0 - fa))
    auc = float(roc_auc_score(rew.astype(int), score)) if rew.any() and (~rew).any() else float("nan")
    return {"hit": hit, "fa": fa, "bacc": bacc, "auc": auc}


def evaluate_gonogo_cv(codes_by_fix, reward, folds, reps, ks, lr: float = 1.0, graded: bool = False) -> dict:
    """Reward-gated go/no-go over CV folds with fixation *evidence accumulation*.

    ``codes_by_fix[rep]`` is ``[n_fix, N, code_dim]`` per-fixation KC codes (unaveraged). Per fold a
    ``RewardMBON`` learns the appetitive/aversive valence boundary from the training split's first
    fixation; the approach score for every sample is then read at each fixation and **averaged over the
    first k fixations** (integrating the decision variable, not the sparse codes -- the repair for the
    scan regression). The go threshold is the midpoint of the train rewarded/unrewarded score means.

    Returns ``{(rep, k): {auc, bacc, hit, fa, ci_auc, ci_bacc, fold_auc, fold_bacc}}``.
    """
    reward = np.asarray(reward, dtype=np.float32)
    metrics = {}
    for r in reps:
        stack = codes_by_fix[r]                 # [K, N, D]
        n_fix, _, code_dim = stack.shape
        per_fold = {k: {"auc": [], "bacc": [], "hit": [], "fa": []} for k in ks}
        for tr, te in folds:
            mbon = RewardMBON(code_dim=code_dim, lr=lr, graded=graded)
            # Learn the valence over the SAME jitter distribution the scan produces (every training
            # fixation), so the prototype is fixation-robust and accumulating looks denoises the decision
            # rather than dragging a clean estimate toward off-centre views.
            train_codes = stack[:, tr, :].reshape(-1, code_dim)
            train_reward = np.tile(reward[tr], n_fix)
            mbon.store(torch.from_numpy(train_codes).float(), train_reward)
            scores = np.stack([mbon(torch.from_numpy(stack[f]).float()).numpy()
                               for f in range(n_fix)], axis=0)               # [K, N] approach drive
            for k in ks:
                score = scores[:k].mean(axis=0)                             # accumulate evidence over k looks
                tr_rew, tr_unrew = score[tr][reward[tr] > 0.5], score[tr][reward[tr] <= 0.5]
                thr = 0.5 * (float(tr_rew.mean()) + float(tr_unrew.mean()))
                st = _gonogo_stats(score[te], reward[te], thr)
                for m in ("auc", "bacc", "hit", "fa"):
                    per_fold[k][m].append(st[m])
        for k in ks:
            metrics[(r, k)] = {m: float(np.nanmean(per_fold[k][m])) for m in ("auc", "bacc", "hit", "fa")}
            metrics[(r, k)]["ci_auc"] = _ci95(per_fold[k]["auc"])
            metrics[(r, k)]["ci_bacc"] = _ci95(per_fold[k]["bacc"])
            metrics[(r, k)]["fold_auc"] = np.asarray(per_fold[k]["auc"], dtype=np.float64)
            metrics[(r, k)]["fold_bacc"] = np.asarray(per_fold[k]["bacc"], dtype=np.float64)
    return metrics
