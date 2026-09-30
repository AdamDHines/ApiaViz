"""Descriptive code metrics and cluster bootstrap confidence intervals."""
import numpy as np
import torch


def code_metrics(codes, positions=None):
    active = (codes > 0).float()
    occupancy = active.mean(1)
    lifetime = active.mean(0)
    result = {"active_fraction": float(occupancy.mean()),
              "active_fraction_std": float(occupancy.std(unbiased=False)),
              "silent_fraction": float((occupancy == 0).float().mean()),
              "unused_kc_fraction": float((lifetime == 0).float().mean()),
              "lifetime_activity_std": float(lifetime.std(unbiased=False)),
              "code_dim": codes.shape[1]}
    if len(codes) > 1:
        # Upper triangular pairs only; no self-pairs. Bound the diagnostic workload.
        a = active[:256]
        intersection = a @ a.T
        union = a.sum(1)[:, None] + a.sum(1)[None, :] - intersection
        overlap = intersection / union.clamp_min(1)
        row, col = torch.triu_indices(len(a), len(a), offset=1, device=a.device)
        valid = union[row, col] > 0
        result["mean_pair_jaccard"] = float(overlap[row, col][valid].mean()) if valid.any() else None
        if positions is not None:
            pos = torch.as_tensor(positions[:len(a)], device=a.device, dtype=torch.float32)
            distances = torch.cdist(pos, pos)[row, col]
            for name, mask in (("near", distances <= .25), ("distant", distances >= 1.)):
                mask &= valid
                result[f"{name}_jaccard"] = float(overlap[row, col][mask].mean()) if mask.any() else None
                result[f"{name}_collision_fraction"] = float((overlap[row, col][mask] == 1).float().mean()) if mask.any() else None
    return result


def cluster_bootstrap(values, groups, seed=7, samples=1000):
    """Mean and percentile CI, resampling whole ants/images (not views or folds).

    One cluster cannot estimate between-cluster uncertainty; return a null CI.
    """
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    if len(values) == 0 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires aligned nonempty finite values and groups")
    unique = np.unique(groups)
    result = {"mean": float(values.mean()), "clusters": len(unique), "ci95": None}
    if len(unique) < 2:
        return result
    clusters = [values[groups == g] for g in unique]
    rng = np.random.default_rng(seed)
    estimates = [np.concatenate([clusters[i] for i in rng.integers(len(unique), size=len(unique))]).mean()
                 for _ in range(samples)]
    result["ci95"] = np.quantile(estimates, [.025, .975]).tolist()
    return result


def reward_metrics(reward, score, decision):
    from sklearn.metrics import roc_auc_score
    reward = np.asarray(reward, dtype=bool)
    decision = np.asarray(decision, dtype=bool)
    hit = float(decision[reward].mean()) if reward.any() else None
    fa = float(decision[~reward].mean()) if (~reward).any() else None
    both = hit is not None and fa is not None
    return {"auc": float(roc_auc_score(reward, score)) if both else None,
            "balanced_accuracy": (hit + 1 - fa) / 2 if both else None,
            "hit": hit, "false_alarm": fa}


def reward_bootstrap(reward, score, decision, image_ids, seed=7, samples=1000):
    """Image-cluster bootstrap: all repeats of an image move together."""
    reward, score, decision, image_ids = map(np.asarray, (reward, score, decision, image_ids))
    result = reward_metrics(reward, score, decision)
    unique = np.unique(image_ids)
    groups = [np.flatnonzero(image_ids == key) for key in unique]
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in result}
    if len(unique) > 1:
        for _ in range(samples):
            idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            metrics = reward_metrics(reward[idx], score[idx], decision[idx])
            for key, value in metrics.items():
                if value is not None:
                    draws[key].append(value)
    return {key: {"value": value, "ci95": np.quantile(draws[key], [.025, .975]).tolist()
                  if draws[key] else None} for key, value in result.items()}
