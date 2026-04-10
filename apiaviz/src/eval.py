import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import apiaviz.src.functional as avf
from apiaviz.dataset.datagen import DataMode, InsectVisionDataset
from apiaviz.src.modules import RewardMemoryHead, VisionBackbone, VisionProjection, resolve_kc_sparsity_target
from apiaviz.src.projection_utils import infer_projection_config, resolve_projection_checkpoint
from apiaviz.src.reward_memory import (
    REWARD_FEATURE_CHOICES,
    compute_reward_metrics,
    infer_reward_head_config,
    plot_reward_by_class,
    resolve_reward_feature_dim,
    resolve_rewarded_classes,
)
from apiaviz.src.spatial_finetune import resolve_pretrained_checkpoint


EVAL_FEATURE_CHOICES = (
    "lobula",
    "feature_vpn",
    "spatial_vpn",
    "conjunctive_vpn",
    "vpn",
    "kenyon_drive",
    "kenyon_code",
    "reward_logit",
    "reward_probability",
)


class ProjectionEvalModel(nn.Module):
    def __init__(self, backbone, projection):
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, x):
        backbone_outputs = self.backbone(x, return_maps=True)
        projection_outputs = self.projection(backbone_outputs)
        return {
            **backbone_outputs,
            **projection_outputs,
        }


class RewardEvalModel(nn.Module):
    def __init__(self, backbone, projection, reward_head, reward_feature):
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.reward_head = reward_head
        self.reward_feature = reward_feature

    def forward(self, x):
        backbone_outputs = self.backbone(x, return_maps=True)
        projection_outputs = self.projection(backbone_outputs)
        combined_outputs = {
            **backbone_outputs,
            **projection_outputs,
        }
        reward_outputs = self.reward_head(combined_outputs[self.reward_feature])
        return {
            **combined_outputs,
            **reward_outputs,
        }


class EvalVision:
    """Evaluate the v1 backbone-plus-projection stack on class separability."""

    def __init__(self, args, logger, outdir):
        for k in vars(args):
            setattr(self, k, getattr(args, k))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.outdir = Path(outdir)
        self.models_dir = Path(self.models_dir)
        self.dataset_dir = Path(self.dataset_dir)
        self.eval_feature = getattr(self, "eval_feature", "kenyon_code")
        self.projection_checkpoint_name = getattr(self, "projection_checkpoint", "")
        self.reward_checkpoint_name = getattr(self, "reward_checkpoint", "")
        self.reward_feature = getattr(self, "reward_feature", "kenyon_code")
        self.eval_input_size = int(getattr(self, "spatial_image_size", 64))
        self.kc_decay_factor = 0.9
        self.reward_eval_mode = self.eval_feature in {"reward_logit", "reward_probability"}

        if self.snn:
            raise NotImplementedError(
                "SNN evaluation has not been updated for the v1 backbone/projection stack."
            )
        if self.eval_feature not in EVAL_FEATURE_CHOICES:
            raise ValueError(
                f"Unsupported --eval_feature '{self.eval_feature}'. Choices: {', '.join(EVAL_FEATURE_CHOICES)}"
            )
        if self.reward_eval_mode and self.reward_feature not in REWARD_FEATURE_CHOICES:
            raise ValueError(
                f"Unsupported --reward_feature '{self.reward_feature}'. Choices: {', '.join(REWARD_FEATURE_CHOICES)}"
            )

        self.backbone_checkpoint = resolve_pretrained_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=self.backbone_checkpoint,
            model_name=self.lobula_plate_model,
        )
        self.projection_checkpoint = resolve_projection_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=self.projection_checkpoint_name,
            model_name=self.projection_model,
        )

        self.backbone = VisionBackbone().to(self.device)
        backbone_state_dict = torch.load(
            self.backbone_checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        self.backbone.eval()

        projection_state_dict = torch.load(
            self.projection_checkpoint,
            map_location=self.device,
            weights_only=True,
        )
        inferred_kc_dim = int(projection_state_dict["kc_projection.weight"].shape[0])
        self.projection_effective_kc_sparsity, self.projection_target_active_count = resolve_kc_sparsity_target(
            inferred_kc_dim,
            kc_sparsity=self.projection_kc_sparsity,
            kc_target_active=getattr(self, "projection_kc_target_active", 0),
        )
        projection_kwargs = infer_projection_config(
            projection_state_dict,
            effective_kc_sparsity=self.projection_effective_kc_sparsity,
            apl_feedback_strength=self.projection_apl_feedback_strength,
            apl_gain_adapt_rate=self.projection_apl_gain_adapt_rate,
            apl_threshold_lr=self.projection_apl_threshold_lr,
            apl_num_iters=self.projection_apl_num_iters,
        )
        self.projection_kc_dim = projection_kwargs["kc_dim"]
        self.projection = VisionProjection(**projection_kwargs).to(self.device)
        projection_load = self.projection.load_state_dict(projection_state_dict, strict=False)
        allowed_unexpected = {"kc_compete.running_mean", "kc_compete.thresholds"}
        allowed_missing = {
            "kc_compete.homeostatic_mean",
            "kc_compete.kc_thresholds",
            "kc_compete.apl_gain",
        }
        unexpected_keys = set(projection_load.unexpected_keys)
        missing_keys = set(projection_load.missing_keys)
        if not missing_keys.issubset(allowed_missing) or not unexpected_keys.issubset(allowed_unexpected):
            raise RuntimeError(
                "Projection checkpoint did not match the evaluation module. "
                f"Missing keys: {sorted(missing_keys)}. Unexpected keys: {sorted(unexpected_keys)}."
            )
        self.projection.eval()

        for module in (self.backbone, self.projection):
            for param in module.parameters():
                param.requires_grad_(False)

        self.reward_checkpoint = None
        if self.reward_eval_mode:
            self.reward_checkpoint = resolve_pretrained_checkpoint(
                models_dir=self.models_dir,
                checkpoint_name=self.reward_checkpoint_name,
                model_name=self.reward_model,
            )
            reward_head_input_dim = resolve_reward_feature_dim(
                self.reward_feature,
                lobula_dim=int(self.backbone.lobula.embedding.out_features),
                vpn_dim=int(projection_kwargs["vpn_dim"]),
                kc_dim=int(projection_kwargs["kc_dim"]),
            )
            reward_state_dict = torch.load(
                self.reward_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
            reward_head_kwargs = infer_reward_head_config(reward_state_dict, reward_head_input_dim)
            self.reward_head = RewardMemoryHead(**reward_head_kwargs).to(self.device)
            self.reward_head.load_state_dict(reward_state_dict, strict=True)
            self.reward_head.eval()
            for param in self.reward_head.parameters():
                param.requires_grad_(False)
            self.model = RewardEvalModel(
                self.backbone,
                self.projection,
                self.reward_head,
                self.reward_feature,
            ).to(self.device)
        else:
            self.model = ProjectionEvalModel(self.backbone, self.projection).to(self.device)
        self.model.eval()

        self.logger.info("Evaluation checkpoints loaded successfully.")
        self.logger.info(f"  - Backbone checkpoint: {self.backbone_checkpoint}")
        self.logger.info(f"  - Projection checkpoint: {self.projection_checkpoint}")
        if self.reward_eval_mode:
            self.logger.info(f"  - Reward checkpoint: {self.reward_checkpoint}")
        self.logger.info(f"  - Feature space: {self.eval_feature}")
        if self.reward_eval_mode:
            self.logger.info(f"  - Reward input feature: {self.reward_feature}")
        self.logger.info(
            "  - KC config: "
            f"dim={self.projection_kc_dim}, "
            f"target_active={self.projection_target_active_count}, "
            f"effective_sparsity={self.projection_effective_kc_sparsity:.4f}"
        )
        self.logger.info(
            "  - Competition: "
            f"APL-like inhibition (gain={self.projection_apl_feedback_strength:.4f}, "
            f"adapt={self.projection_apl_gain_adapt_rate:.4f}, "
            f"threshold_lr={self.projection_apl_threshold_lr:.4f}, "
            f"iters={self.projection_apl_num_iters})"
        )
        self.logger.info(f"  - Input preprocessing: resize to {self.eval_input_size}x{self.eval_input_size}, normalize GB to [-1, 1]")
        self.logger.info("")

        if not self.scanning:
            self.dataset = InsectVisionDataset(
                root=str(self.dataset_dir),
                dataset=self.eval_dataset,
                mode=DataMode.STATIC_FULL,
                logger=self.logger,
                patch_size=self.patch_size,
                samples_per_image=self.eval_samples,
            )
        else:
            self.dataset = InsectVisionDataset(
                root=str(self.dataset_dir),
                dataset=self.eval_dataset,
                mode=DataMode.SCANNING_PATCH,
                logger=self.logger,
                patch_size=self.patch_size,
                samples_per_image=self.eval_samples,
                num_steps=self.num_steps,
            )

        eval_num_workers = max(0, int(getattr(self, "num_workers", 0)))

        if self.device.type == "mps":
            self.loader = DataLoader(
                self.dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.loader = DataLoader(
                self.dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=eval_num_workers,
                pin_memory=self.device.type == "cuda",
            )

        if self.reward_eval_mode:
            self.rewarded_class_indices, self.rewarded_class_names = resolve_rewarded_classes(
                getattr(self.dataset, "class_names", []),
                getattr(self, "rewarded_classes", ""),
            )
            self.logger.info(
                "Rewarded classes for evaluation: "
                + ", ".join(
                    f"{name}({idx})" for idx, name in zip(self.rewarded_class_indices, self.rewarded_class_names)
                )
            )
            self.logger.info("")

    def _prepare_inputs(self, imgs):
        imgs = imgs.float()
        imgs = imgs.clamp(0.0, 1.0)
        if imgs.shape[-2:] != (self.eval_input_size, self.eval_input_size):
            imgs = F.interpolate(
                imgs,
                size=(self.eval_input_size, self.eval_input_size),
                mode="bilinear",
                align_corners=False,
            )
        return imgs * 2.0 - 1.0

    def _unpack_eval_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Evaluation batch must contain at least images and labels.")
            return batch[0], batch[1]
        raise TypeError(f"Unsupported batch type for evaluation: {type(batch)!r}")

    def _reward_labels_from_class_labels(self, class_labels):
        labels = np.asarray(class_labels, dtype=np.int64)
        rewarded_mask = np.isin(labels, np.asarray(self.rewarded_class_indices, dtype=np.int64))
        return rewarded_mask.astype(np.int64)

    def _forward_once(self, imgs):
        prepared = self._prepare_inputs(imgs.to(self.device))
        return self.model(prepared)

    def _extract_outputs(self, imgs, is_scanning_mode):
        if not is_scanning_mode:
            return self._forward_once(imgs)

        accumulated = None
        tracked_keys = (self.eval_feature, "kc_active_counts", "kc_active_fraction")

        for step in range(imgs.shape[1]):
            current_outputs = self._forward_once(imgs[:, step])
            if accumulated is None:
                accumulated = {
                    key: current_outputs[key]
                    for key in tracked_keys
                    if key in current_outputs
                }
                continue

            for key in tracked_keys:
                if key not in current_outputs:
                    continue
                accumulated[key] = (self.kc_decay_factor * accumulated[key]) + current_outputs[key]

        return accumulated

    def _save_metrics(self, metrics, kenyon_stats, sparse_overlap=None, reward_config=None):
        payload = {
            "feature_space": self.eval_feature,
            "dataset": {
                "name": self.eval_dataset,
                "root": str(self.dataset.root),
                "samples": len(self.dataset),
                "classes": len(getattr(self.dataset, "class_names", [])),
                "scanning": bool(self.scanning),
            },
            "checkpoints": {
                "backbone": str(self.backbone_checkpoint),
                "projection": str(self.projection_checkpoint),
            },
            "projection_config": {
                "kc_dim": int(self.projection_kc_dim),
                "target_active_count": int(self.projection_target_active_count),
                "effective_sparsity": float(self.projection_effective_kc_sparsity),
                "competition_module": "APLCompetition",
                "apl_feedback_strength": float(self.projection_apl_feedback_strength),
                "apl_gain_adapt_rate": float(self.projection_apl_gain_adapt_rate),
                "apl_threshold_lr": float(self.projection_apl_threshold_lr),
                "apl_num_iters": int(self.projection_apl_num_iters),
            },
            "metrics": {key: float(value) for key, value in metrics.items()},
            "kenyon_activity": {key: float(value) for key, value in kenyon_stats.items()},
        }
        if self.reward_eval_mode:
            payload["checkpoints"]["reward"] = str(self.reward_checkpoint)
        if reward_config:
            payload["reward_config"] = reward_config
        if sparse_overlap:
            payload["sparse_overlap"] = {
                key: float(value) if isinstance(value, (float, np.floating)) else int(value)
                for key, value in sparse_overlap.items()
            }
        output_path = self.outdir / f"{self.eval_feature}_evaluation_metrics.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info(f"Saved evaluation metrics to {output_path}")

    def eval(self):
        is_scanning_mode = getattr(self.loader.dataset, "mode", None) == DataMode.SCANNING_PATCH

        feats = []
        labs = []
        original_class_labels = []
        reward_logits = []
        reward_probabilities = []
        kc_active_counts = []
        kc_active_fractions = []
        batch_idx = 0

        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Extracting Features", unit="batch"):
                start_time = time()
                batch_idx += 1
                imgs, lbl = self._unpack_eval_batch(batch)
                outputs = self._extract_outputs(imgs, is_scanning_mode)

                if self.reward_eval_mode:
                    class_labels = lbl.detach().cpu().numpy()
                    reward_labels = self._reward_labels_from_class_labels(class_labels)
                    reward_logits.append(outputs["reward_logit"].detach().cpu().numpy())
                    reward_probabilities.append(outputs["reward_probability"].detach().cpu().numpy())
                    labs.append(reward_labels)
                    original_class_labels.append(class_labels)
                else:
                    feats.append(outputs[self.eval_feature].detach().cpu().numpy())
                    labs.append(lbl.detach().cpu().numpy())

                if "kc_active_counts" in outputs:
                    kc_active_counts.append(outputs["kc_active_counts"].detach().cpu().numpy())
                if "kc_active_fraction" in outputs:
                    kc_active_fractions.append(outputs["kc_active_fraction"].detach().cpu().numpy())

                self.logger.debug(
                    f"Processed batch {batch_idx}/{len(self.loader)} in {time() - start_time:.2f} seconds."
                )

        self.logger.info("")
        labs = np.concatenate(labs, axis=0)
        if self.reward_eval_mode:
            reward_logits = np.concatenate(reward_logits, axis=0)
            reward_probabilities = np.concatenate(reward_probabilities, axis=0)
            original_class_labels = np.concatenate(original_class_labels, axis=0)
        else:
            feats = np.concatenate(feats, axis=0)

        evaluator = avf.ModelEvaluator(self.model, self.logger, self.device, output_dir=self.outdir)
        metrics = {}

        self.logger.info("\n" + "=" * 60 + "\nRUNNING FULL MODEL EVALUATION\n" + "=" * 60 + "\n")
        if self.reward_eval_mode:
            self.logger.info("1. Reward Memory Evaluation\n" + "-" * 30)
            metrics = compute_reward_metrics(reward_logits, labs, threshold=self.reward_threshold)
            self.logger.info(f"   accuracy                 : {100.0 * float(metrics['accuracy']):6.2f}%")
            self.logger.info(f"   balanced_accuracy        : {100.0 * float(metrics['balanced_accuracy']):6.2f}%")
            self.logger.info(f"   auroc                    : {float(metrics['auroc']):6.4f}")
            self.logger.info(f"   average_precision        : {float(metrics['average_precision']):6.4f}")
            self.logger.info(f"   mean_positive_probability: {float(metrics['mean_positive_probability']):6.4f}")
            self.logger.info(f"   mean_negative_probability: {float(metrics['mean_negative_probability']):6.4f}")
        else:
            unique_labels, counts = np.unique(labs, return_counts=True)
            can_run_quantitative_eval = bool(len(unique_labels) > 1 and np.all(counts >= 2))
            self.logger.info("1. Quantitative Evaluation\n" + "-" * 30)
            if can_run_quantitative_eval:
                self.logger.info("   Running representation metrics on the selected feature space...")
                metrics = evaluator.evaluate_representations(feats, labs, self.dataset.groups)
                for metric, value in metrics.items():
                    unit = "%" if "accuracy" in metric else ""
                    self.logger.info(f"   {metric:<25}: {float(value):6.2f}{unit}")
            else:
                self.logger.info("   Skipping quantitative metrics because at least one class has fewer than 2 samples.")

        kenyon_stats = {}
        sparse_overlap_stats = {}
        if kc_active_counts and kc_active_fractions:
            active_counts = np.concatenate(kc_active_counts, axis=0)
            active_fractions = np.concatenate(kc_active_fractions, axis=0)
            kenyon_stats = {
                "mean_active_count": float(active_counts.mean()),
                "std_active_count": float(active_counts.std()),
                "mean_active_fraction": float(active_fractions.mean()),
                "std_active_fraction": float(active_fractions.std()),
            }
            self.logger.info("\nKenyon Activity Summary\n" + "-" * 30)
            self.logger.info(f"   mean_active_count       : {kenyon_stats['mean_active_count']:.2f}")
            self.logger.info(f"   std_active_count        : {kenyon_stats['std_active_count']:.2f}")
            self.logger.info(f"   mean_active_fraction    : {kenyon_stats['mean_active_fraction']:.4f}")
            self.logger.info(f"   std_active_fraction     : {kenyon_stats['std_active_fraction']:.4f}")

        class_names = getattr(self.dataset, "class_names", None)

        if not self.reward_eval_mode and self.eval_feature == "kenyon_code":
            self.logger.info("\n2. Sparse Kenyon Overlap Analysis\n" + "-" * 30)
            sparse_overlap_stats, overlap_details = evaluator.analyze_sparse_code_overlap(feats, labs)
            self.logger.info(f"   same_class_mean_overlap_count : {sparse_overlap_stats['same_class_mean_overlap_count']:.2f}")
            self.logger.info(f"   different_class_mean_overlap_count: {sparse_overlap_stats['different_class_mean_overlap_count']:.2f}")
            self.logger.info(f"   overlap_count_gap             : {sparse_overlap_stats['overlap_count_gap']:.2f}")
            self.logger.info(f"   same_class_mean_jaccard       : {sparse_overlap_stats['same_class_mean_jaccard']:.4f}")
            self.logger.info(f"   different_class_mean_jaccard  : {sparse_overlap_stats['different_class_mean_jaccard']:.4f}")
            self.logger.info(f"   jaccard_gap                   : {sparse_overlap_stats['jaccard_gap']:.4f}")

            fig_overlap_hist = evaluator.plot_sparse_overlap_histogram(
                overlap_details,
                save_path=f"{self.eval_feature}_overlap_histogram.png",
            )
            plt.close(fig_overlap_hist)

            fig_overlap_matrix = evaluator.plot_class_overlap_matrix(
                overlap_details["class_jaccard_matrix"],
                class_names=class_names,
                title="Mean Active KC Jaccard per Class Pair",
                save_path=f"{self.eval_feature}_class_jaccard_overlap.png",
            )
            plt.close(fig_overlap_matrix)

            self.logger.info("   ✓ KC overlap histogram saved.")
            self.logger.info("   ✓ KC class-pair Jaccard matrix saved.")

        reward_config = None
        if self.reward_eval_mode:
            self.logger.info("\n2. Reward Class Summary\n" + "-" * 30)
            fig_reward = plot_reward_by_class(
                class_names=class_names,
                class_labels=original_class_labels,
                reward_probabilities=reward_probabilities,
                rewarded_class_indices=self.rewarded_class_indices,
                save_path=f"{self.eval_feature}_reward_probability_by_class.png",
            )
            plt.close(fig_reward)
            self.logger.info("   ✓ Reward probability by class plot saved.")
            reward_config = {
                "reward_feature": str(self.reward_feature),
                "rewarded_class_indices": [int(idx) for idx in self.rewarded_class_indices],
                "rewarded_class_names": list(self.rewarded_class_names),
                "threshold": float(self.reward_threshold),
            }
        else:
            self.logger.info("\n3. Representation Space Analysis\n" + "-" * 30)

            fig_cos = evaluator.plot_class_similarity_matrix(
                feats,
                labs,
                class_names=class_names,
                save_path=f"{self.eval_feature}_class_cosine_similarity.png",
            )
            plt.close(fig_cos)

            fig_tsne = evaluator.plot_tsne(
                feats,
                labs,
                save_path=f"{self.eval_feature}_tsne_visualization.png",
            )
            plt.close(fig_tsne)

            self.logger.info("   ✓ Class similarity matrix saved.")
            self.logger.info("   ✓ t-SNE visualization saved.")

        self._save_metrics(metrics, kenyon_stats, sparse_overlap=sparse_overlap_stats, reward_config=reward_config)

        self.logger.info("\n" + "=" * 60 + "\nEVALUATION COMPLETE\n" + f"Results saved to: {self.outdir}\n" + "=" * 60 + "\n")
        return {
            "metrics": metrics,
            "kenyon_activity": kenyon_stats,
            "sparse_overlap": sparse_overlap_stats,
        }
