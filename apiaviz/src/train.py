# Imports
import json
import random
import secrets

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import apiaviz.src.functional as avf

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from apiaviz.dataset.datagen import (
    DenseSpatialPairDataset,
    DataMode,
    InsectVisionDataset,
    ProjectionTupleDataset,
    RewardLabelDataset,
    TinyImageNetPairDataset,
    TinyImageNetProjectionDataset,
    WILDSCENES_DATASET_NAMES,
)
from apiaviz.dataset.wildscenes import (
    WildScenesPosePairDataset,
    build_wildscenes_resized_cache,
    build_wildscenes_pose_pairs,
    collect_pose_pair_image_paths,
    load_wildscenes_frames,
    remap_pose_pairs_image_paths,
    split_wildscenes_frames,
)
from apiaviz.src.modules import VisionBackbone
from apiaviz.src.modules import RewardMemoryHead, VisionProjection, resolve_kc_sparsity_target
from apiaviz.src.projection_utils import infer_projection_config, resolve_projection_checkpoint
from apiaviz.src.projection_finetune import (
    ProjectionShiftHead,
    build_projection_snapshot,
    descriptor_similarity_summary,
    kenyon_overlap_triplet_loss,
    kenyon_ordering_loss,
    kenyon_sparsity_regularizer,
    kenyon_winner_usage_regularizer,
    load_balance_regularizer,
    plot_projection_history,
    plot_projection_snapshot,
    shift_regression_loss,
    supervised_contrastive_loss,
    write_projection_snapshot_json,
)
from apiaviz.src.spatial_finetune import (
    centroid_shift_loss,
    descriptor_place_matching_loss,
    dense_spatial_contrastive_loss,
    PoseRelationHead,
    relative_pose_regression_loss,
    resolve_pretrained_checkpoint,
)
from apiaviz.src.reward_memory import (
    REWARD_FEATURE_CHOICES,
    compute_reward_metrics,
    plot_reward_by_class,
    plot_reward_history,
    resolve_reward_feature_dim,
    resolve_rewarded_classes,
)

# Set multiprocessing start method to 'spawn' for compatibility on macOS
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


class TrainVision(nn.Module):
    # ────────── ctor ──────────
    def __init__(self, args, logger, outdir):
        super().__init__()

        for k in vars(args):
            setattr(self, k, getattr(args, k))

        self.train_stage = getattr(args, "train_stage", "backbone")
        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if self.train_stage == "lobula_plate":
            self.model_path = self.models_dir / f"{self.lobula_plate_model}.pth"
        elif self.train_stage == "projection":
            self.model_path = self.models_dir / f"{self.projection_model}.pth"
        elif self.train_stage == "reward_memory":
            self.model_path = self.models_dir / f"{self.reward_model}.pth"
        elif self.snn:
            self.model_path = self.models_dir / f"{self.snn_vision_model}.pth"
        else:
            self.model_path = self.models_dir / f"{self.vision_model}.pth"

        self.logger = logger
        self.outdir = Path(outdir)

        self.projection_feature_loss_weight = 1.0
        self.projection_shift_loss_weight = 0.5
        self.projection_kc_loss_weight = 0.35
        self.projection_class_feature_loss_weight = 0.25
        self.projection_class_kc_loss_weight = 0.0
        self.projection_kc_sparsity_loss_weight = 0.20
        self.projection_balance_loss_weight = 0.05
        self.projection_kc_overlap_loss_weight = 1.0
        self.projection_kc_usage_loss_weight = 0.10

        self.projection_near_min_shift = 1
        self.projection_near_max_shift = 4
        self.projection_far_min_shift = 8
        self.projection_far_max_shift = 18
        self.projection_kc_negative_overlap_target = 0.05
        self.projection_kc_overlap_margin = 0.05

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cpu":
            self.logger.info(
                "\n========================== WARNING ========================\n"
                ".       Training on CPU will be extremely slow. \n"
                ".     Please use a CUDA-enabled GPU or HPC if available.\n"
                "  =======================================================  \n"
            )

    def select_GB(self, chw: torch.Tensor) -> torch.Tensor:
        """Return the G & B channels from a 3-channel tensor."""
        return chw[1:3]

    def _projection_kc_config(self, kc_dim: int | None = None):
        effective_kc_dim = int(self.projection_kc_dim if kc_dim is None else kc_dim)
        effective_sparsity, target_active = resolve_kc_sparsity_target(
            effective_kc_dim,
            kc_sparsity=self.projection_kc_sparsity,
            kc_target_active=getattr(self, "projection_kc_target_active", 0),
        )
        return {
            "kc_dim": effective_kc_dim,
            "kc_sparsity": effective_sparsity,
            "kc_target_active": target_active,
        }

    def _projection_class_grouping_enabled(self):
        mode = str(getattr(self, "projection_class_grouping", "auto")).strip().lower()
        if mode == "auto":
            return str(self.training_dataset).strip().lower() == "tiny-imagenet"
        return mode == "on"

    def _projection_class_kc_weight(self, epoch_num: int | None):
        peak_weight = max(0.0, float(getattr(self, "projection_class_kc_loss_weight", 0.0)))
        if peak_weight <= 0.0:
            return 0.0
        if epoch_num is None or epoch_num <= 0:
            return peak_weight

        start_epoch = max(1, int(getattr(self, "projection_class_kc_start_epoch", 1)))
        ramp_epochs = max(0, int(getattr(self, "projection_class_kc_ramp_epochs", 0)))

        if epoch_num < start_epoch:
            return 0.0
        if ramp_epochs <= 0:
            return peak_weight

        ramp_progress = min(1.0, float(epoch_num - start_epoch + 1) / float(ramp_epochs))
        return peak_weight * ramp_progress

    def _prepare_projection_like_inputs(self, imgs: torch.Tensor):
        imgs = imgs.float().clamp(0.0, 1.0)
        input_size = int(getattr(self, "spatial_image_size", 64))
        if imgs.shape[-2:] != (input_size, input_size):
            imgs = F.interpolate(
                imgs,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        return imgs * 2.0 - 1.0

    def _resolve_reward_feature_tensor(self, outputs: dict):
        reward_feature = str(getattr(self, "reward_feature", "kenyon_code")).strip().lower()
        if reward_feature not in REWARD_FEATURE_CHOICES:
            raise ValueError(
                f"Unsupported reward feature '{reward_feature}'. Choices: {', '.join(REWARD_FEATURE_CHOICES)}"
            )
        if reward_feature not in outputs:
            raise KeyError(f"Projection outputs did not contain reward feature '{reward_feature}'.")
        return outputs[reward_feature]

    def _load_reward_frozen_backbone_projection(self):
        backbone_checkpoint = resolve_pretrained_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=self.backbone_checkpoint,
            model_name=self.lobula_plate_model,
        )
        projection_checkpoint = resolve_projection_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=getattr(self, "projection_checkpoint", ""),
            model_name=self.projection_model,
        )

        self.logger.info(f"Loading frozen reward-memory backbone from {backbone_checkpoint}")
        self.logger.info(f"Loading frozen reward-memory projection from {projection_checkpoint}")

        backbone = VisionBackbone().to(self.device)
        backbone_state_dict = torch.load(backbone_checkpoint, map_location=self.device, weights_only=True)
        backbone.load_state_dict(backbone_state_dict, strict=True)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()

        projection_state_dict = torch.load(projection_checkpoint, map_location=self.device, weights_only=True)
        inferred_kc_dim = int(projection_state_dict["kc_projection.weight"].shape[0])
        effective_kc_sparsity, target_active = resolve_kc_sparsity_target(
            inferred_kc_dim,
            kc_sparsity=self.projection_kc_sparsity,
            kc_target_active=getattr(self, "projection_kc_target_active", 0),
        )
        projection_kwargs = infer_projection_config(
            projection_state_dict,
            effective_kc_sparsity=effective_kc_sparsity,
            apl_feedback_strength=self.projection_apl_feedback_strength,
            apl_gain_adapt_rate=self.projection_apl_gain_adapt_rate,
            apl_threshold_lr=self.projection_apl_threshold_lr,
            apl_num_iters=self.projection_apl_num_iters,
        )
        projection = VisionProjection(**projection_kwargs).to(self.device)
        projection_load = projection.load_state_dict(projection_state_dict, strict=False)
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
                "Projection checkpoint did not match the reward-memory projection module. "
                f"Missing keys: {sorted(missing_keys)}. Unexpected keys: {sorted(unexpected_keys)}."
            )
        for parameter in projection.parameters():
            parameter.requires_grad = False
        projection.eval()

        self.reward_backbone_checkpoint = backbone_checkpoint
        self.reward_projection_checkpoint = projection_checkpoint
        self.reward_backbone_lobula_dim = int(backbone.lobula.embedding.out_features)
        self.reward_projection_kc_dim = int(projection_kwargs["kc_dim"])
        self.reward_projection_vpn_dim = int(projection_kwargs["vpn_dim"])
        self.reward_projection_target_active = int(target_active)
        return backbone, projection

    def _build_reward_dataset(self):
        base_dataset = InsectVisionDataset(
            root=str(self.dataset_dir),
            dataset=self.reward_dataset,
            mode=DataMode.STATIC_FULL,
            logger=self.logger,
            patch_size=self.patch_size,
            samples_per_image=1,
        )
        rewarded_indices, rewarded_names = resolve_rewarded_classes(
            getattr(base_dataset, "class_names", []),
            getattr(self, "rewarded_classes", ""),
        )
        reward_dataset = RewardLabelDataset(base_dataset, rewarded_indices)
        self.rewarded_class_indices = rewarded_indices
        self.rewarded_class_names = rewarded_names
        self.logger.info(
            "Reward mapping: "
            + ", ".join(f"{name}({idx})" for idx, name in zip(rewarded_indices, rewarded_names))
        )
        return reward_dataset

    def _split_reward_indices(self, reward_dataset: RewardLabelDataset):
        indices = np.arange(len(reward_dataset), dtype=int)
        class_labels = np.asarray(reward_dataset.class_labels, dtype=np.int64)
        val_split = float(getattr(self, "reward_val_split", 0.2))

        if not 0.0 < val_split < 1.0:
            raise ValueError(f"reward_val_split must be in (0, 1), got {val_split}")

        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=self.split_seed,
            stratify=class_labels,
        )
        return np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)

    def _reward_head_input_dim(self):
        return resolve_reward_feature_dim(
            getattr(self, "reward_feature", "kenyon_code"),
            lobula_dim=int(getattr(self, "reward_backbone_lobula_dim", 128)),
            vpn_dim=int(getattr(self, "reward_projection_vpn_dim", self.projection_vpn_dim)),
            kc_dim=int(getattr(self, "reward_projection_kc_dim", self.projection_kc_dim)),
        )

    def _build_reward_optimizer(self, reward_head: RewardMemoryHead):
        return torch.optim.AdamW(
            reward_head.parameters(),
            lr=self.lr,
            weight_decay=float(getattr(self, "reward_weight_decay", 1e-4)),
        )

    def _resolve_reward_pos_weight(self, reward_labels: np.ndarray):
        configured = float(getattr(self, "reward_pos_weight", 0.0))
        if configured > 0.0:
            return configured

        positive_count = float(np.sum(reward_labels > 0.5))
        negative_count = float(np.sum(reward_labels <= 0.5))
        if positive_count <= 0.0:
            return 1.0
        return max(1.0, negative_count / positive_count)

    def _reward_epoch_metrics(self, total_loss: float, processed: int, logits, reward_labels):
        metrics = compute_reward_metrics(logits, reward_labels, threshold=self.reward_threshold)
        metrics["loss"] = float(total_loss / max(1, processed))
        return metrics

    def _run_reward_epoch(
        self,
        backbone: VisionBackbone,
        projection: VisionProjection,
        reward_head: RewardMemoryHead,
        loader: DataLoader,
        loss_fn,
        optimizer=None,
        scaler=None,
    ):
        is_training = optimizer is not None
        reward_head.train(is_training)

        total_loss = 0.0
        processed = 0
        all_logits = []
        all_reward_labels = []
        all_class_labels = []

        if not is_training:
            reward_head.eval()

        iterator = tqdm(loader, desc="Reward epoch", unit="batch") if is_training else loader
        for batch in iterator:
            imgs = batch["input"].to(self.device, non_blocking=self.device.type == "cuda")
            reward_labels = batch["reward_label"].to(self.device, non_blocking=self.device.type == "cuda")
            class_labels = batch["class_label"].to(self.device, non_blocking=self.device.type == "cuda")

            prepared = self._prepare_projection_like_inputs(imgs)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with self._backbone_autocast_context():
                with torch.no_grad():
                    backbone_outputs = backbone(prepared, return_maps=True)
                    projection_outputs = projection(backbone_outputs)
                    combined_outputs = {
                        **backbone_outputs,
                        **projection_outputs,
                    }
                    reward_features = self._resolve_reward_feature_tensor(combined_outputs)
                reward_outputs = reward_head(reward_features)
                reward_logits = reward_outputs["reward_logit"]
                loss = loss_fn(reward_logits, reward_labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(reward_head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            batch_size = reward_labels.size(0)
            processed += batch_size
            total_loss += float(loss.item()) * batch_size
            all_logits.append(reward_logits.detach().cpu().numpy())
            all_reward_labels.append(reward_labels.detach().cpu().numpy())
            all_class_labels.append(class_labels.detach().cpu().numpy())

            if is_training:
                batch_metrics = compute_reward_metrics(
                    all_logits[-1],
                    all_reward_labels[-1],
                    threshold=self.reward_threshold,
                )
                iterator.set_postfix(
                    loss=f"{total_loss / max(1, processed):.4f}",
                    acc=f"{batch_metrics['accuracy']:.3f}",
                    bal_acc=f"{batch_metrics['balanced_accuracy']:.3f}",
                )

        logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros(0, dtype=np.float32)
        reward_labels = np.concatenate(all_reward_labels, axis=0) if all_reward_labels else np.zeros(0, dtype=np.float32)
        class_labels = np.concatenate(all_class_labels, axis=0) if all_class_labels else np.zeros(0, dtype=np.int64)
        metrics = self._reward_epoch_metrics(total_loss, processed, logits, reward_labels)
        return metrics, {
            "logits": logits,
            "reward_labels": reward_labels,
            "class_labels": class_labels,
            "probabilities": 1.0 / (1.0 + np.exp(-logits)),
        }

    def _save_reward_checkpoint(
        self,
        reward_head: RewardMemoryHead,
        epoch_num: int,
        metric_value: float,
        best_metric: float,
        loss_value: float,
        best_loss: float,
    ):
        if self.best_only:
            improved = metric_value > best_metric + 1e-6 or (
                abs(metric_value - best_metric) <= 1e-6 and loss_value < best_loss
            )
            if improved:
                torch.save(reward_head.state_dict(), str(self.model_path))
                return metric_value, loss_value, True
            return best_metric, best_loss, False

        epoch_path = self.models_dir / f"{self.reward_model}_Epoch{epoch_num}.pth"
        torch.save(reward_head.state_dict(), str(epoch_path))
        improved = metric_value > best_metric + 1e-6 or (
            abs(metric_value - best_metric) <= 1e-6 and loss_value < best_loss
        )
        if improved:
            return metric_value, loss_value, True
        return best_metric, best_loss, False

    def _set_seed(self) -> int:
        seed = secrets.randbits(32)
        self.run_seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = bool(getattr(self, "deterministic", False))
            torch.backends.cudnn.benchmark = not bool(getattr(self, "deterministic", False))
        self.logger.info(f"Random seed: {seed}")
        return seed

    def _loader_kwargs(self, batch_size: int, shuffle: bool, drop_last: bool):
        num_workers = max(0, int(getattr(self, "num_workers", 0)))
        kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def _confirm_overwrite(self):
        if not self.model_path.exists():
            return

        self.logger.info(f"A checkpoint already exists at {self.model_path}. Overwrite? ((y)/n)")
        ans = input().strip().lower()
        if ans == "n":
            self.logger.info("Exiting training.")
            raise SystemExit(0)
        if ans not in ("", "y"):
            self.logger.info("Invalid input. Exiting training.")
            raise SystemExit(0)
        self.logger.info("Continuing training and overwriting existing models and logs.")

    def _resolve_dataset_root(self) -> Path:
        dataset_key = str(self.training_dataset).strip().lower()
        dataset_base = Path(self.dataset_dir)
        dataset_root = Path(self.dataset_dir) / self.training_dataset
        if self.training_dataset == "tiny-imagenet" and (dataset_root / "train").exists():
            dataset_root = dataset_root / "train"
        elif dataset_key in WILDSCENES_DATASET_NAMES and not dataset_root.exists():
            dataset_root = dataset_base
        return dataset_root

    def _ensure_dataset_root_exists(self, dataset_root: Path):
        dataset_key = str(self.training_dataset).strip().lower()
        if dataset_key in WILDSCENES_DATASET_NAMES:
            resolved_root = DenseSpatialPairDataset.resolve_wildscenes2d_root(dataset_root)
            if resolved_root.exists() and resolved_root != dataset_root:
                self.logger.info(f"Resolved WildScenes root from {dataset_root} to {resolved_root}")
                return
            if resolved_root.exists():
                return
            raise FileNotFoundError(
                "WildScenes2D dataset could not be resolved from "
                f"{dataset_root}. Expected one of these root styles:\n"
                f"  - {dataset_root}/WildScenes2d\n"
                f"  - {dataset_root}/WildScenes/WildScenes2d\n"
                f"  - {dataset_root}/data/WildScenes/WildScenes2d\n"
                "For the CSIRO download bundle on this machine, the detected layout is under:\n"
                "  - /media/adam/vprdatasets/data/61541v003/data/WildScenes/WildScenes2d"
            )

        if not dataset_root.exists():
            raise FileNotFoundError(f"Training dataset not found at {dataset_root}")

    def _resolve_split_file(self, split_file: str, dataset_root: Path) -> Path | None:
        if split_file == "":
            return None

        candidate = Path(split_file)
        search_roots = [
            Path.cwd(),
            dataset_root,
            dataset_root.parent,
            Path(self.dataset_dir),
        ]
        for base in search_roots:
            resolved = candidate if candidate.is_absolute() else base / candidate
            if resolved.exists():
                return resolved

        raise FileNotFoundError(
            f"Spatial split file {split_file} was not found relative to the workspace or dataset roots."
        )

    def _discover_spatial_image_paths(self, dataset_root: Path, split_file: Path | None = None) -> list[Path]:
        return DenseSpatialPairDataset.discover_image_paths(
            dataset_root,
            dataset_name=self.training_dataset,
            split_file=split_file,
        )

    def _spatial_supervision_mode(self) -> str:
        mode = str(getattr(self, "spatial_supervision", "auto")).strip().lower()
        if mode == "auto":
            return "pose_pairs" if str(self.training_dataset).strip().lower() in WILDSCENES_DATASET_NAMES else "synthetic_shift"
        return mode

    def _is_pose_pair_supervision(self) -> bool:
        mode = self._spatial_supervision_mode()
        if mode == "pose_pairs" and str(self.training_dataset).strip().lower() not in WILDSCENES_DATASET_NAMES:
            raise ValueError("Pose-pair supervision is currently implemented for WildScenes datasets only.")
        return mode == "pose_pairs"

    def _backbone_autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _save_checkpoint(self, backbone: VisionBackbone, epoch: int, epoch_loss: float, best_loss: float) -> float:
        if self.best_only:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(backbone.state_dict(), str(self.model_path))
                self.logger.debug(f"\n[Epoch {epoch + 1}] ↳ New best loss {best_loss:.4f} → {self.model_path}")
            return best_loss

        if self.train_stage == "lobula_plate":
            new_path = self.models_dir / f"{self.lobula_plate_model}_Epoch{epoch + 1}.pth"
        elif self.snn:
            new_path = self.models_dir / f"{self.snn_vision_model}_Epoch{epoch + 1}.pth"
        else:
            new_path = self.models_dir / f"{self.vision_model}_Epoch{epoch + 1}.pth"

        torch.save(backbone.state_dict(), str(new_path))
        self.logger.debug(f"\n[Epoch {epoch + 1}] ↳ Model saved to {new_path}")
        return epoch_loss

    def train(self):
        if self.train_stage == "lobula_plate":
            return self.train_lobula_plate()
        if self.train_stage == "projection":
            return self.train_projection()
        if self.train_stage == "reward_memory":
            return self.train_reward_memory()
        return self.train_backbone()

    # ────────── backbone pretraining ──────────
    def train_backbone(self):
        self._confirm_overwrite()

        feature_aug = transforms.Compose([
            transforms.RandomResizedCrop(64, (0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, fill=128),
            transforms.ColorJitter(hue=.2, saturation=.3, brightness=.3, contrast=.3),
            transforms.GaussianBlur(3, sigma=(.1, 2.)),
            transforms.ToTensor(),
            transforms.Lambda(self.select_GB),
            transforms.Normalize([0.5, 0.5], [0.5, 0.5]),
        ])

        spatial_aug = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Lambda(self.select_GB),
            transforms.Normalize([0.5, 0.5], [0.5, 0.5]),
        ])

        ds_root = self._resolve_dataset_root()
        self._ensure_dataset_root_exists(ds_root)

        image_paths = DenseSpatialPairDataset.discover_image_paths(
            ds_root,
            dataset_name=self.training_dataset,
        )

        train_ds = TinyImageNetPairDataset(
            str(ds_root),
            feature_transform=feature_aug,
            spatial_transform=spatial_aug,
            image_paths=image_paths,
        )
        train_dl = DataLoader(train_ds, **self._loader_kwargs(self.batch_size, shuffle=True, drop_last=False))

        self._set_seed()

        backbone = VisionBackbone().to(self.device)
        backbone.train()

        opt = torch.optim.AdamW(backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        best_loss = float("inf")

        mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).to(self.device)

        for epoch in range(self.epochs):
            running, processed = 0.0, 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch")

            for v1, v2 in pbar:
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)
                opt.zero_grad(set_to_none=True)

                with self._backbone_autocast_context():
                    h1, _ = backbone(v1)
                    h2, _ = backbone(v2)
                    h1 = F.normalize(mlp(h1), dim=1)
                    h2 = F.normalize(mlp(h2), dim=1)
                    loss = avf.nt_xent(h1, h2)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                batch_size = v1.size(0)
                running += loss.item() * batch_size
                processed += batch_size
                pbar.set_postfix(
                    curr_loss=f"{running / processed:.4f}",
                    best_loss=f"{best_loss:.4f}",
                )

            sched.step()
            epoch_loss = running / processed

            with open(self.outdir / "training_log.txt", "a") as f:
                f.write(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}\n")

            best_loss = self._save_checkpoint(backbone, epoch, epoch_loss, best_loss)

        self.logger.info(f"\nTraining complete. Best loss {best_loss:.4f} → {self.model_path}")

    # ────────── lobula plate fine-tuning ──────────
    def _spatial_appearance_transform(self):
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.03, hue=0.01),
        ])

    def _build_spatial_dataset(
        self,
        image_paths,
        max_translation: int,
        min_translation: int,
        max_samples: int | None,
        deterministic: bool,
        seed: int,
        appearance_transform,
    ) -> DenseSpatialPairDataset:
        return DenseSpatialPairDataset(
            image_paths=image_paths,
            image_size=self.spatial_image_size,
            max_translation=max_translation,
            min_translation=min_translation,
            appearance_transform=appearance_transform,
            max_samples=max_samples,
            deterministic=deterministic,
            seed=seed,
        )

    def _split_spatial_images(self):
        ds_root = self._resolve_dataset_root()
        self._ensure_dataset_root_exists(ds_root)

        train_split_file = self._resolve_split_file(self.spatial_train_split_file, ds_root)
        val_split_file = self._resolve_split_file(self.spatial_val_split_file, ds_root)

        if train_split_file is not None:
            self.logger.info(f"Using spatial training split file: {train_split_file}")
        if val_split_file is not None:
            self.logger.info(f"Using spatial validation split file: {val_split_file}")

        if train_split_file is not None:
            train_paths = self._discover_spatial_image_paths(ds_root, split_file=train_split_file)
        else:
            train_paths = self._discover_spatial_image_paths(ds_root)

        if len(train_paths) == 0:
            raise RuntimeError(f"No spatial fine-tuning images found in {ds_root}")

        if val_split_file is not None:
            val_paths = self._discover_spatial_image_paths(ds_root, split_file=val_split_file)
            val_set = set(val_paths)
            train_paths = [path for path in train_paths if path not in val_set]
            return train_paths, val_paths

        split_rng = random.Random(self.split_seed)
        split_rng.shuffle(train_paths)

        if self.val_split <= 0.0 or len(train_paths) < 2:
            return train_paths, []

        val_count = max(1, int(len(train_paths) * self.val_split))
        val_count = min(val_count, len(train_paths) - 1)
        val_paths = train_paths[:val_count]
        train_paths = train_paths[val_count:]
        return train_paths, val_paths

    def _split_wildscenes_pose_pairs(self):
        ds_root = self._resolve_dataset_root()
        self._ensure_dataset_root_exists(ds_root)

        train_split_file = self._resolve_split_file(self.spatial_train_split_file, ds_root)
        val_split_file = self._resolve_split_file(self.spatial_val_split_file, ds_root)

        if train_split_file is not None:
            self.logger.info(f"Using spatial training split file: {train_split_file}")
        if val_split_file is not None:
            self.logger.info(f"Using spatial validation split file: {val_split_file}")

        resolved_root = None
        if train_split_file is not None:
            train_paths = set(self._discover_spatial_image_paths(ds_root, split_file=train_split_file))
            resolved_root, train_frames = load_wildscenes_frames(ds_root, allowed_image_paths=train_paths)
        else:
            resolved_root, all_frames = load_wildscenes_frames(ds_root)
            train_frames, auto_val_frames = split_wildscenes_frames(all_frames, self.val_split, self.split_seed)

        if val_split_file is not None:
            val_paths = set(self._discover_spatial_image_paths(ds_root, split_file=val_split_file))
            _, val_frames = load_wildscenes_frames(ds_root, allowed_image_paths=val_paths)
        elif train_split_file is not None:
            if self.val_split <= 0.0:
                val_frames = {}
            else:
                _, train_frames = load_wildscenes_frames(ds_root, allowed_image_paths=train_paths)
                train_frames, val_frames = split_wildscenes_frames(train_frames, self.val_split, self.split_seed)
        else:
            val_frames = auto_val_frames

        train_pairs, train_stats = build_wildscenes_pose_pairs(
            train_frames,
            match_radius_m=self.wildscenes_match_radius_m,
            yaw_threshold_deg=self.wildscenes_yaw_threshold_deg,
            max_candidates_per_anchor=self.wildscenes_max_candidates,
        )
        val_pairs = []
        val_stats = {}
        if val_frames:
            val_pairs, val_stats = build_wildscenes_pose_pairs(
                val_frames,
                match_radius_m=self.wildscenes_match_radius_m,
                yaw_threshold_deg=self.wildscenes_yaw_threshold_deg,
                max_candidates_per_anchor=self.wildscenes_max_candidates,
            )

        if len(train_pairs) == 0:
            raise RuntimeError(
                "No WildScenes pose pairs were mined. "
                "Try increasing --wildscenes_match_radius_m or --wildscenes_yaw_threshold_deg."
            )

        train_sequences = sum(len(frames) > 0 for frames in train_frames.values())
        val_sequences = sum(len(frames) > 0 for frames in val_frames.values())
        self.logger.info(
            "WildScenes pose split: "
            f"{len(train_pairs)} training anchors across {train_sequences} traverses, "
            f"{len(val_pairs)} validation anchors across {val_sequences} traverses"
        )
        self.logger.info(
            "WildScenes pose mining: "
            f"train avg candidates={train_stats['avg_candidates_per_anchor']:.2f}, "
            f"train avg distance={train_stats['avg_distance_m']:.2f}m, "
            f"train avg yaw={train_stats['avg_yaw_deg']:.2f}deg"
        )
        if val_stats:
            self.logger.info(
                "WildScenes validation mining: "
                f"val avg candidates={val_stats['avg_candidates_per_anchor']:.2f}, "
                f"val avg distance={val_stats['avg_distance_m']:.2f}m, "
                f"val avg yaw={val_stats['avg_yaw_deg']:.2f}deg"
            )

        return resolved_root, train_pairs, val_pairs, train_stats, val_stats

    def _curriculum_shift_range(self, epoch: int) -> tuple[int, int]:
        if self.shift_curriculum_warmup_epochs <= 0 and self.shift_curriculum_mid_epochs <= 0:
            return self.spatial_max_shift, min(self.min_spatial_shift, self.spatial_max_shift)

        epoch_num = epoch + 1
        warmup_max = min(self.spatial_max_shift, self.shift_curriculum_warmup_max)
        mid_max = min(self.spatial_max_shift, self.shift_curriculum_mid_max)

        if epoch_num <= self.shift_curriculum_warmup_epochs:
            return warmup_max, 0
        if epoch_num <= self.shift_curriculum_mid_epochs:
            return mid_max, min(1, mid_max)
        return self.spatial_max_shift, min(self.min_spatial_shift, self.spatial_max_shift)

    def _shift_loss_weight_for_epoch(self, epoch: int) -> float:
        epoch_num = epoch + 1
        if self.shift_loss_final_epoch > 0 and epoch_num >= self.shift_loss_final_epoch:
            return self.shift_loss_final_weight
        if self.shift_loss_mid_epoch > 0 and epoch_num >= self.shift_loss_mid_epoch:
            return self.shift_loss_mid_weight
        return self.shift_loss_weight

    def _is_top_lobula_param(self, name: str) -> bool:
        return (
            name.startswith("lobula.integrate.")
            or name.startswith("lobula.norm.")
            or name.startswith("lobula.mixer.refine.")
            or name.startswith("lobula.mixer.norm.")
        )

    def _is_lobula_place_head_param(self, name: str) -> bool:
        return name.startswith("lobula.embedding.") or name.startswith("lobula.gem_pool.")

    def _configure_lobula_trainability(self, backbone: VisionBackbone, unfreeze_top_lobula: bool):
        for name, parameter in backbone.named_parameters():
            if name.startswith("lobula_plate."):
                parameter.requires_grad = True
            elif self._is_top_lobula_param(name):
                parameter.requires_grad = unfreeze_top_lobula
            else:
                parameter.requires_grad = False

        backbone.train()
        backbone.R1_R6.eval()
        backbone.R8.eval()
        backbone.lamina.eval()
        backbone.medulla.eval()
        backbone.lobula.train(mode=unfreeze_top_lobula)
        backbone.lobula_plate.train()

    def _configure_pose_pair_trainability(self, backbone: VisionBackbone, unfreeze_top_lobula: bool):
        for name, parameter in backbone.named_parameters():
            if name.startswith("lobula_plate."):
                parameter.requires_grad = True
            elif self._is_lobula_place_head_param(name):
                parameter.requires_grad = True
            elif self._is_top_lobula_param(name):
                parameter.requires_grad = unfreeze_top_lobula
            else:
                parameter.requires_grad = False

        backbone.train()
        backbone.R1_R6.eval()
        backbone.R8.eval()
        backbone.lamina.eval()
        backbone.medulla.eval()
        backbone.lobula.train()
        backbone.lobula_plate.train()

    def _trainable_parameters(self, backbone: VisionBackbone):
        return [parameter for parameter in backbone.parameters() if parameter.requires_grad]

    def _build_lobula_plate_optimizer(self, backbone: VisionBackbone):
        plate_params = []
        top_lobula_params = []

        for name, parameter in backbone.named_parameters():
            if name.startswith("lobula_plate."):
                plate_params.append(parameter)
            elif self._is_top_lobula_param(name):
                top_lobula_params.append(parameter)

        return torch.optim.AdamW(
            [
                {"params": plate_params, "lr": self.lr},
                {"params": top_lobula_params, "lr": self.lr * self.lobula_lr_scale},
            ],
            weight_decay=1e-4,
        )

    def _build_pose_pair_optimizer(self, backbone: VisionBackbone, pose_head: PoseRelationHead):
        plate_params = []
        place_head_params = []
        top_lobula_params = []

        for name, parameter in backbone.named_parameters():
            if name.startswith("lobula_plate."):
                plate_params.append(parameter)
            elif self._is_lobula_place_head_param(name):
                place_head_params.append(parameter)
            elif self._is_top_lobula_param(name):
                top_lobula_params.append(parameter)

        param_groups = []
        if plate_params:
            param_groups.append({"params": plate_params, "lr": self.lr})
        if place_head_params:
            param_groups.append({"params": place_head_params, "lr": self.lr})
        if top_lobula_params:
            param_groups.append({"params": top_lobula_params, "lr": self.lr * self.lobula_lr_scale})
        param_groups.append({"params": pose_head.parameters(), "lr": self.lr})

        return torch.optim.AdamW(param_groups, weight_decay=1e-4)

    def _empty_spatial_metrics(self):
        return {
            "loss": 0.0,
            "dense_loss": 0.0,
            "shift_loss": 0.0,
            "dense_top1": 0.0,
            "positive_cosine": 0.0,
            "shift_mae_px": 0.0,
            "overlap_ratio": 0.0,
            "pairs": 0.0,
        }

    def _empty_pose_metrics(self):
        return {
            "loss": 0.0,
            "place_loss": 0.0,
            "pose_loss": 0.0,
            "translation_loss": 0.0,
            "yaw_loss": 0.0,
            "place_top1": 0.0,
            "positive_cosine": 0.0,
            "translation_mae_m": 0.0,
            "yaw_mae_deg": 0.0,
            "distance_m": 0.0,
            "pairs": 0.0,
        }

    def _prepare_wildscenes_resized_cache(
        self,
        resolved_root: Path,
        train_pairs,
        val_pairs,
    ):
        if bool(getattr(self, "wildscenes_disable_resized_cache", False)):
            self.logger.info("WildScenes resized cache disabled; decoding original source frames on the fly.")
            return train_pairs, val_pairs, None

        cache_root = Path(self.wildscenes_cache_dir)
        image_paths = collect_pose_pair_image_paths(train_pairs + val_pairs)
        self.logger.info(
            f"Preparing WildScenes resized cache for {len(image_paths)} unique frames at {cache_root}"
        )
        path_map, cache_stats = build_wildscenes_resized_cache(
            image_paths=image_paths,
            source_root=resolved_root,
            cache_root=cache_root,
            image_size=self.spatial_image_size,
            overwrite=bool(getattr(self, "wildscenes_cache_overwrite", False)),
        )
        train_pairs = remap_pose_pairs_image_paths(train_pairs, path_map)
        val_pairs = remap_pose_pairs_image_paths(val_pairs, path_map)
        self.logger.info(
            "WildScenes resized cache ready: "
            f"created={int(cache_stats['created_images'])}, "
            f"reused={int(cache_stats['reused_images'])}, "
            f"root={cache_stats['cache_root']}"
        )
        return train_pairs, val_pairs, cache_stats

    def _finalize_spatial_metrics(self, running, processed: int, num_batches: int):
        return {
            "loss": running["loss"] / processed,
            "dense_loss": running["dense_loss"] / processed,
            "shift_loss": running["shift_loss"] / processed,
            "dense_top1": running["dense_top1"] / processed,
            "positive_cosine": running["positive_cosine"] / processed,
            "shift_mae_px": running["shift_mae_px"] / processed,
            "overlap_ratio": running["overlap_ratio"] / processed,
            "pairs_per_batch": running["pairs"] / max(1, num_batches),
        }

    def _finalize_pose_metrics(self, running, processed: int, num_batches: int):
        return {
            "loss": running["loss"] / processed,
            "place_loss": running["place_loss"] / processed,
            "pose_loss": running["pose_loss"] / processed,
            "translation_loss": running["translation_loss"] / processed,
            "yaw_loss": running["yaw_loss"] / processed,
            "place_top1": running["place_top1"] / processed,
            "positive_cosine": running["positive_cosine"] / processed,
            "translation_mae_m": running["translation_mae_m"] / processed,
            "yaw_mae_deg": running["yaw_mae_deg"] / processed,
            "distance_m": running["distance_m"] / processed,
            "pairs_per_batch": running["pairs"] / max(1, num_batches),
        }

    def _evaluate_spatial_loader(
        self,
        backbone: VisionBackbone,
        loader: DataLoader,
        shift_weight: float,
        detach_lobula_for_plate: bool,
    ):
        running = self._empty_spatial_metrics()
        processed = 0

        backbone.eval()
        with torch.no_grad():
            for batch in loader:
                anchor = batch["anchor"].to(self.device, non_blocking=self.device.type == "cuda")
                positive = batch["positive"].to(self.device, non_blocking=self.device.type == "cuda")
                shift = batch["shift"].to(self.device, non_blocking=self.device.type == "cuda")

                with self._backbone_autocast_context():
                    _, anchor_plate = backbone(anchor, detach_lobula_for_plate=detach_lobula_for_plate)
                    _, positive_plate = backbone(positive, detach_lobula_for_plate=detach_lobula_for_plate)

                    dense_metrics = dense_spatial_contrastive_loss(
                        anchor_plate,
                        positive_plate,
                        shift,
                        temperature=self.dense_temperature,
                        samples_per_image=self.dense_samples,
                    )
                    shift_metrics = centroid_shift_loss(anchor_plate, positive_plate, shift)
                    total_loss = (
                        self.dense_loss_weight * dense_metrics["loss"]
                        + shift_weight * shift_metrics["loss"]
                    )

                batch_size = anchor.size(0)
                processed += batch_size
                running["loss"] += total_loss.item() * batch_size
                running["dense_loss"] += dense_metrics["loss"].item() * batch_size
                running["shift_loss"] += shift_metrics["loss"].item() * batch_size
                running["dense_top1"] += dense_metrics["top1"].item() * batch_size
                running["positive_cosine"] += dense_metrics["positive_cosine"].item() * batch_size
                running["shift_mae_px"] += shift_metrics["mae_px"].item() * batch_size
                running["overlap_ratio"] += dense_metrics["overlap_ratio"].item() * batch_size
                running["pairs"] += dense_metrics["pairs"]

        return self._finalize_spatial_metrics(running, processed, len(loader))

    def _evaluate_pose_loader(
        self,
        backbone: VisionBackbone,
        pose_head: PoseRelationHead,
        loader: DataLoader,
        detach_lobula_for_plate: bool,
    ):
        running = self._empty_pose_metrics()
        processed = 0

        backbone.eval()
        pose_head.eval()
        with torch.no_grad():
            for batch in loader:
                anchor = batch["anchor"].to(self.device, non_blocking=self.device.type == "cuda")
                positive = batch["positive"].to(self.device, non_blocking=self.device.type == "cuda")
                relative_translation = batch["relative_translation"].to(self.device, non_blocking=self.device.type == "cuda")
                relative_yaw = batch["relative_yaw"].to(self.device, non_blocking=self.device.type == "cuda")
                distance_m = batch["distance_m"].to(self.device, non_blocking=self.device.type == "cuda")

                with self._backbone_autocast_context():
                    anchor_outputs = backbone(anchor, detach_lobula_for_plate=detach_lobula_for_plate, return_maps=True)
                    positive_outputs = backbone(positive, detach_lobula_for_plate=detach_lobula_for_plate, return_maps=True)
                    anchor_lobula = anchor_outputs["lobula"]
                    positive_lobula = positive_outputs["lobula"]
                    anchor_plate = anchor_outputs["lobula_plate"]
                    positive_plate = positive_outputs["lobula_plate"]

                    place_metrics = descriptor_place_matching_loss(
                        anchor_lobula,
                        positive_lobula,
                        temperature=self.dense_temperature,
                    )
                    pose_predictions = pose_head(anchor_plate, positive_plate)
                    pose_metrics = relative_pose_regression_loss(
                        pose_predictions,
                        target_translation_m=relative_translation,
                        target_yaw_rad=relative_yaw,
                        translation_scale_m=self.wildscenes_match_radius_m,
                    )
                    total_loss = (
                        self.place_loss_weight * place_metrics["loss"]
                        + self.pose_loss_weight * pose_metrics["loss"]
                    )

                batch_size = anchor.size(0)
                processed += batch_size
                running["loss"] += total_loss.item() * batch_size
                running["place_loss"] += place_metrics["loss"].item() * batch_size
                running["pose_loss"] += pose_metrics["loss"].item() * batch_size
                running["translation_loss"] += pose_metrics["translation_loss"].item() * batch_size
                running["yaw_loss"] += pose_metrics["yaw_loss"].item() * batch_size
                running["place_top1"] += place_metrics["top1"].item() * batch_size
                running["positive_cosine"] += place_metrics["positive_cosine"].item() * batch_size
                running["translation_mae_m"] += pose_metrics["translation_mae_m"].item() * batch_size
                running["yaw_mae_deg"] += pose_metrics["yaw_mae_deg"].item() * batch_size
                running["distance_m"] += distance_m.mean().item() * batch_size
                running["pairs"] += batch_size

        return self._finalize_pose_metrics(running, processed, len(loader))

    def _split_projection_images(self):
        train_paths, val_paths = self._split_spatial_images()
        if val_paths:
            return train_paths, val_paths

        if len(train_paths) < 4:
            return train_paths, []

        split_rng = random.Random(self.split_seed)
        split_rng.shuffle(train_paths)
        val_count = max(1, int(len(train_paths) * 0.1))
        val_count = min(val_count, len(train_paths) - 1)
        val_paths = train_paths[:val_count]
        train_paths = train_paths[val_count:]
        self.logger.info(
            "Projection training did not receive an explicit validation split, "
            f"so a 10% holdout was created automatically ({len(val_paths)} images)."
        )
        return train_paths, val_paths

    def _build_projection_dataset(
        self,
        image_paths,
        max_samples,
        deterministic,
        seed,
        appearance_transform=None,  # kept for interface compatibility
    ):
        return TinyImageNetProjectionDataset(
            image_paths=image_paths,
            select_GB=self.select_GB,
            deterministic=deterministic,
            seed=seed,
            image_size=64,
            near_min_shift=self.projection_near_min_shift,
            near_max_shift=self.projection_near_max_shift,
            far_min_shift=self.projection_far_min_shift,
            far_max_shift=self.projection_far_max_shift,
            max_samples=max_samples,
        )

    def _configure_projection_backbone(self, backbone: VisionBackbone):
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        backbone.eval()

    def _set_projection_backbone_trainability(self, backbone, epoch_num: int):
        """
        Warm up projection-only, then partially unfreeze the late backbone.
        """
        warmup_epochs = 3

        for p in backbone.parameters():
            p.requires_grad = False

        if epoch_num > warmup_epochs:
            for module in [backbone.lobula, backbone.lobula_plate]:
                for p in module.parameters():
                    p.requires_grad = True

    def _build_projection_optimizer(self, backbone, projection, shift_head):
        proj_lr = self.lr
        backbone_lr = self.lr * 0.1

        return torch.optim.AdamW(
            [
                {
                    "params": list(projection.parameters()),
                    "lr": proj_lr,
                    "weight_decay": 1e-4,
                },
                {
                    "params": list(shift_head.parameters()),
                    "lr": proj_lr,
                    "weight_decay": 1e-4,
                },
                {
                    "params": list(backbone.lobula.parameters()),
                    "lr": backbone_lr,
                    "weight_decay": 1e-4,
                },
                {
                    "params": list(backbone.lobula_plate.parameters()),
                    "lr": backbone_lr,
                    "weight_decay": 1e-4,
                },
            ]
        )

    def _empty_projection_metrics(self):
        return {
            "loss": 0.0,
            "feature_loss": 0.0,
            "shift_loss": 0.0,
            "kc_loss": 0.0,
            "class_feature_loss": 0.0,
            "class_kc_loss": 0.0,
            "kc_sparsity_loss": 0.0,
            "balance_loss": 0.0,
            "kc_overlap_loss": 0.0,
            "kc_usage_loss": 0.0,
            "feature_top1": 0.0,
            "shift_mae_px": 0.0,
            "kc_ordering_acc": 0.0,
            "kc_overlap_ordering_acc": 0.0,
            "feature_ordering_acc": 0.0,
            "spatial_ordering_acc": 0.0,
            "conjunctive_ordering_acc": 0.0,
            "kc_active_fraction": 0.0,
            "kc_active_count": 0.0,
            "feature_near_similarity": 0.0,
            "feature_far_similarity": 0.0,
            "feature_negative_similarity": 0.0,
            "feature_class_similarity": 0.0,
            "spatial_near_similarity": 0.0,
            "spatial_far_similarity": 0.0,
            "spatial_negative_similarity": 0.0,
            "conjunctive_near_similarity": 0.0,
            "conjunctive_far_similarity": 0.0,
            "conjunctive_negative_similarity": 0.0,
            "kc_near_similarity": 0.0,
            "kc_far_similarity": 0.0,
            "kc_negative_similarity": 0.0,
            "kc_class_similarity": 0.0,
            "kc_near_overlap": 0.0,
            "kc_far_overlap": 0.0,
            "kc_negative_overlap": 0.0,
            "kc_negative_overlap_excess": 0.0,
            "kc_usage_cv2": 0.0,
            "kc_usage_effective_fraction": 0.0,
            "kc_usage_max_share": 0.0,
            "near_overlap_ratio": 0.0,
            "far_overlap_ratio": 0.0,
        }

    def _finalize_projection_metrics(self, running, processed: int):
        return {key: value / processed for key, value in running.items()}

    def _projection_checkpoint_paths(self):
        return (
            self.model_path,
            self.models_dir / f"{self.projection_model}_ShiftHead.pth",
        )

    def _projection_forward(
        self,
        backbone: VisionBackbone,
        projection: VisionProjection,
        shift_head: ProjectionShiftHead,
        batch,
        class_kc_loss_weight: float | None = None,
    ):
        anchor = batch["anchor"].to(self.device, non_blocking=self.device.type == "cuda")
        near_positive = batch["near_positive"].to(self.device, non_blocking=self.device.type == "cuda")
        far_positive = batch["far_positive"].to(self.device, non_blocking=self.device.type == "cuda")
        negative = batch["negative"].to(self.device, non_blocking=self.device.type == "cuda")
        use_class_grouping = self._projection_class_grouping_enabled()
        class_positive = None
        class_labels = None
        if use_class_grouping:
            class_positive = batch["class_positive"].to(self.device, non_blocking=self.device.type == "cuda")
            class_labels = batch["class_label"].to(self.device, non_blocking=self.device.type == "cuda")
        near_shift = batch["near_shift"].to(self.device, non_blocking=self.device.type == "cuda").float()
        far_shift = batch["far_shift"].to(self.device, non_blocking=self.device.type == "cuda").float()

        backbone_trainable = any(p.requires_grad for p in backbone.parameters())

        if backbone_trainable:
            anchor_outputs = backbone(anchor, return_maps=True)
            near_outputs = backbone(near_positive, return_maps=True)
            far_outputs = backbone(far_positive, return_maps=True)
            negative_outputs = backbone(negative, return_maps=True)
            class_positive_outputs = backbone(class_positive, return_maps=True) if use_class_grouping else None
        else:
            with torch.no_grad():
                anchor_outputs = backbone(anchor, return_maps=True)
                near_outputs = backbone(near_positive, return_maps=True)
                far_outputs = backbone(far_positive, return_maps=True)
                negative_outputs = backbone(negative, return_maps=True)
                class_positive_outputs = backbone(class_positive, return_maps=True) if use_class_grouping else None

        anchor_projection = projection(anchor_outputs)
        near_projection = projection(near_outputs)
        far_projection = projection(far_outputs)
        negative_projection = projection(negative_outputs)
        class_positive_projection = projection(class_positive_outputs) if use_class_grouping else None

        feature_near_metrics = descriptor_place_matching_loss(
            anchor_projection["feature_vpn"],
            near_projection["feature_vpn"],
            temperature=self.dense_temperature,
        )
        feature_far_metrics = descriptor_place_matching_loss(
            anchor_projection["feature_vpn"],
            far_projection["feature_vpn"],
            temperature=self.dense_temperature,
        )
        feature_loss = 0.5 * (feature_near_metrics["loss"] + feature_far_metrics["loss"])
        feature_top1 = 0.5 * (feature_near_metrics["top1"] + feature_far_metrics["top1"])
        zero = feature_loss * 0.0

        if use_class_grouping:
            supcon_labels = torch.cat([class_labels, class_labels], dim=0)
            class_feature_metrics = supervised_contrastive_loss(
                torch.cat(
                    [
                        anchor_projection["feature_vpn"],
                        class_positive_projection["feature_vpn"],
                    ],
                    dim=0,
                ),
                supcon_labels,
                temperature=self.dense_temperature,
            )
            class_kc_metrics = {"loss": zero, "top1": zero.detach()}
            feature_class_similarity = F.cosine_similarity(
                F.normalize(anchor_projection["feature_vpn"], dim=1),
                F.normalize(class_positive_projection["feature_vpn"], dim=1),
                dim=1,
            ).mean()
            kc_class_similarity = zero.detach()
        else:
            class_feature_metrics = {"loss": zero, "top1": zero.detach()}
            class_kc_metrics = {"loss": zero, "top1": zero.detach()}
            feature_class_similarity = zero.detach()
            kc_class_similarity = zero.detach()

        effective_class_kc_loss_weight = (
            float(self.projection_class_kc_loss_weight)
            if class_kc_loss_weight is None
            else float(class_kc_loss_weight)
        )
        near_shift_metrics = shift_regression_loss(
            shift_head(anchor_projection["spatial_vpn"], near_projection["spatial_vpn"]),
            near_shift,
            shift_scale_px=self.projection_far_max_shift,
        )
        far_shift_metrics = shift_regression_loss(
            shift_head(anchor_projection["spatial_vpn"], far_projection["spatial_vpn"]),
            far_shift,
            shift_scale_px=self.projection_far_max_shift,
        )
        shift_loss = 0.5 * (near_shift_metrics["loss"] + far_shift_metrics["loss"])
        shift_mae_px = 0.5 * (near_shift_metrics["mae_px"] + far_shift_metrics["mae_px"])

        feature_summary = descriptor_similarity_summary(
            anchor_projection["feature_vpn"],
            near_projection["feature_vpn"],
            far_projection["feature_vpn"],
            negative_projection["feature_vpn"],
        )
        spatial_summary = descriptor_similarity_summary(
            anchor_projection["spatial_vpn"],
            near_projection["spatial_vpn"],
            far_projection["spatial_vpn"],
            negative_projection["spatial_vpn"],
        )
        conjunctive_summary = descriptor_similarity_summary(
            anchor_projection["conjunctive_vpn"],
            near_projection["conjunctive_vpn"],
            far_projection["conjunctive_vpn"],
            negative_projection["conjunctive_vpn"],
        )
        kc_metrics = kenyon_ordering_loss(
            anchor_projection["kenyon_code"],
            near_projection["kenyon_code"],
            far_projection["kenyon_code"],
            negative_projection["kenyon_code"],
            pose_margin=self.projection_pose_margin,
            negative_margin=self.projection_negative_margin,
        )
        balance_metrics = load_balance_regularizer(
            torch.cat(
                [
                    anchor_projection["kenyon_drive"],
                    near_projection["kenyon_drive"],
                    far_projection["kenyon_drive"],
                    negative_projection["kenyon_drive"],
                ],
                dim=0,
            )
        )
        near_target = 0.20 + 0.45 * batch["near_overlap_ratio"].to(self.device)
        far_target = 0.08 + 0.25 * batch["far_overlap_ratio"].to(self.device)

        overlap_metrics = kenyon_overlap_triplet_loss(
            anchor_projection["kenyon_code"],
            near_projection["kenyon_code"],
            far_projection["kenyon_code"],
            negative_projection["kenyon_code"],
            near_target=near_target,
            far_target=far_target,
            negative_target=getattr(self, "projection_kc_negative_overlap_target", 0.05),
            ordering_margin=getattr(self, "projection_kc_overlap_margin", 0.05),
        )
        usage_inputs = [
            anchor_projection["kenyon_code"],
            near_projection["kenyon_code"],
            far_projection["kenyon_code"],
            negative_projection["kenyon_code"],
        ]
        if use_class_grouping:
            usage_inputs.append(class_positive_projection["kenyon_code"])
        usage_metrics = kenyon_winner_usage_regularizer(*usage_inputs)
        kc_sparsity_inputs = [
            anchor_projection["kc_active_counts"].float(),
            near_projection["kc_active_counts"].float(),
            far_projection["kc_active_counts"].float(),
            negative_projection["kc_active_counts"].float(),
        ]
        if use_class_grouping:
            kc_sparsity_inputs.append(class_positive_projection["kc_active_counts"].float())
        kc_sparsity_metrics = kenyon_sparsity_regularizer(
            torch.cat(kc_sparsity_inputs, dim=0),
            target_active_count=getattr(self, "projection_target_kc_active", self._projection_kc_config()["kc_target_active"]),
        )

        loss = (
            self.projection_feature_loss_weight * feature_loss
            + self.projection_shift_loss_weight * shift_loss
            + self.projection_kc_loss_weight * kc_metrics["loss"]
            + self.projection_class_feature_loss_weight * class_feature_metrics["loss"]
            + effective_class_kc_loss_weight * class_kc_metrics["loss"]
            + self.projection_kc_sparsity_loss_weight * kc_sparsity_metrics["loss"]
            + self.projection_balance_loss_weight * balance_metrics["loss"]
            + getattr(self, "projection_kc_overlap_loss_weight", 0.0) * overlap_metrics["loss"]
            + getattr(self, "projection_kc_usage_loss_weight", 0.0) * usage_metrics["loss"]
        )

        kc_active_fraction = torch.cat(
            [
                anchor_projection["kc_active_fraction"],
                near_projection["kc_active_fraction"],
                far_projection["kc_active_fraction"],
                negative_projection["kc_active_fraction"],
            ],
            dim=0,
        ).mean()
        kc_active_count = torch.cat(
            [
                anchor_projection["kc_active_counts"].float(),
                near_projection["kc_active_counts"].float(),
                far_projection["kc_active_counts"].float(),
                negative_projection["kc_active_counts"].float(),
            ],
            dim=0,
        ).mean()

        metrics = {
            "loss": loss.detach(),
            "feature_loss": feature_loss.detach(),
            "shift_loss": shift_loss.detach(),
            "kc_loss": kc_metrics["loss"].detach(),
            "class_feature_loss": class_feature_metrics["loss"].detach(),
            "class_kc_loss": class_kc_metrics["loss"].detach(),
            "kc_sparsity_loss": kc_sparsity_metrics["loss"].detach(),
            "balance_loss": balance_metrics["loss"].detach(),
            "kc_overlap_loss": overlap_metrics["loss"].detach(),
            "kc_usage_loss": usage_metrics["loss"].detach(),
            "feature_top1": feature_top1.detach(),
            "shift_mae_px": shift_mae_px.detach(),
            "kc_ordering_acc": kc_metrics["ordering_acc"],
            "kc_overlap_ordering_acc": overlap_metrics["ordering_acc"],
            "feature_ordering_acc": feature_summary["ordering_acc"],
            "spatial_ordering_acc": spatial_summary["ordering_acc"],
            "conjunctive_ordering_acc": conjunctive_summary["ordering_acc"],
            "kc_active_fraction": kc_active_fraction.detach(),
            "kc_active_count": kc_active_count.detach(),
            "feature_near_similarity": feature_summary["near_mean"],
            "feature_far_similarity": feature_summary["far_mean"],
            "feature_negative_similarity": feature_summary["negative_mean"],
            "feature_class_similarity": feature_class_similarity.detach(),
            "spatial_near_similarity": spatial_summary["near_mean"],
            "spatial_far_similarity": spatial_summary["far_mean"],
            "spatial_negative_similarity": spatial_summary["negative_mean"],
            "conjunctive_near_similarity": conjunctive_summary["near_mean"],
            "conjunctive_far_similarity": conjunctive_summary["far_mean"],
            "conjunctive_negative_similarity": conjunctive_summary["negative_mean"],
            "kc_near_similarity": kc_metrics["sim_near"].mean(),
            "kc_far_similarity": kc_metrics["sim_far"].mean(),
            "kc_negative_similarity": kc_metrics["sim_negative"].mean(),
            "kc_class_similarity": kc_class_similarity.detach(),
            "kc_near_overlap": overlap_metrics["near_overlap"].mean(),
            "kc_far_overlap": overlap_metrics["far_overlap"].mean(),
            "kc_negative_overlap": overlap_metrics["negative_overlap"].mean(),
            "kc_negative_overlap_excess": overlap_metrics["negative_excess"],
            "kc_usage_cv2": usage_metrics["cv2"],
            "kc_usage_effective_fraction": usage_metrics["effective_fraction"],
            "kc_usage_max_share": usage_metrics["max_usage"],
            "near_overlap_ratio": batch["near_overlap_ratio"].to(self.device).mean(),
            "far_overlap_ratio": batch["far_overlap_ratio"].to(self.device).mean(),
        }

        outputs = {
            "anchor_projection": anchor_projection,
            "near_projection": near_projection,
            "far_projection": far_projection,
            "negative_projection": negative_projection,
            "image_index": batch["image_index"],
            "negative_index": batch["negative_index"],
            "batch": {
                "anchor": batch["anchor"].detach().cpu(),
                "near_positive": batch["near_positive"].detach().cpu(),
                "far_positive": batch["far_positive"].detach().cpu(),
                "class_positive": batch["class_positive"].detach().cpu(),
                "negative": batch["negative"].detach().cpu(),
                "near_shift": batch["near_shift"].detach().cpu(),
                "far_shift": batch["far_shift"].detach().cpu(),
                "near_overlap_ratio": batch["near_overlap_ratio"].detach().cpu(),
                "far_overlap_ratio": batch["far_overlap_ratio"].detach().cpu(),
            },
        }
        return loss, metrics, outputs

    def _accumulate_projection_metrics(self, running, metrics, batch_size: int):
        for key in running:
            running[key] += float(metrics[key].item()) * batch_size

    def _evaluate_projection_loader(
        self,
        backbone: VisionBackbone,
        projection: VisionProjection,
        shift_head: ProjectionShiftHead,
        loader: DataLoader,
        class_kc_loss_weight: float | None = None,
    ):
        running = self._empty_projection_metrics()
        processed = 0

        projection.eval()
        shift_head.eval()
        with torch.no_grad():
            for batch in loader:
                loss, metrics, _ = self._projection_forward(
                    backbone,
                    projection,
                    shift_head,
                    batch,
                    class_kc_loss_weight=class_kc_loss_weight,
                )
                batch_size = batch["anchor"].size(0)
                processed += batch_size
                self._accumulate_projection_metrics(running, metrics, batch_size)

        return self._finalize_projection_metrics(running, processed)

    def _save_projection_checkpoint(
        self,
        projection: VisionProjection,
        shift_head: ProjectionShiftHead,
        epoch_num: int,
        metric_value: float,
        best_metric: float,
        best_ordering: float,
        ordering_value: float,
    ):
        projection_path, shift_head_path = self._projection_checkpoint_paths()

        if self.best_only:
            improved = (
                metric_value < (best_metric - self.early_stop_min_delta)
                or (
                    abs(metric_value - best_metric) <= self.early_stop_min_delta
                    and ordering_value > best_ordering
                )
            )
            if improved:
                torch.save(projection.state_dict(), str(projection_path))
                torch.save(shift_head.state_dict(), str(shift_head_path))
                return metric_value, ordering_value, True
            return best_metric, best_ordering, False

        epoch_projection_path = self.models_dir / f"{self.projection_model}_Epoch{epoch_num}.pth"
        epoch_shift_head_path = self.models_dir / f"{self.projection_model}_ShiftHead_Epoch{epoch_num}.pth"
        torch.save(projection.state_dict(), str(epoch_projection_path))
        torch.save(shift_head.state_dict(), str(epoch_shift_head_path))
        return metric_value, ordering_value, True

    def _load_projection_backbone(self) -> VisionBackbone:
        checkpoint_path = resolve_pretrained_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=self.backbone_checkpoint,
            model_name=self.lobula_plate_model,
        )
        self.logger.info(f"Loading pretrained projection backbone from {checkpoint_path}")

        backbone = VisionBackbone().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
        self._configure_projection_backbone(backbone)
        return backbone

    def train_projection(self):
        if self.snn:
            raise NotImplementedError("Projection training is currently implemented for the ANN backbone only.")

        self._confirm_overwrite()
        self._set_seed()
        train_paths, val_paths = self._split_projection_images()

        if len(train_paths) == 0:
            raise RuntimeError("No images found for projection training.")

        self.logger.info(
            f"Projection training pool: {len(train_paths)} images, validation pool: {len(val_paths)} images"
        )

        backbone = self._load_projection_backbone()
        self._set_projection_backbone_trainability(backbone, epoch_num=0)
        kc_config = self._projection_kc_config()
        self.projection_target_kc_active = kc_config["kc_target_active"]
        projection = VisionProjection(
            lobula_dim=backbone.lobula.embedding.out_features,
            lobula_feature_channels=backbone.lobula.norm.num_channels,
            lobula_plate_channels=backbone.lobula_plate.norm.num_channels,
            vpn_dim=self.projection_vpn_dim,
            spatial_pool_size=self.projection_spatial_pool_size,
            spatial_token_dim=self.projection_spatial_token_dim,
            kc_dim=kc_config["kc_dim"],
            kc_fan_in=self.projection_kc_fan_in,
            kc_sparsity=kc_config["kc_sparsity"],
            apl_feedback_strength=self.projection_apl_feedback_strength,
            apl_gain_adapt_rate=self.projection_apl_gain_adapt_rate,
            apl_threshold_lr=self.projection_apl_threshold_lr,
            apl_num_iters=self.projection_apl_num_iters,
        ).to(self.device)
        shift_head = ProjectionShiftHead(in_dim=self.projection_vpn_dim).to(self.device)

        print("built projection + shift head")
        opt = self._build_projection_optimizer(backbone, projection, shift_head)
        print("built optimizer")
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        preview_paths = val_paths if val_paths else train_paths
        preview_ds = self._build_projection_dataset(
            image_paths=preview_paths,
            max_samples=min(self.projection_preview_samples, max(1, len(preview_paths))),
            deterministic=True,
            seed=self.split_seed + 50_000,
            appearance_transform=None,
        )
        preview_dl = DataLoader(
            preview_ds,
            **self._loader_kwargs(
                min(self.batch_size, len(preview_ds)),
                shuffle=False,
                drop_last=False,
            ),
        )
        preview_batch = next(iter(preview_dl))

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "class_kc_loss_weight": [],
            "kc_overlap_loss_weight": [],
            "kc_usage_loss_weight": [],
            "train_shift_mae_px": [],
            "val_shift_mae_px": [],
            "train_kc_ordering_acc": [],
            "val_kc_ordering_acc": [],
            "train_kc_overlap_ordering_acc": [],
            "val_kc_overlap_ordering_acc": [],
            "train_kc_active_fraction": [],
            "val_kc_active_fraction": [],
            "train_kc_negative_overlap": [],
            "val_kc_negative_overlap": [],
            "train_kc_usage_effective_fraction": [],
            "val_kc_usage_effective_fraction": [],
            "val_feature_near_similarity": [],
            "val_feature_far_similarity": [],
            "val_feature_negative_similarity": [],
            "val_feature_class_similarity": [],
            "val_spatial_near_similarity": [],
            "val_spatial_far_similarity": [],
            "val_spatial_negative_similarity": [],
            "val_kc_near_similarity": [],
            "val_kc_far_similarity": [],
            "val_kc_negative_similarity": [],
            "val_kc_class_similarity": [],
        }
        history_json_path = self.outdir / "projection_history.json"
        history_plot_path = self.outdir / "projection_history.png"
        best_metric = float("inf")
        best_ordering = float("-inf")
        best_epoch = 0
        patience = 0
        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            self._set_projection_backbone_trainability(backbone, epoch_num)

            if any(p.requires_grad for p in backbone.parameters()):
                backbone.train()
            else:
                backbone.eval()
            current_class_kc_weight = self._projection_class_kc_weight(epoch_num)
            projection.train()
            shift_head.train()

            train_ds = self._build_projection_dataset(
                image_paths=train_paths,
                max_samples=self.train_samples,
                deterministic=False,
                seed=self.run_seed + epoch,
                appearance_transform=self._spatial_appearance_transform(),
            )
            train_dl = DataLoader(
                train_ds,
                **self._loader_kwargs(
                    self.batch_size,
                    shuffle=True,
                    drop_last=len(train_ds) >= self.batch_size,
                ),
            )

            val_loader = None
            if val_paths:
                val_ds = self._build_projection_dataset(
                    image_paths=val_paths,
                    max_samples=self.val_samples,
                    deterministic=True,
                    seed=self.split_seed + 10_000,
                    appearance_transform=None,
                )
                val_loader = DataLoader(
                    val_ds,
                    **self._loader_kwargs(
                        self.spatial_val_batch_size or self.batch_size,
                        shuffle=False,
                        drop_last=False,
                    ),
                )

            running = self._empty_projection_metrics()
            processed = 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch_num}/{self.epochs}", unit="batch")

            for batch in pbar:
                opt.zero_grad(set_to_none=True)
                with self._backbone_autocast_context():
                    loss, metrics, _ = self._projection_forward(
                        backbone,
                        projection,
                        shift_head,
                        batch,
                        class_kc_loss_weight=current_class_kc_weight,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clip_params = list(projection.parameters()) + list(shift_head.parameters())
                clip_params += [p for p in backbone.parameters() if p.requires_grad]

                torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                scaler.step(opt)
                scaler.update()

                batch_size = batch["anchor"].size(0)
                processed += batch_size
                self._accumulate_projection_metrics(running, metrics, batch_size)

                pbar.set_postfix(
                    loss=f"{running['loss'] / processed:.4f}",
                    shift_mae=f"{running['shift_mae_px'] / processed:.2f}px",
                    kc_order=f"{running['kc_ordering_acc'] / processed:.3f}",
                    kc_frac=f"{running['kc_active_fraction'] / processed:.3f}",
                    kc_neg_ov=f"{running['kc_negative_overlap'] / processed:.3f}",
                )

            train_metrics = self._finalize_projection_metrics(running, processed)
            val_metrics = None
            if val_loader is not None:
                val_metrics = self._evaluate_projection_loader(
                    backbone,
                    projection,
                    shift_head,
                    val_loader,
                    class_kc_loss_weight=current_class_kc_weight,
                )

            sched.step()

            monitor_metric = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            monitor_ordering = (
                val_metrics["kc_ordering_acc"] if val_metrics is not None else train_metrics["kc_ordering_acc"]
            )
            best_metric, best_ordering, improved = self._save_projection_checkpoint(
                projection,
                shift_head,
                epoch_num,
                monitor_metric,
                best_metric,
                best_ordering,
                monitor_ordering,
            )

            if improved:
                best_epoch = epoch_num
                patience = 0
            else:
                patience += 1

            history["epoch"].append(epoch_num)
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"] if val_metrics is not None else train_metrics["loss"])
            history["class_kc_loss_weight"].append(current_class_kc_weight)
            history["kc_overlap_loss_weight"].append(float(getattr(self, "projection_kc_overlap_loss_weight", 0.0)))
            history["kc_usage_loss_weight"].append(float(getattr(self, "projection_kc_usage_loss_weight", 0.0)))
            history["train_shift_mae_px"].append(train_metrics["shift_mae_px"])
            history["val_shift_mae_px"].append(
                val_metrics["shift_mae_px"] if val_metrics is not None else train_metrics["shift_mae_px"]
            )
            history["train_kc_ordering_acc"].append(train_metrics["kc_ordering_acc"])
            history["val_kc_ordering_acc"].append(
                val_metrics["kc_ordering_acc"] if val_metrics is not None else train_metrics["kc_ordering_acc"]
            )
            history["train_kc_overlap_ordering_acc"].append(train_metrics["kc_overlap_ordering_acc"])
            history["val_kc_overlap_ordering_acc"].append(
                val_metrics["kc_overlap_ordering_acc"] if val_metrics is not None else train_metrics["kc_overlap_ordering_acc"]
            )
            history["train_kc_active_fraction"].append(train_metrics["kc_active_fraction"])
            history["val_kc_active_fraction"].append(
                val_metrics["kc_active_fraction"] if val_metrics is not None else train_metrics["kc_active_fraction"]
            )
            history["train_kc_negative_overlap"].append(train_metrics["kc_negative_overlap"])
            history["val_kc_negative_overlap"].append(
                val_metrics["kc_negative_overlap"] if val_metrics is not None else train_metrics["kc_negative_overlap"]
            )
            history["train_kc_usage_effective_fraction"].append(train_metrics["kc_usage_effective_fraction"])
            history["val_kc_usage_effective_fraction"].append(
                val_metrics["kc_usage_effective_fraction"]
                if val_metrics is not None
                else train_metrics["kc_usage_effective_fraction"]
            )
            reference_metrics = val_metrics if val_metrics is not None else train_metrics
            history["val_feature_near_similarity"].append(reference_metrics["feature_near_similarity"])
            history["val_feature_far_similarity"].append(reference_metrics["feature_far_similarity"])
            history["val_feature_negative_similarity"].append(reference_metrics["feature_negative_similarity"])
            history["val_feature_class_similarity"].append(reference_metrics["feature_class_similarity"])
            history["val_spatial_near_similarity"].append(reference_metrics["spatial_near_similarity"])
            history["val_spatial_far_similarity"].append(reference_metrics["spatial_far_similarity"])
            history["val_spatial_negative_similarity"].append(reference_metrics["spatial_negative_similarity"])
            history["val_kc_near_similarity"].append(reference_metrics["kc_near_similarity"])
            history["val_kc_far_similarity"].append(reference_metrics["kc_far_similarity"])
            history["val_kc_negative_similarity"].append(reference_metrics["kc_negative_similarity"])
            history["val_kc_class_similarity"].append(reference_metrics["kc_class_similarity"])

            history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            plot_projection_history(history, history_plot_path, target_kc_sparsity=kc_config["kc_sparsity"])

            projection.eval()
            shift_head.eval()
            with torch.no_grad():
                _, _, preview_outputs = self._projection_forward(
                    backbone,
                    projection,
                    shift_head,
                    preview_batch,
                    class_kc_loss_weight=current_class_kc_weight,
                )
                preview_snapshot = build_projection_snapshot(
                    preview_outputs["image_index"],
                    preview_outputs["negative_index"],
                    preview_outputs["batch"],
                    preview_outputs["anchor_projection"],
                    preview_outputs["near_projection"],
                    preview_outputs["far_projection"],
                    preview_outputs["negative_projection"],
                        metadata={
                            "near_shift_range": f"{self.projection_near_min_shift} to {self.projection_near_max_shift} px",
                            "far_shift_range": f"{self.projection_far_min_shift} to {self.projection_far_max_shift} px",
                            "target_kc_sparsity": f"{kc_config['kc_sparsity']:.4f}",
                            "target_kc_active_count": f"{kc_config['kc_target_active']}",
                            "competition_module": "APLCompetition",
                            "apl_feedback_strength": f"{self.projection_apl_feedback_strength:.4f}",
                            "class_objective": "supervised_contrastive",
                            "class_kc_loss_weight": f"{current_class_kc_weight:.4f}",
                            "kc_overlap_loss_weight": f"{float(getattr(self, 'projection_kc_overlap_loss_weight', 0.0)):.4f}",
                            "kc_overlap_margin": f"{float(getattr(self, 'projection_kc_overlap_margin', 0.15)):.4f}",
                            "kc_negative_overlap_target": f"{float(getattr(self, 'projection_kc_negative_overlap_target', 0.10)):.4f}",
                            "kc_usage_loss_weight": f"{float(getattr(self, 'projection_kc_usage_loss_weight', 0.0)):.4f}",
                        },
                    )
            snapshot_epoch_path = self.outdir / f"projection_snapshot_epoch{epoch_num:03d}.png"
            snapshot_latest_path = self.outdir / "projection_snapshot_latest.png"
            snapshot_json_epoch_path = self.outdir / f"projection_snapshot_epoch{epoch_num:03d}.json"
            snapshot_json_latest_path = self.outdir / "projection_snapshot_latest.json"
            plot_projection_snapshot(preview_snapshot, snapshot_epoch_path)
            plot_projection_snapshot(preview_snapshot, snapshot_latest_path)
            write_projection_snapshot_json(preview_snapshot, snapshot_json_epoch_path)
            write_projection_snapshot_json(preview_snapshot, snapshot_json_latest_path)
            if improved:
                plot_projection_snapshot(preview_snapshot, self.outdir / "projection_snapshot_best.png")
                write_projection_snapshot_json(preview_snapshot, self.outdir / "projection_snapshot_best.json")

            with open(self.outdir / "training_log.txt", "a", encoding="utf-8") as f:
                line = (
                    f"Epoch {epoch_num}/{self.epochs}, "
                    f"TrainLoss: {train_metrics['loss']:.4f}, TrainFeatureLoss: {train_metrics['feature_loss']:.4f}, "
                    f"TrainClassFeatureLoss: {train_metrics['class_feature_loss']:.4f}, "
                    f"TrainShiftLoss: {train_metrics['shift_loss']:.4f}, TrainKCLoss: {train_metrics['kc_loss']:.4f}, "
                    f"TrainClassKCLoss: {train_metrics['class_kc_loss']:.4f}, TrainClassKCWeight: {current_class_kc_weight:.4f}, "
                    f"TrainKCSparsityLoss: {train_metrics['kc_sparsity_loss']:.4f}, "
                    f"TrainKCOverlapLoss: {train_metrics['kc_overlap_loss']:.4f}, TrainKCUsageLoss: {train_metrics['kc_usage_loss']:.4f}, "
                    f"TrainShiftMAE(px): {train_metrics['shift_mae_px']:.4f}, TrainKCOrder: {train_metrics['kc_ordering_acc']:.4f}, "
                    f"TrainKCOverlapOrder: {train_metrics['kc_overlap_ordering_acc']:.4f}, "
                    f"TrainKCActiveFrac: {train_metrics['kc_active_fraction']:.4f}, "
                    f"TrainKCNegativeOverlap: {train_metrics['kc_negative_overlap']:.4f}, "
                    f"TrainKCUsageEffectiveFrac: {train_metrics['kc_usage_effective_fraction']:.4f}, "
                    f"ValLoss: {(val_metrics['loss'] if val_metrics is not None else train_metrics['loss']):.4f}, "
                    f"ValShiftMAE(px): {(val_metrics['shift_mae_px'] if val_metrics is not None else train_metrics['shift_mae_px']):.4f}, "
                    f"ValKCOrder: {(val_metrics['kc_ordering_acc'] if val_metrics is not None else train_metrics['kc_ordering_acc']):.4f}, "
                    f"ValKCOverlapOrder: {(val_metrics['kc_overlap_ordering_acc'] if val_metrics is not None else train_metrics['kc_overlap_ordering_acc']):.4f}, "
                    f"ValKCActiveFrac: {(val_metrics['kc_active_fraction'] if val_metrics is not None else train_metrics['kc_active_fraction']):.4f}, "
                    f"ValKCNegativeOverlap: {(val_metrics['kc_negative_overlap'] if val_metrics is not None else train_metrics['kc_negative_overlap']):.4f}, "
                    f"ValKCUsageEffectiveFrac: {(val_metrics['kc_usage_effective_fraction'] if val_metrics is not None else train_metrics['kc_usage_effective_fraction']):.4f}\n"
                )
                f.write(line)

            self.logger.info(
                f"Epoch {epoch_num}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={(val_metrics['loss'] if val_metrics is not None else train_metrics['loss']):.4f}, "
                f"shift_mae={(val_metrics['shift_mae_px'] if val_metrics is not None else train_metrics['shift_mae_px']):.3f}px, "
                f"kc_order={(val_metrics['kc_ordering_acc'] if val_metrics is not None else train_metrics['kc_ordering_acc']):.3f}, "
                f"kc_overlap={(val_metrics['kc_overlap_ordering_acc'] if val_metrics is not None else train_metrics['kc_overlap_ordering_acc']):.3f}, "
                f"kc_active={(val_metrics['kc_active_fraction'] if val_metrics is not None else train_metrics['kc_active_fraction']):.3f}, "
                f"kc_neg_ov={(val_metrics['kc_negative_overlap'] if val_metrics is not None else train_metrics['kc_negative_overlap']):.3f}, "
                f"kc_usage_eff={(val_metrics['kc_usage_effective_fraction'] if val_metrics is not None else train_metrics['kc_usage_effective_fraction']):.3f}, "
                f"class_kc_w={current_class_kc_weight:.3f}, "
                f"class_feat={(val_metrics['feature_class_similarity'] if val_metrics is not None else train_metrics['feature_class_similarity']):.3f}, "
                f"class_kc={(val_metrics['kc_class_similarity'] if val_metrics is not None else train_metrics['kc_class_similarity']):.3f}"
            )

            if val_loader is not None and self.early_stop_patience > 0 and patience >= self.early_stop_patience:
                self.logger.info(
                    f"Early stopping projection training at epoch {epoch_num}; "
                    f"best epoch was {best_epoch} with loss {best_metric:.4f}"
                )
                break

        self.logger.info(
            "\nProjection training complete. "
            f"Best monitored loss {best_metric:.4f} at epoch {best_epoch} → {self.model_path}"
        )

    def train_reward_memory(self):
        if self.snn:
            raise NotImplementedError("Reward-memory training is currently implemented for the ANN stack only.")

        self._confirm_overwrite()
        self._set_seed()

        reward_dataset = self._build_reward_dataset()
        train_indices, val_indices = self._split_reward_indices(reward_dataset)
        train_dataset = Subset(reward_dataset, train_indices.tolist())
        val_dataset = Subset(reward_dataset, val_indices.tolist())

        self.logger.info(
            "Reward-memory dataset split: "
            f"train={len(train_dataset)} images, val={len(val_dataset)} images, "
            f"rewarded={len(self.rewarded_class_names)} class(es)"
        )

        backbone, projection = self._load_reward_frozen_backbone_projection()
        reward_head = RewardMemoryHead(
            in_dim=self._reward_head_input_dim(),
            hidden_dim=int(getattr(self, "reward_hidden_dim", 0)),
            dropout=float(getattr(self, "reward_dropout", 0.0)),
        ).to(self.device)
        optimizer = self._build_reward_optimizer(reward_head)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

        train_reward_labels = reward_dataset.reward_labels[train_indices]
        pos_weight = self._resolve_reward_pos_weight(train_reward_labels)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device, dtype=torch.float32)
        )

        config_payload = {
            "reward_dataset": str(self.reward_dataset),
            "reward_feature": str(self.reward_feature),
            "rewarded_class_indices": [int(idx) for idx in self.rewarded_class_indices],
            "rewarded_class_names": list(self.rewarded_class_names),
            "reward_head_hidden_dim": int(getattr(self, "reward_hidden_dim", 0)),
            "reward_threshold": float(getattr(self, "reward_threshold", 0.5)),
            "reward_pos_weight": float(pos_weight),
            "backbone_checkpoint": str(self.reward_backbone_checkpoint),
            "projection_checkpoint": str(self.reward_projection_checkpoint),
        }
        (self.outdir / "reward_memory_config.json").write_text(
            json.dumps(config_payload, indent=2),
            encoding="utf-8",
        )

        train_loader = DataLoader(
            train_dataset,
            **self._loader_kwargs(self.batch_size, shuffle=True, drop_last=False),
        )
        val_loader = DataLoader(
            val_dataset,
            **self._loader_kwargs(self.spatial_val_batch_size or self.batch_size, shuffle=False, drop_last=False),
        )

        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_balanced_accuracy": [],
            "val_balanced_accuracy": [],
            "train_auroc": [],
            "val_auroc": [],
            "train_average_precision": [],
            "val_average_precision": [],
        }
        history_json_path = self.outdir / "reward_memory_history.json"
        history_plot_path = self.outdir / "reward_memory_history.png"
        best_metric = float("-inf")
        best_loss = float("inf")
        best_epoch = 0
        patience = 0

        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            self.logger.info(f"Reward-memory epoch {epoch_num}/{self.epochs}")
            train_metrics, _ = self._run_reward_epoch(
                backbone,
                projection,
                reward_head,
                train_loader,
                loss_fn,
                optimizer=optimizer,
                scaler=scaler,
            )
            val_metrics, val_outputs = self._run_reward_epoch(
                backbone,
                projection,
                reward_head,
                val_loader,
                loss_fn,
                optimizer=None,
                scaler=None,
            )

            best_metric, best_loss, improved = self._save_reward_checkpoint(
                reward_head,
                epoch_num,
                metric_value=float(val_metrics["balanced_accuracy"]),
                best_metric=best_metric,
                loss_value=float(val_metrics["loss"]),
                best_loss=best_loss,
            )

            if improved:
                best_epoch = epoch_num
                patience = 0
            else:
                patience += 1

            history["epoch"].append(epoch_num)
            history["train_loss"].append(float(train_metrics["loss"]))
            history["val_loss"].append(float(val_metrics["loss"]))
            history["train_accuracy"].append(float(train_metrics["accuracy"]))
            history["val_accuracy"].append(float(val_metrics["accuracy"]))
            history["train_balanced_accuracy"].append(float(train_metrics["balanced_accuracy"]))
            history["val_balanced_accuracy"].append(float(val_metrics["balanced_accuracy"]))
            history["train_auroc"].append(float(train_metrics["auroc"]))
            history["val_auroc"].append(float(val_metrics["auroc"]))
            history["train_average_precision"].append(float(train_metrics["average_precision"]))
            history["val_average_precision"].append(float(val_metrics["average_precision"]))

            history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
            fig_history = plot_reward_history(history, history_plot_path)
            if fig_history is not None:
                plt.close(fig_history)

            fig_class = plot_reward_by_class(
                class_names=reward_dataset.class_names,
                class_labels=val_outputs["class_labels"],
                reward_probabilities=val_outputs["probabilities"],
                rewarded_class_indices=self.rewarded_class_indices,
                save_path=self.outdir / "reward_probability_by_class_latest.png",
            )
            plt.close(fig_class)
            if improved:
                fig_best = plot_reward_by_class(
                    class_names=reward_dataset.class_names,
                    class_labels=val_outputs["class_labels"],
                    reward_probabilities=val_outputs["probabilities"],
                    rewarded_class_indices=self.rewarded_class_indices,
                    save_path=self.outdir / "reward_probability_by_class_best.png",
                )
                plt.close(fig_best)

            with open(self.outdir / "training_log.txt", "a", encoding="utf-8") as handle:
                handle.write(
                    f"Epoch {epoch_num}/{self.epochs}, "
                    f"TrainLoss: {train_metrics['loss']:.4f}, TrainAcc: {train_metrics['accuracy']:.4f}, "
                    f"TrainBalAcc: {train_metrics['balanced_accuracy']:.4f}, TrainAUROC: {train_metrics['auroc']:.4f}, "
                    f"TrainAP: {train_metrics['average_precision']:.4f}, "
                    f"ValLoss: {val_metrics['loss']:.4f}, ValAcc: {val_metrics['accuracy']:.4f}, "
                    f"ValBalAcc: {val_metrics['balanced_accuracy']:.4f}, ValAUROC: {val_metrics['auroc']:.4f}, "
                    f"ValAP: {val_metrics['average_precision']:.4f}\n"
                )

            self.logger.info(
                f"Epoch {epoch_num}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"train_bal_acc={train_metrics['balanced_accuracy']:.3f}, "
                f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}, "
                f"val_auroc={val_metrics['auroc']:.3f}, "
                f"val_ap={val_metrics['average_precision']:.3f}, "
                f"pos_prob={val_metrics['mean_positive_probability']:.3f}, "
                f"neg_prob={val_metrics['mean_negative_probability']:.3f}"
            )

            if self.early_stop_patience > 0 and patience >= self.early_stop_patience:
                self.logger.info(
                    f"Early stopping reward-memory training at epoch {epoch_num}; "
                    f"best epoch was {best_epoch} with val balanced accuracy {best_metric:.4f}"
                )
                break

        summary_payload = {
            "best_epoch": int(best_epoch),
            "best_val_balanced_accuracy": float(best_metric),
            "best_val_loss": float(best_loss),
            "rewarded_class_indices": [int(idx) for idx in self.rewarded_class_indices],
            "rewarded_class_names": list(self.rewarded_class_names),
        }
        (self.outdir / "reward_memory_summary.json").write_text(
            json.dumps(summary_payload, indent=2),
            encoding="utf-8",
        )

        self.logger.info(
            "\nReward-memory training complete. "
            f"Best val balanced accuracy {best_metric:.4f} at epoch {best_epoch} → {self.model_path}"
        )

    def _load_pretrained_backbone(self) -> VisionBackbone:
        checkpoint_path = resolve_pretrained_checkpoint(
            models_dir=self.models_dir,
            checkpoint_name=self.backbone_checkpoint,
            model_name=self.vision_model,
        )
        self.logger.info(f"Loading pretrained backbone from {checkpoint_path}")

        backbone = VisionBackbone().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        backbone.load_state_dict(state_dict, strict=True)
        self._configure_lobula_trainability(backbone, unfreeze_top_lobula=False)
        return backbone

    def train_lobula_plate_pose_pairs(self):
        if self.snn:
            raise NotImplementedError("Lobula plate fine-tuning is currently implemented for the ANN backbone only.")

        self._confirm_overwrite()
        self._set_seed()
        resolved_root, train_pairs, val_pairs, _, _ = self._split_wildscenes_pose_pairs()
        train_pairs, val_pairs, _ = self._prepare_wildscenes_resized_cache(
            resolved_root,
            train_pairs,
            val_pairs,
        )
        effective_train_samples = len(train_pairs) if self.train_samples <= 0 else min(len(train_pairs), self.train_samples)
        effective_val_samples = 0
        if val_pairs:
            effective_val_samples = len(val_pairs) if self.val_samples <= 0 else min(len(val_pairs), self.val_samples)
        self.logger.info(
            "WildScenes pose epochs will use "
            f"{effective_train_samples} unique training anchors"
            + (
                f" and {effective_val_samples} unique validation anchors"
                if val_pairs
                else ""
            )
            + " per epoch."
        )

        backbone = self._load_pretrained_backbone()
        self._configure_pose_pair_trainability(backbone, unfreeze_top_lobula=False)
        pose_head = PoseRelationHead(
            in_channels=backbone.lobula_plate.norm.num_channels,
            pool_size=self.wildscenes_pose_pool_size,
        ).to(self.device)

        opt = self._build_pose_pair_optimizer(backbone, pose_head)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        best_val_loss = float("inf")
        best_val_top1 = float("-inf")
        best_epoch = 0
        patience = 0
        top_lobula_unfrozen = False
        pose_head_path = self.models_dir / f"{self.lobula_plate_model}_PoseHead.pth"

        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            if self.unfreeze_lobula_epoch > 0 and (not top_lobula_unfrozen) and epoch_num >= self.unfreeze_lobula_epoch:
                top_lobula_unfrozen = True
                self._configure_pose_pair_trainability(backbone, unfreeze_top_lobula=True)
                self.logger.info(
                    f"Epoch {epoch_num}: unfreezing top lobula layers at LR scale {self.lobula_lr_scale}"
                )

            self._configure_pose_pair_trainability(backbone, unfreeze_top_lobula=top_lobula_unfrozen)
            train_ds = WildScenesPosePairDataset(
                pairs=train_pairs,
                image_size=self.spatial_image_size,
                appearance_transform=self._spatial_appearance_transform(),
                max_samples=self.train_samples,
                deterministic=False,
                seed=self.run_seed + epoch,
            )
            train_dl = DataLoader(
                train_ds,
                **self._loader_kwargs(
                    self.batch_size,
                    shuffle=True,
                    drop_last=len(train_ds) >= self.batch_size,
                ),
            )

            val_loader = None
            if val_pairs:
                val_batch_size = self.spatial_val_batch_size or self.batch_size
                val_ds = WildScenesPosePairDataset(
                    pairs=val_pairs,
                    image_size=self.spatial_image_size,
                    appearance_transform=None,
                    max_samples=self.val_samples,
                    deterministic=True,
                    seed=self.split_seed + 20_000,
                )
                val_loader = DataLoader(
                    val_ds,
                    **self._loader_kwargs(val_batch_size, shuffle=False, drop_last=False),
                )

            running = self._empty_pose_metrics()
            processed = 0
            detach_lobula_for_plate = not top_lobula_unfrozen
            pose_head.train()
            pbar = tqdm(train_dl, desc=f"Epoch {epoch_num}/{self.epochs}", unit="batch")

            for batch in pbar:
                anchor = batch["anchor"].to(self.device, non_blocking=self.device.type == "cuda")
                positive = batch["positive"].to(self.device, non_blocking=self.device.type == "cuda")
                relative_translation = batch["relative_translation"].to(self.device, non_blocking=self.device.type == "cuda")
                relative_yaw = batch["relative_yaw"].to(self.device, non_blocking=self.device.type == "cuda")
                distance_m = batch["distance_m"].to(self.device, non_blocking=self.device.type == "cuda")

                opt.zero_grad(set_to_none=True)
                with self._backbone_autocast_context():
                    anchor_outputs = backbone(anchor, detach_lobula_for_plate=detach_lobula_for_plate, return_maps=True)
                    positive_outputs = backbone(positive, detach_lobula_for_plate=detach_lobula_for_plate, return_maps=True)
                    anchor_lobula = anchor_outputs["lobula"]
                    positive_lobula = positive_outputs["lobula"]
                    anchor_plate = anchor_outputs["lobula_plate"]
                    positive_plate = positive_outputs["lobula_plate"]

                    place_metrics = descriptor_place_matching_loss(
                        anchor_lobula,
                        positive_lobula,
                        temperature=self.dense_temperature,
                    )
                    pose_predictions = pose_head(anchor_plate, positive_plate)
                    pose_metrics = relative_pose_regression_loss(
                        pose_predictions,
                        target_translation_m=relative_translation,
                        target_yaw_rad=relative_yaw,
                        translation_scale_m=self.wildscenes_match_radius_m,
                    )
                    loss = (
                        self.place_loss_weight * place_metrics["loss"]
                        + self.pose_loss_weight * pose_metrics["loss"]
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                trainable = self._trainable_parameters(backbone) + list(pose_head.parameters())
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()

                batch_size = anchor.size(0)
                processed += batch_size
                running["loss"] += loss.item() * batch_size
                running["place_loss"] += place_metrics["loss"].item() * batch_size
                running["pose_loss"] += pose_metrics["loss"].item() * batch_size
                running["translation_loss"] += pose_metrics["translation_loss"].item() * batch_size
                running["yaw_loss"] += pose_metrics["yaw_loss"].item() * batch_size
                running["place_top1"] += place_metrics["top1"].item() * batch_size
                running["positive_cosine"] += place_metrics["positive_cosine"].item() * batch_size
                running["translation_mae_m"] += pose_metrics["translation_mae_m"].item() * batch_size
                running["yaw_mae_deg"] += pose_metrics["yaw_mae_deg"].item() * batch_size
                running["distance_m"] += distance_m.mean().item() * batch_size
                running["pairs"] += batch_size

                pbar.set_postfix(
                    loss=f"{running['loss'] / processed:.4f}",
                    place_top1=f"{running['place_top1'] / processed:.3f}",
                    trans_mae=f"{running['translation_mae_m'] / processed:.2f}m",
                    yaw_mae=f"{running['yaw_mae_deg'] / processed:.2f}deg",
                    best_val=f"{best_val_loss:.3f}" if best_epoch > 0 else "n/a",
                )

            train_metrics = self._finalize_pose_metrics(running, processed, len(train_dl))
            val_metrics = (
                self._evaluate_pose_loader(
                    backbone,
                    pose_head,
                    val_loader,
                    detach_lobula_for_plate=detach_lobula_for_plate,
                )
                if val_loader is not None
                else train_metrics
            )

            sched.step()

            improved = (
                (best_val_loss - val_metrics["loss"]) > self.early_stop_min_delta
                or (
                    abs(val_metrics["loss"] - best_val_loss) <= self.early_stop_min_delta
                    and val_metrics["place_top1"] > best_val_top1
                )
            )
            if improved:
                best_val_loss = val_metrics["loss"]
                best_val_top1 = val_metrics["place_top1"]
                best_epoch = epoch_num
                patience = 0
                torch.save(backbone.state_dict(), str(self.model_path))
                torch.save(pose_head.state_dict(), str(pose_head_path))
                self.logger.info(
                    f"Epoch {epoch_num}: new best validation pose loss {best_val_loss:.4f} "
                    f"(place_top1 {best_val_top1:.4f}) → {self.model_path}"
                )
            else:
                patience += 1

            if not self.best_only:
                epoch_path = self.models_dir / f"{self.lobula_plate_model}_Epoch{epoch_num}.pth"
                epoch_head_path = self.models_dir / f"{self.lobula_plate_model}_PoseHead_Epoch{epoch_num}.pth"
                torch.save(backbone.state_dict(), str(epoch_path))
                torch.save(pose_head.state_dict(), str(epoch_head_path))

            with open(self.outdir / "training_log.txt", "a") as f:
                f.write(
                    f"Epoch {epoch_num}/{self.epochs}, "
                    f"TrainLoss: {train_metrics['loss']:.4f}, TrainPlaceLoss: {train_metrics['place_loss']:.4f}, "
                    f"TrainPoseLoss: {train_metrics['pose_loss']:.4f}, TrainPlaceTop1: {train_metrics['place_top1']:.4f}, "
                    f"TrainPositiveCosine: {train_metrics['positive_cosine']:.4f}, "
                    f"TrainTranslationMAE(m): {train_metrics['translation_mae_m']:.4f}, "
                    f"TrainYawMAE(deg): {train_metrics['yaw_mae_deg']:.4f}, "
                    f"ValLoss: {val_metrics['loss']:.4f}, ValPlaceLoss: {val_metrics['place_loss']:.4f}, "
                    f"ValPoseLoss: {val_metrics['pose_loss']:.4f}, ValPlaceTop1: {val_metrics['place_top1']:.4f}, "
                    f"ValPositiveCosine: {val_metrics['positive_cosine']:.4f}, "
                    f"ValTranslationMAE(m): {val_metrics['translation_mae_m']:.4f}, "
                    f"ValYawMAE(deg): {val_metrics['yaw_mae_deg']:.4f}, "
                    f"AvgPairDistance(m): {train_metrics['distance_m']:.4f}, "
                    f"Pairs/Batch: {train_metrics['pairs_per_batch']:.1f}, "
                    f"TopLobulaUnfrozen: {top_lobula_unfrozen}\n"
                )

            self.logger.info(
                f"Epoch {epoch_num}: train place_top1={train_metrics['place_top1']:.4f}, "
                f"val place_top1={val_metrics['place_top1']:.4f}, "
                f"train translation_mae={train_metrics['translation_mae_m']:.3f}m, "
                f"val translation_mae={val_metrics['translation_mae_m']:.3f}m, "
                f"train yaw_mae={train_metrics['yaw_mae_deg']:.2f}deg, "
                f"val yaw_mae={val_metrics['yaw_mae_deg']:.2f}deg"
            )

            if val_loader is not None and patience >= self.early_stop_patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch_num}: validation pose loss has not improved "
                    f"for {self.early_stop_patience} epochs."
                )
                break

        self.logger.info(
            "\nWildScenes pose-pair lobula plate fine-tuning complete. "
            f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch} → {self.model_path} "
            f"(pose head: {pose_head_path})"
        )

    def train_lobula_plate(self):
        if self.snn:
            raise NotImplementedError("Lobula plate fine-tuning is currently implemented for the ANN backbone only.")

        self._confirm_overwrite()
        self._set_seed()
        ds_root = self._resolve_dataset_root()
        self._ensure_dataset_root_exists(ds_root)
        train_split_file = self._resolve_split_file(self.spatial_train_split_file, ds_root)
        if train_split_file is not None:
            self.logger.info(f"Using spatial training split file: {train_split_file}")
            image_paths = self._discover_spatial_image_paths(ds_root, split_file=train_split_file)
        else:
            image_paths = self._discover_spatial_image_paths(ds_root)

        if len(image_paths) == 0:
            raise RuntimeError(f"No spatial fine-tuning images found in {ds_root}")

        self.logger.info(f"Spatial training pool: {len(image_paths)} images")

        backbone = self._load_pretrained_backbone()
        opt = self._build_lobula_plate_optimizer(backbone)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        best_loss = float("inf")

        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            self._configure_lobula_trainability(backbone, unfreeze_top_lobula=False)

            train_ds = self._build_spatial_dataset(
                image_paths=image_paths,
                max_translation=self.spatial_max_shift,
                min_translation=self.min_spatial_shift,
                max_samples=self.train_samples,
                deterministic=False,
                seed=self.run_seed + epoch,
                appearance_transform=self._spatial_appearance_transform(),
            )
            train_dl = DataLoader(
                train_ds,
                **self._loader_kwargs(
                    self.batch_size,
                    shuffle=True,
                    drop_last=len(train_ds) >= self.batch_size,
                ),
            )

            running = self._empty_spatial_metrics()
            processed = 0
            detach_lobula_for_plate = True
            pbar = tqdm(train_dl, desc=f"Epoch {epoch_num}/{self.epochs}", unit="batch")

            for batch in pbar:
                anchor = batch["anchor"].to(self.device, non_blocking=self.device.type == "cuda")
                positive = batch["positive"].to(self.device, non_blocking=self.device.type == "cuda")
                shift = batch["shift"].to(self.device, non_blocking=self.device.type == "cuda")

                opt.zero_grad(set_to_none=True)
                with self._backbone_autocast_context():
                    _, anchor_plate = backbone(anchor, detach_lobula_for_plate=detach_lobula_for_plate)
                    _, positive_plate = backbone(positive, detach_lobula_for_plate=detach_lobula_for_plate)

                    dense_metrics = dense_spatial_contrastive_loss(
                        anchor_plate,
                        positive_plate,
                        shift,
                        temperature=self.dense_temperature,
                        samples_per_image=self.dense_samples,
                    )
                    shift_metrics = centroid_shift_loss(anchor_plate, positive_plate, shift)
                    loss = (
                        self.dense_loss_weight * dense_metrics["loss"]
                        + self.shift_loss_weight * shift_metrics["loss"]
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self._trainable_parameters(backbone), 1.0)
                scaler.step(opt)
                scaler.update()

                batch_size = anchor.size(0)
                processed += batch_size
                running["loss"] += loss.item() * batch_size
                running["dense_loss"] += dense_metrics["loss"].item() * batch_size
                running["shift_loss"] += shift_metrics["loss"].item() * batch_size
                running["dense_top1"] += dense_metrics["top1"].item() * batch_size
                running["positive_cosine"] += dense_metrics["positive_cosine"].item() * batch_size
                running["shift_mae_px"] += shift_metrics["mae_px"].item() * batch_size
                running["overlap_ratio"] += dense_metrics["overlap_ratio"].item() * batch_size
                running["pairs"] += dense_metrics["pairs"]

                pbar.set_postfix(
                    loss=f"{running['loss'] / processed:.4f}",
                    dense_top1=f"{running['dense_top1'] / processed:.3f}",
                    shift_mae=f"{running['shift_mae_px'] / processed:.2f}px",
                    best=f"{best_loss:.4f}" if best_loss < float('inf') else "inf",
                )

            train_metrics = self._finalize_spatial_metrics(running, processed, len(train_dl))

            sched.step()
            best_loss = self._save_checkpoint(backbone, epoch, train_metrics["loss"], best_loss)

            with open(self.outdir / "training_log.txt", "a") as f:
                f.write(
                    f"Epoch {epoch_num}/{self.epochs}, "
                    f"TrainLoss: {train_metrics['loss']:.4f}, TrainDenseLoss: {train_metrics['dense_loss']:.4f}, "
                    f"TrainShiftLoss: {train_metrics['shift_loss']:.4f}, TrainDenseTop1: {train_metrics['dense_top1']:.4f}, "
                    f"TrainPositiveCosine: {train_metrics['positive_cosine']:.4f}, TrainShiftMAE(px): {train_metrics['shift_mae_px']:.4f}, "
                    f"Overlap: {train_metrics['overlap_ratio']:.4f}, Pairs/Batch: {train_metrics['pairs_per_batch']:.1f}, "
                    f"ShiftWeight: {self.shift_loss_weight:.3f}, MaxShift: {self.spatial_max_shift}\n"
                )

            self.logger.info(
                f"Epoch {epoch_num}: dense_top1={train_metrics['dense_top1']:.4f}, "
                f"shift_mae={train_metrics['shift_mae_px']:.3f}px, "
                f"loss={train_metrics['loss']:.4f}"
            )

        self.logger.info(
            "\nLobula plate fine-tuning complete. "
            f"Best loss {best_loss:.4f} → {self.model_path}"
        )
