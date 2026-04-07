# Imports
import random
import secrets

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import apiaviz.src.functional as avf

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from apiaviz.dataset.datagen import (
    DenseSpatialPairDataset,
    TinyImageNetPairDataset,
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
from apiaviz.src.spatial_finetune import (
    centroid_shift_loss,
    descriptor_place_matching_loss,
    dense_spatial_contrastive_loss,
    PoseRelationHead,
    relative_pose_regression_loss,
    resolve_pretrained_checkpoint,
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
        elif self.snn:
            self.model_path = self.models_dir / f"{self.snn_vision_model}.pth"
        else:
            self.model_path = self.models_dir / f"{self.vision_model}.pth"

        self.logger = logger
        self.outdir = Path(outdir)

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
