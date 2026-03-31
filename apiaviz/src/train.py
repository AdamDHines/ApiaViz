# Imports
import torch, random, secrets, time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import apiaviz.src.functional as avf

from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from apiaviz.dataset.datagen import TinyImageNetPairDataset
from apiaviz.src.modules import VisionBackbone

# Set multiprocessing start method to 'spawn' for compatibility on macOS
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

class TrainVision(nn.Module):
    # ────────── ctor ──────────
    def __init__(self, args, logger, outdir):
        super().__init__()

        for k in vars(args):
            setattr(self, k, getattr(args, k))

        self.models_dir = Path(self.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if self.snn:
            self.model_path = Path(f"{self.models_dir}/{self.snn_vision_model}.pth")
        else:
            self.model_path = Path(f"{self.models_dir}/{self.vision_model}.pth")

        self.logger = logger
        self.outdir = Path(outdir)

        # device selection (no changes needed here)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.device.type == "cpu":
            self.logger.info("\n========================== WARNING ========================\n"
                  ".       Training on CPU will be extremely slow. \n"
                  ".     Please use a CUDA-enabled GPU or HPC if available.\n"
                  "  =======================================================  \n")

    def select_GB(self, chw: torch.Tensor) -> torch.Tensor:
        """Return the G & B channels from a 3-channel tensor."""
        return chw[1:3]
    
    # ────────── main training loop ──────────
    def train(self):
        if self.model_path.exists():
            self.logger.info(f"A model directory already exists at {self.models_dir}. Overwrite? ((y)/n)")
            ans = input().strip().lower()
            if ans == "n":
                self.logger.info("Exiting training."); return
            if ans not in ("", "y"):
                self.logger.info("Invalid input. Exiting training."); return
            self.logger.info("Continuing training and overwriting existing models and logs.")

        # --- AUGMENTATION PIPELINE -----------------------------
        self.full_image_size = 64 

        feature_aug = transforms.Compose([
            transforms.RandomResizedCrop(64, (0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, fill=128),
            transforms.ColorJitter(hue=.2, saturation=.3, brightness=.3, contrast=.3),
            transforms.GaussianBlur(3, sigma=(.1,2.)),
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

        ds_root = f"{self.dataset_dir}/{self.training_dataset}"
        # append the train directory if using Tiny ImageNet
        if self.training_dataset == "tiny-imagenet": 
            ds_root = f"{ds_root}/train"

        # ANN mode: standard instantiation
        train_ds = TinyImageNetPairDataset(ds_root, feature_transform=feature_aug, spatial_transform=spatial_aug)
            
        train_dl = DataLoader(train_ds, batch_size=self.batch_size)

        # --- MODEL, SEEDING, and OPTIMIZER (minor changes) -------------------------
        seed = secrets.randbits(32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        backbone = VisionBackbone().to(self.device)

        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        backbone.train()

        opt = torch.optim.AdamW(backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best_loss = float("inf")

        # define an MLP for the projection head (if needed by the loss function)
        mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(self.device)

        # --- Pre-training VisionBackbone
        for epoch in range(self.epochs):
            running, processed = 0.0, 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
            scaler = torch.amp.GradScaler()
            for v1, v2 in pbar:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                opt.zero_grad(set_to_none=True)

                # Original ANN forward pass
                h1, _ = backbone(v1)
                h2, _ = backbone(v2)

                # Run through MLP
                h1 = F.normalize(mlp(h1), dim=1)
                h2 = F.normalize(mlp(h2), dim=1)

                # The loss calculation remains the same, as we've prepared `h1` and `h2`
                loss = avf.nt_xent(h1, h2)

                # The backward pass and optimization step are also the same.
                # For the SNN, this is where Backpropagation Through Time (BPTT) happens.
                # PyTorch handles the gradient flow back through all `num_steps`.
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                # ---- stats (no changes needed) ----
                bs = v1.size(0)
                running += loss.item() * bs
                processed += bs
                pbar.set_postfix(
                    curr_loss=f"{running/processed:.4f}",
                    best_loss=f"{best_loss:.4f}"
                )

            sched.step()
            epoch_loss = running / processed

            # Store the epoch_loss into a log file
            with open(self.outdir / "training_log.txt", "a") as f:
                f.write(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}\n")

            # ---- checkpoint (modified to save best model separately) ----
            if self.best_only:
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), str(self.model_path))
                    self.logger.debug(f"\n[Epoch {epoch+1}] ↳ New best loss {best_loss:.4f} → {self.model_path}")
            else:
                # Print a confirmation for the per-epoch model save
                if self.snn:
                    new_path = Path(f"{self.models_dir}/{self.snn_vision_model}_Epoch{epoch+1}.pth")
                else:
                    new_path = Path(f"{self.models_dir}/{self.vision_model}_Epoch{epoch+1}.pth")
                torch.save(model.state_dict(), str(new_path))
                self.logger.debug(f"\n[Epoch {epoch+1}] ↳ Model saved to {new_path}")

        self.logger.info(f"\nTraining complete. Best loss {best_loss:.4f} → {self.model_path}")