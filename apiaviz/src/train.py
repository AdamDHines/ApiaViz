'''
This script consists of the training classes for the ApiaNet system.

classes:
    - TrainVision: Trains the VisionModule using the visual synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
    - TrainGustatory: Trains the GustatoryModule using the gustatory synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
    - TrainMotor: Trains the MotorModule using the gustatory synthetic dataset and module to inform flight behaviors to attractive and aversive stimuli.
'''

# Imports
import torch
import random
import secrets

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import apiaviz.src.functional as avf

from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from apiaviz.src.modules import VisionModule
from apiaviz.dataset.datagen import SyntheticDataset, TinyImageNetPairDataset

# Set multiprocessing start method to 'spawn' for compatibility on macOS
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

class TrainVision(nn.Module):
    # ────────── ctor ──────────
    def __init__(self, args):
        super().__init__()

        # expose argparse.Namespace → attributes
        for k in vars(args):
            setattr(self, k, getattr(args, k))

        self.models_dir = Path(self.models_dir)          # <─ comes from args
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / self.vision_model

        # device selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # info / warnings
        if self.device.type == "cpu":
            print('')
            print("========================== WARNING ========================")
            print(".       Training on CPU will be extremely slow. ")
            print(".     Please use a CUDA-enabled GPU or HPC if available.")
            print("  =======================================================  ")
            print('')

    def select_GB(self, chw: torch.Tensor) -> torch.Tensor:
        """Return the G & B channels from a 3-channel tensor."""
        return chw[1:3]
    
    # ────────── main training loop ──────────
    def train(self):
        if self.model_path.exists():
            print(f"Model already exists at {self.model_path}. Overwrite? ((y)/n)")
            ans = input().strip().lower()
            if ans == "n":
                print("Exiting training."); return
            if ans not in ("", "y"):
                print("Invalid input. Exiting training."); return
            print("Continuing training and overwriting existing model.")

        # --- AUGMENTATION PIPELINE ------------------------------------------------
        aug = transforms.Compose([
            # 1. Geometric augmentations - these operate on PIL images.
            transforms.RandomResizedCrop(64, (0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Fix the rotation by filling with a neutral gray (128) instead of black (0).
            transforms.RandomRotation(10, fill=128), 
            
            # 2. Color/Pixel augmentations - also best on PIL images.
            transforms.ColorJitter(hue=.2, saturation=.3, brightness=.3, contrast=.3),
            # Apply blur HERE, before converting to a tensor. The PIL backend handles edges better.
            transforms.GaussianBlur(3, sigma=(.1,2.)),

            # 3. Conversion to Tensor and Channel Selection.
            transforms.ToTensor(),
            transforms.Lambda(self.select_GB),
            avf.MaybeGray2Ch(0.5),
            
            # 4. Normalization - ALWAYS LAST.
            transforms.Normalize([0.5, 0.5], [0.5, 0.5]),
        ])

        # dataset returns (v1, v2)
        if self.train_dataset == "tiny":
            ds_root = "./apiaviz/dataset/tiny-imagenet/train"
            train_ds = TinyImageNetPairDataset(ds_root, transform=aug)
        else:  # synthetic
            train_ds = SyntheticDataset(num_samples=self.train_samples,
                                    image_transform=aug,
                                    green_pct=self.green_pct)
            
        if self.device.type == "mps":
            train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                        shuffle=True, num_workers=0, pin_memory=False)
        else:
            train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)

        # model & optimiser
        seed = secrets.randbits(32)        # e.g. 3857281742

        # 2. seed every RNG you rely on
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 3. (optional) make cudnn deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        model = VisionModule(training=True).to(self.device)
        # check if models folder exists, if not create it
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
        ckpt = f"./apiaviz/models/VisionModel_untrained.pth"
        torch.save(model.state_dict(), ckpt)
        model.train()

        opt   = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=self.epochs  # begin decay after warm-up
                )

        best_loss = float("inf")

        for epoch in range(self.epochs):
            running, processed = 0.0, 0

            pbar = tqdm(train_dl,
                        desc=f"Epoch {epoch+1}/{self.epochs}",
                        unit="batch")

            for v1, v2 in pbar:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                h1, h2 = F.normalize(model(v1), dim=1), F.normalize(model(v2), dim=1)

                loss = avf.nt_xent(h1, h2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                # ---- stats ----
                bs = v1.size(0)
                running   += loss.item() * bs
                processed += bs
                pbar.set_postfix(
                    curr_loss=f"{running/processed:.4f}",
                    best_loss=f"{best_loss:.4f}"
                )

            sched.step()

            epoch_loss = running / processed

            # ---- checkpoint ----
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f"VisionModel.pth")
                print(f"[Epoch {epoch+1}]  ↳ new best {best_loss:.4f} → {self.model_path}")

        print(f"Training complete. Best loss {best_loss:.4f} → {self.model_path}")