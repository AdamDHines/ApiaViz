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
from apiaviz.src.modules import VisionModule, SNNVisionModule

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
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
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
        if self.snn:
            aug = transforms.Compose([
                transforms.RandomResizedCrop(64, (0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10, fill=128),
                transforms.ColorJitter(hue=.2, saturation=.3, brightness=.3, contrast=.3),
                transforms.GaussianBlur(3, sigma=(.1,2.)),
                transforms.ToTensor(),
                transforms.Lambda(self.select_GB),              
                avf.MaybeGray2Ch(0.5)
            ])
        else:
            aug = transforms.Compose([
                transforms.RandomResizedCrop(64, (0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10, fill=128),
                transforms.ColorJitter(hue=.2, saturation=.3, brightness=.3, contrast=.3),
                transforms.GaussianBlur(3, sigma=(.1,2.)),
                transforms.ToTensor(),
                transforms.Lambda(self.select_GB),
                avf.MaybeGray2Ch(0.5),
                transforms.Normalize([0.5, 0.5], [0.5, 0.5]),
            ])

        ds_root = "./apiaviz/dataset/tiny-imagenet/train"
        if self.snn:
            # SNN mode: pass snn_mode=True and num_steps to the dataset
            train_ds = TinyImageNetPairDataset(
                ds_root, 
                transform=aug, 
                snn_mode=True,
                num_steps=self.num_steps
            )
        else:
            # ANN mode: standard instantiation
            train_ds = TinyImageNetPairDataset(ds_root, 
                                            transform=aug, 
                                            snn_mode=False)
            
        train_dl = DataLoader(train_ds, batch_size=self.batch_size)

        # --- MODEL, SEEDING, and OPTIMIZER (minor changes) -------------------------
        seed = secrets.randbits(32)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if self.snn:
            # Pass any necessary SNN-specific args here
            model = SNNVisionModule().to(self.device)
        else:
            model = VisionModule().to(self.device)

        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # This saving logic is fine.
        ckpt = f"./apiaviz/models/{self.vision_model}_untrained.pth"
        torch.save(model.state_dict(), ckpt)
        model.train()

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best_loss = float("inf")

        # --- CORE TRAINING LOOP (key modifications here) --------------------------
        for epoch in range(self.epochs):
            # MOD: Record the start time of the epoch
            epoch_start_time = time.time()
            
            running, processed = 0.0, 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{self.epochs}", unit="batch")
            scaler = torch.amp.GradScaler()
            for v1, v2 in pbar:
                v1, v2 = v1.to(self.device), v2.to(self.device)
                opt.zero_grad(set_to_none=True)
                if self.snn:
                    # 1. Pass the input and the number of time steps to the model.
                    # The output will be spikes over time: (num_steps, batch_size, features)
                    v1 = v1.permute(1, 0, 2, 3, 4)
                    v2 = v2.permute(1, 0, 2, 3, 4)
                    spk_h1 = model(v1, num_steps=self.num_steps)
                    spk_h2 = model(v2, num_steps=self.num_steps)

                    # 2. Decode the spike train into a feature vector (Rate Coding).
                    h1 = spk_h1.mean(dim=0)
                    h2 = spk_h2.mean(dim=0)
                    
                    # 3. Normalize the spike counts.
                    h1 = F.normalize(h1, dim=1)
                    h2 = F.normalize(h2, dim=1)

                else:
                    # Original ANN forward pass
                    h1 = F.normalize(model(v1), dim=1)
                    h2 = F.normalize(model(v2), dim=1)
                
                # The loss calculation remains the same, as we've prepared `h1` and `h2`
                loss = avf.nt_xent(h1, h2)

                # The backward pass and optimization step are also the same.
                # For the SNN, this is where Backpropagation Through Time (BPTT) happens.
                # PyTorch handles the gradient flow back through all `num_steps`.
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            
            # MOD: Calculate epoch duration
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # MOD: Log epoch stats to the text file
            with open(self.log_path, 'a') as f:
                log_entry = (f"Epoch: {epoch+1:03d} | "
                             f"Loss: {epoch_loss:.4f} | "
                             f"Duration: {epoch_duration:.2f}s\n")
                f.write(log_entry)
            
            # MOD: Define a unique path for the model of this specific epoch
            epoch_model_path = self.models_dir / f"{self.vision_model}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), str(epoch_model_path))

            # ---- checkpoint (modified to save best model separately) ----
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), str(self.model_path))
                self.logger.debug(f"\n[Epoch {epoch+1}] ↳ New best loss {best_loss:.4f} → {self.model_path}")
            else:
                # Print a confirmation for the per-epoch model save
                self.logger.debug(f"\n[Epoch {epoch+1}] ↳ Epoch model saved → {epoch_model_path}")


        self.logger.info(f"\nTraining complete. Best loss {best_loss:.4f} → {self.model_path}")