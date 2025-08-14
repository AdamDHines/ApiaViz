import torch, os, operator

import numpy as np
import apiaviz.src.functional as avf

from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from apiaviz.src.modules import VisionModule, SNNVisionModule
from apiaviz.dataset.datagen import BalancedEvalVisionDataset, InsectVisionDataset, DataMode

class EvalVision:
    """Evaluate a VisionModule or SNNVisionModule encoder with several quantitative metrics and visualization.
    """
    def __init__(self, args):
        for k in vars(args): setattr(self, k, getattr(args, k))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.outdir = Path(self.outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        
        # SNN mode requires specific image properties for on-the-fly conversion
        self.full_image_size = 256 

        # --- Model Loading (Now conditional) ---
        if self.snn:
            model_path = os.path.join(self.models_dir, f"{self.snn_vision_model}.pth")
        else:
            model_path = os.path.join(self.models_dir, f"{self.vision_model}.pth")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        
        if getattr(self, 'snn', False):
            print("Loading SNNVisionModule for evaluation.")
            self.model = SNNVisionModule().to(self.device)
            # Store SNN parameters for conversion
            self.num_steps = getattr(self, 'num_steps', 50)
            self.patch_size = getattr(self, 'patch_size', 28)
            self.scan_method = getattr(self, 'scan_method', 'saccade')
            self.scan_waypoints = getattr(self, 'scan_waypoints', 5)
        else:
            print("Loading VisionModule for evaluation.")
            self.model = VisionModule().to(self.device)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # --- Dataset Loading ---
        if not self.scanning and not self.patching:
            self.dataset = InsectVisionDataset(
                root='./apiaviz/dataset/',
                dataset=self.eval_dataset,
                mode=DataMode.STATIC_FULL,
                patch_size=self.patch_size,
                samples_per_image=self.eval_samples
            )
        elif self.scanning and not self.patching:
            print("Using SCANNING_PATCH mode for evaluation.")
            self.dataset = InsectVisionDataset(
                root='./apiaviz/dataset/',
                dataset=self.eval_dataset,
                mode=DataMode.SCANNING_PATCH,
                patch_size=self.patch_size,
                samples_per_image=self.eval_samples,
                num_steps=self.num_steps
            )
        else:
            self.dataset = InsectVisionDataset(
                    root='./apiaviz/dataset/',
                    dataset=self.eval_dataset,
                    mode=DataMode.STATIC_PATCH,
                    patch_size=75,
                    samples_per_image=300
                )
            
        # --- DataLoader ---
        if self.device.type == "mps":
            self.loader = DataLoader(self.dataset,
                            batch_size=self.eval_batch_size, shuffle=False, num_workers=0)
        else:
            self.loader = DataLoader(self.dataset,
                                    batch_size=self.eval_batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
            
        # --- Evaluator ---
        self.evaluator = avf.ModelEvaluator(self.model, self.device, output_dir=self.outdir)

    def bernoulli_spikes(self, x: torch.Tensor, rate_scale: float = 1.5) -> torch.Tensor:
        """Converts a rate-coded tensor to Bernoulli spikes."""
        p = (x * rate_scale).clamp(0, 1) # Use clamp, not clamp_ for wider compatibility
        return (torch.rand_like(p) < p).float()
    
    def _create_spike_train(self, static_tensor) -> torch.Tensor:
        """Generates a spike train from a static tensor using a pre-defined path."""
        frames = []
        for idx in range(self.num_steps):
            frames.append(self.bernoulli_spikes(static_tensor[:,idx]))
        return torch.stack(frames, dim=0)

    def eval(self):
        """
        Refactored feature extraction that correctly handles all four modes:
        - ANN Static & Scanning
        - SNN Static & Scanning
        """
        # Determine the data mode from the dataset attached to the loader
        is_scanning_mode = (getattr(self.loader.dataset, 'mode', None) == DataMode.SCANNING_PATCH)
        
        # SNN models might have num_steps defined in their params
        num_steps = getattr(self, 'num_steps', 25) 

        feats, labs = [], []
        with torch.no_grad():
            for imgs, lbl, _, _ in tqdm(self.loader, desc="Extracting Features", unit="batch"):
                imgs = imgs.to(self.device)
                
                # --- SNN Feature Extraction ---
                if getattr(self, 'snn', False):
                    spiked_input_frames = []
                    
                    # Check if input is a time-series (Batch, Time, C, H, W)
                    if imgs.dim() == 5 and is_scanning_mode:
                        # Case 1: SNN with SCANNING data.
                        # Input is already a time-series of patches.
                        # We just need to convert each patch to spikes.
                        for t in range(imgs.shape[1]):
                            frame = imgs[:, t, :, :, :]
                            spiked_input_frames.append(self.bernoulli_spikes(frame))
                    else:
                        # Case 2: SNN with STATIC data.
                        # Input is a single static patch/image.
                        # We must present it repeatedly to the SNN over `num_steps`.
                        for _ in range(num_steps):
                            spiked_input_frames.append(self.bernoulli_spikes(imgs))
                    
                    # Stack frames to create the final SNN input tensor
                    # Shape: (time, batch, channels, H, W)
                    snn_input = torch.stack(spiked_input_frames, dim=0)
                    
                    output_spikes = self.model(snn_input, num_steps=num_steps)
                    z = output_spikes.sum(dim=0) # Rate coding: sum spikes over time

                # --- ANN Feature Extraction ---
                else:
                    if is_scanning_mode:
                        # Case 3: ANN with SCANNING data.
                        # Iterate through the time-series and feed each patch to the model.
                        z_feats = [self.model(imgs[:, n]) for n in range(imgs.shape[1])]
                        z = torch.mean(torch.stack(z_feats, dim=1), dim=1)
                    else:
                        # Case 4: ANN with STATIC data.
                        # A single forward pass is sufficient.
                        z = self.model(imgs)

                feats.append(z.cpu().numpy())
                labs.append(lbl.numpy())

        # Concatenate results from all batches
        feats = np.concatenate(feats)
        labs = np.concatenate(labs)
        
        # --- Run evaluation (Unchanged) ---
        snn_params_for_evaluator = None
        if getattr(self, 'snn', False):
            # Collect all the SNN parameters into a dictionary
            snn_params_for_evaluator = {
                'num_steps': self.num_steps,
                'patch_size': self.patch_size,
                'full_image_size': self.full_image_size,
                'scan_method': self.scan_method,
                'scan_waypoints': self.scan_waypoints
            }

        self.evaluator = avf.ModelEvaluator(self.model, self.device, output_dir=self.outdir, snn_params=snn_params_for_evaluator)
        self.evaluator.run_full_evaluation(self.loader, feats, labs, is_scanning=self.scanning)