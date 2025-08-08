import torch
import numpy as np
from pathlib import Path
import os
import operator
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# Make sure these imports are correct for your project structure
from apiaviz.src.modules import VisionModule, SNNVisionModule
from apiaviz.dataset.datagen import (
    BalancedEvalVisionDataset,
    FlowerPatchDataset,
    FacePatchDataset,
    VarietyDataset
)

import apiaviz.src.functional as avf

# --- Add the self-contained, robust path generation function ---
def generate_smooth_scan_path(num_steps, max_coord, method='saccade', num_waypoints=5):
    """
    Generates a 2D smooth scanning path for an image patch.
    This version guarantees the correct output length.
    """
    if method == 'saccade':
        waypoints_x = np.random.randint(0, max_coord + 1, num_waypoints)
        waypoints_y = np.random.randint(0, max_coord + 1, num_waypoints)
        control_points = np.linspace(0, num_steps - 1, num_waypoints)
        full_timeline = np.arange(num_steps)
        path_x = np.interp(full_timeline, control_points, waypoints_x)
        path_y = np.interp(full_timeline, control_points, waypoints_y)
        return torch.from_numpy(path_x.astype(np.int32)), torch.from_numpy(path_y.astype(np.int32))
    elif method == 'lissajous':
        center = max_coord / 2.0
        freq_x, freq_y = np.random.uniform(1.0, 3.0, 2)
        phase = np.random.uniform(0, np.pi)
        t = torch.linspace(0, 2 * np.pi, num_steps)
        x = torch.sin(freq_x * t + phase); y = torch.cos(freq_y * t)
        path_x = ((x * center) + center).to(torch.int32)
        path_y = ((y * center) + center).to(torch.int32)
        return path_x, path_y
    else:
        raise ValueError(f"Unknown scan_method '{method}'.")


class EvalVision:
    """Evaluate a VisionModule or SNNVisionModule encoder with several quantitative metrics and visualization.
    """
    def __init__(self, args):
        for k in vars(args): setattr(self, k, getattr(args, k))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.outdir = Path(self.outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        
        # SNN mode requires specific image properties for on-the-fly conversion
        self.full_image_size = 64 

        # --- Model Loading (Now conditional) ---
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

        # --- Image transform (used by synthetic dataset & SNN converter) ---
        # This transform is explicitly passed to datasets that need it.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(operator.itemgetter(slice(1, 3))),  # keep G,B
            transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

        # --- Dataset Loading (EXACTLY AS YOU PROVIDED) ---
        if self.eval_dataset == "synthetic":
            self.dataset = BalancedEvalVisionDataset(
                num_samples=self.eval_samples,
                image_transform=self.transform,
                green_pct_high=self.green_pct_high,
                green_pct_low=self.green_pct_low
            )
        elif self.eval_dataset == "natural-scenes":
            self.dataset = FlowerPatchDataset(patches_per_file=self.eval_samples)
        elif self.eval_dataset == "variety":
            self.dataset = VarietyDataset(patches_per_file=self.eval_samples)
        elif self.eval_dataset == "faces":
            self.dataset = FacePatchDataset("./apiaviz/dataset/faces/", patches_per_file=self.eval_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.eval_dataset}")
        
        # --- DataLoader (Unchanged) ---
        if self.device.type == "mps":
            self.loader = DataLoader(self.dataset,
                            batch_size=self.eval_batch_size, shuffle=False, num_workers=0)
        else:
            self.loader = DataLoader(self.dataset,
                                    batch_size=self.eval_batch_size, shuffle=False,
                                    num_workers=4, pin_memory=True)
            
        # --- Evaluator (Unchanged) ---
        self.evaluator = avf.ModelEvaluator(self.model, self.device, output_dir=self.outdir)

    def _convert_static_to_spiking_batch(self, static_batch):
        """
        Helper function to convert a batch of static images to spike trains.
        This ensures all data is consistently sized and normalized before spiking.
        """
        # 1. Ensure consistent size
        resize_transform = transforms.Resize((self.full_image_size, self.full_image_size), antialias=True)
        resized_batch = resize_transform(static_batch)
        
        # 2. Ensure consistent normalization to [-1, 1] for probability conversion
        # This assumes input tensors are in [0, 1] range from ToTensor().
        # If they are already normalized, this will be roughly idempotent.
        norm_transform = transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        norm_batch = norm_transform(resized_batch)

        # 3. Convert to probability space [0, 1]
        prob_batch = (norm_batch + 1.0) / 2.0
        max_coord = self.full_image_size - self.patch_size
        
        # 4. Generate one shared path for the entire batch
        path_x, path_y = generate_smooth_scan_path(
            self.num_steps, max_coord, self.scan_method, self.scan_waypoints
        )
        path_x, path_y = path_x.to(self.device), path_y.to(self.device)

        # 5. Create spike frames
        batch_spike_frames = []
        for t in range(self.num_steps):
            x, y = path_x[t], path_y[t]
            patch_prob = prob_batch[:, :, y:y+self.patch_size, x:x+self.patch_size]
            spike_frame = (torch.rand_like(patch_prob) < patch_prob).float()
            batch_spike_frames.append(spike_frame)
        
        return torch.stack(batch_spike_frames, dim=1) # -> [B, T, C, h, w]

    def eval(self):
        feats, labs = [], []
        with torch.no_grad():
            for imgs, lbl in tqdm(self.loader, desc="Extracting Features", unit="batch"):
                imgs = imgs.to(self.device)

                if getattr(self, 'snn', False):
                    # --- SNN-specific logic ---
                    spk_imgs_batch = self._convert_static_to_spiking_batch(imgs)
                    spk_imgs_batch = spk_imgs_batch.permute(1, 0, 2, 3, 4) # -> [T, B, C, H, W]
                    output_spikes = self.model(spk_imgs_batch, num_steps=self.num_steps)
                    z = output_spikes.sum(dim=0) # Rate coding
                else:
                    # --- Original ANN logic ---
                    z = self.model(imgs)

                feats.append(z.cpu().numpy())
                labs.append(lbl.numpy())

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

        self.evaluator = avf.ModelEvaluator(
            self.model, 
            self.device, 
            output_dir=self.outdir,
            snn_params=snn_params_for_evaluator # Pass the dictionary here
        )

        self.evaluator.run_full_evaluation(self.loader, 
                                   feats, 
                                   labs, 
                                   n_visualization_samples=self.n_eval_plots, 
                                   use_random_sampling=self.eval_plot_random,
                                   ind_plot=self.ind_plots
)