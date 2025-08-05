# Imports
import torch
import operator

import numpy as np
import apiaviz.src.functional as avf

from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from apiaviz.src.modules import VisionModule
from apiaviz.dataset.datagen import FacePatchDataset, FlowerPatchDataset, BalancedEvalVisionDataset

class EvalVision:
    """Evaluate a VisionModule encoder with several quantitative metrics and visualization.
    """

    def __init__(self, args):
        # stash config
        for k in vars(args):
            setattr(self, k, getattr(args, k))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # output dir
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # ---------------------------------------------------------------
        #  Model
        # ---------------------------------------------------------------
        state_dict = torch.load("./apianet/models/VisionModel.pth", map_location=self.device, weights_only=True)
        self.model = VisionModule().to(self.device)
        self.model.load_state_dict(state_dict['encoder'], strict=False)
        self.model.eval()

        # ---------------------------------------------------------------
        #  Image transform (2‑channel G,B normalised to [-1,1])
        # ---------------------------------------------------------------
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(operator.itemgetter(slice(1, 3))),  # keep G,B
            transforms.Normalize([0.5, 0.5], [0.5, 0.5])
        ])

        # ---------------------------------------------------------------
        #  Dataset
        # ---------------------------------------------------------------
        if self.eval_dataset == "synthetic":
            self.dataset = BalancedEvalVisionDataset(
                num_samples=self.eval_samples,
                image_transform=self.transform,
                green_pct_high=self.green_pct_high,
                green_pct_low=self.green_pct_low
            )
        elif self.eval_dataset == "flowers":
            self.dataset = FlowerPatchDataset(patches_per_file=self.eval_samples)
        elif self.eval_dataset == "faces":
            self.dataset = FacePatchDataset("./apiaviz/dataset/faces/", patches_per_file=self.eval_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.eval_dataset}")
        
        if self.device.type == "mps":
            self.loader = DataLoader(self.dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=0)
        else:
            self.loader = DataLoader(self.dataset,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True)
            
        # define evaluator
        self.evaluator = avf.ModelEvaluator(self.model, self.device)
    
    def eval(self):
        feats, labs = [], []
        with torch.no_grad():
            for imgs, lbl in tqdm(self.loader, desc="Extracting", unit="batch"):
                z = self.model(imgs.to(self.device)).cpu().numpy()
                feats.append(z)
                labs.append(lbl.numpy())

        feats = np.concatenate(feats)
        labs = np.concatenate(labs)