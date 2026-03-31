import torch
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from apiaviz.src.metrics import recallAtK
from prettytable import PrettyTable
from apiaviz.src.create_GTtol import create_GTtol
from apiaviz.src.sad import run_sad

# Assuming apiaviz is in the project structure
# from apiaviz.src.modules import VisionModule, SNNVisionModule

class GardensPoint:
    """
    A class to perform visual place recognition on the Gardens Point dataset
    using a pre-trained VisionModule (ANN) or SNNVisionModule (spiking).
    """
    def __init__(self, dataset_dir='./apiaviz/dataset/gardens-point', models_dir='./apiaviz/models',
                 model_name='SNNVisionModelNew_epoch_20.pth', spiking=True, timesteps=25, rate_scale=1.5):
        """
        Initializes the GardensPoint class.

        Args:
            dataset_dir (str): The directory containing the Gardens Point dataset.
            models_dir (str): The directory where the pre-trained models are stored.
            model_name (str): The filename of the pre-trained VisionModule.
            spiking (bool): If True, use SNNVisionModule and generate spike trains.
            timesteps (int): Number of SNN timesteps to simulate.
            rate_scale (float): Multiplier for Bernoulli spike probability.
        """
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.spiking = spiking
        self.timesteps = timesteps
        self.rate_scale = rate_scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Model Loading ---
        model_path = self.models_dir / self.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        if spiking:
            self.model = SNNVisionModule().to(self.device)
        else:
            self.model = VisionModule().to(self.device)

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print(("SNN" if self.spiking else "ANN") + " VisionModule loaded successfully.")

        # --- Image Transformations (values in [0,1]) ---
        self.transform = transforms.Compose([
            transforms.Resize((75, 75)),
            transforms.ToTensor()
        ])

    def _load_and_preprocess_images(self, traverse_name):
        """
        Loads and preprocesses images from a specified traverse.

        Args:
            traverse_name (str): The name of the traverse subfolder (e.g., 'day_left').

        Returns:
            torch.Tensor: A tensor of preprocessed images [N, 3, 75, 75].
        """
        traverse_path = self.dataset_dir / traverse_name
        if not traverse_path.exists():
            raise FileNotFoundError(f"Traverse folder not found at: {traverse_path}")

        # Get all .jpg images and sort them numerically
        image_files = sorted(glob.glob(os.path.join(traverse_path, '*.jpg')),
                             key=lambda f: int(''.join(filter(str.isdigit, f))))

        images = []
        for img_path in tqdm(image_files, desc=f"Loading and preprocessing {traverse_name}"):
            img = Image.open(img_path).convert('RGB')
            images.append(self.transform(img))

        return torch.stack(images) if images else torch.empty(0, 3, 75, 75)

    # --- NEW: simple Bernoulli spike train from static GB tensor in [0,1] ---
    def _to_spike_train(self, gb_tensor_2ch: torch.Tensor) -> torch.Tensor:
        """
        Convert a static 2-channel [2, H, W] tensor in [0,1] to a Bernoulli spike train.

        Returns:
            spikes [T, 1, 2, H, W] float tensor on self.device
        """
        x = gb_tensor_2ch.clamp(0, 1).to(self.device)                  # [2, H, W]
        p = (x * self.rate_scale).clamp(0, 1)                          # [2, H, W]
        rand = torch.rand((self.timesteps,) + p.shape, device=self.device)  # [T, 2, H, W]
        spikes = (rand < p).float()                                    # [T, 2, H, W]
        return spikes.unsqueeze(1)                                      # [T, 1, 2, H, W]

    def _extract_features(self, images):
        """
        Extracts features from a batch of images using the VisionModule or SNNVisionModule.

        Args:
            images (torch.Tensor): A tensor of images [N, 3, 75, 75].

        Returns:
            torch.Tensor: A tensor of extracted features [N, D].
        """
        features = []
        if images.numel() == 0:
            return torch.empty(0, 0)

        with torch.no_grad():
            if not self.spiking:
                # --- ANN path (unchanged) ---
                for img in tqdm(images, desc="Extracting features (ANN)"):
                    img = img.unsqueeze(0).to(self.device)             # [1, 3, 75, 75]
                    gb = img[:, 1:3, :, :]                             # [1, 2, 75, 75]
                    feat = self.model(gb)                              # [1, D] (expected)
                    features.append(feat.detach().cpu())
            else:
                # --- SNN path: generate T-step spikes and forward ---
                for img in tqdm(images, desc=f"Extracting features (SNN, T={self.timesteps})"):
                    gb = img[1:3, :, :]                                # [2, 75, 75] in [0,1]
                    spikes_TBCHW = self._to_spike_train(gb)            # [T, 1, 2, 75, 75]

                    # Try [T, B, C, H, W] first
                    try:
                        feat = self.model(spikes_TBCHW, self.timesteps)                # -> could be [T, 1, D] or [1, D]
                    except Exception:
                        # Fallback to [B, T, C, H, W]
                        feat = self.model(spikes_TBCHW.permute(1, 0, 2, 3, 4), self.timesteps)  # [1, T, 2, 75, 75]

                    # Aggregate over time if the model returns per-timestep features
                    if feat.ndim == 3 and feat.size(0) == self.timesteps:
                        feat = feat.mean(dim=0)                        # [1, D]
                    features.append(feat.detach().cpu())

        return torch.cat(features, dim=0)

    def run_vpr(self, query_traverse, reference_traverse):
        """
        Runs the Visual Place Recognition pipeline.

        Args:
            query_traverse (str): The name of the query traverse.
            reference_traverse (str): The name of the reference traverse.
        """
        # 1. & 2. Load and preprocess images
        query_images = self._load_and_preprocess_images(query_traverse)
        ref_images   = self._load_and_preprocess_images(reference_traverse)

        # 3. Extract features
        query_features = self._extract_features(query_images)
        ref_features   = self._extract_features(ref_images)

        # 4. Compare features using cosine similarity and plot
        # Normalize the feature vectors
        query_features = F.normalize(query_features, p=2, dim=1)
        ref_features   = F.normalize(ref_features, p=2, dim=1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(query_features, ref_features.T)

        # Plotting the similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix.numpy(), cmap='viridis', interpolation='nearest', aspect='auto')
        plt.title(f'Cosine Similarity: {query_traverse} (Query) vs. {reference_traverse} (Reference)\n'
                  f'{"SNN T="+str(self.timesteps) if self.spiking else "ANN"}')
        plt.xlabel('Reference Image Index')
        plt.ylabel('Query Image Index')
        plt.colorbar(label='Cosine Similarity')
        plt.tight_layout()
        plt.show()

        # 5. Recall@K (with GT tolerance band)
        gt = np.eye(len(query_features), len(ref_features))
        gt = create_GTtol(gt, distance=1)
        N = [1, 5, 10, 15, 20, 25]

        R = [round(recallAtK(similarity_matrix, gt, K=n), 2) for n in N]

        table = PrettyTable()
        table.field_names = ["N"] + list(map(str, N))
        table.add_row(["Recall"] + R)

        # 6. Compare to SAD
        sad_recall = run_sad(gt, query_images, ref_images)
        table.add_row(["SAD Recall"] + [sad_recall[i] for i in range(len(N))])
        print(table)
