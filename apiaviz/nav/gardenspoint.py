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
from apiaviz.src.modules import VisionModule

class GardensPoint:
    """
    A class to perform visual place recognition on the Gardens Point dataset
    using a pre-trained VisionModule.
    """
    def __init__(self, dataset_dir='./apiaviz/dataset/gardens-point', models_dir='./apiaviz/models', model_name='VisionModel_NoLam_epoch_8.pth'):
        """
        Initializes the GardensPoints class.

        Args:
            dataset_dir (str): The directory containing the Gardens Point dataset.
            models_dir (str): The directory where the pre-trained models are stored.
            model_name (str): The filename of the pre-trained VisionModule.
        """
        self.dataset_dir = Path(dataset_dir)
        self.models_dir = Path(models_dir)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Model Loading ---
        model_path = self.models_dir / self.model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model = VisionModule().to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        print("VisionModule loaded successfully.")

        # --- Image Transformations ---
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
            torch.Tensor: A tensor of preprocessed images.
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
            
        return torch.stack(images)

    def _extract_features(self, images):
        """
        Extracts features from a batch of images using the VisionModule.

        Args:
            images (torch.Tensor): A tensor of images.

        Returns:
            torch.Tensor: A tensor of extracted features.
        """
        features = []
        with torch.no_grad():
            for img in tqdm(images, desc="Extracting features"):
                img = img.unsqueeze(0).to(self.device)  # Add batch dimension
                # Select only the green and blue channels (index 1 and 2)
                gb_channels = img[:, 1:3, :, :]
                feature = self.model(gb_channels)
                features.append(feature.cpu())
        
        return torch.cat(features)

    def run_vpr(self, query_traverse, reference_traverse):
        """
        Runs the Visual Place Recognition pipeline.

        Args:
            query_traverse (str): The name of the query traverse.
            reference_traverse (str): The name of the reference traverse.
        """
        # 1. & 2. Load and preprocess images
        query_images = self._load_and_preprocess_images(query_traverse)
        ref_images = self._load_and_preprocess_images(reference_traverse)

        # 3. Extract features
        query_features = self._extract_features(query_images)
        ref_features = self._extract_features(ref_images)

        # 4. Compare features using cosine similarity and plot
        # Normalize the feature vectors
        query_features = F.normalize(query_features, p=2, dim=1)
        ref_features = F.normalize(ref_features, p=2, dim=1)

        # Compute cosine similarity
        similarity_matrix = torch.matmul(query_features, ref_features.T)

        # Plotting the similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix.numpy(), cmap='viridis', interpolation='nearest')
        plt.title(f'Cosine Similarity: {query_traverse} (Query) vs. {reference_traverse} (Reference)')
        plt.xlabel('Reference Image Index')
        plt.ylabel('Query Image Index')
        plt.colorbar(label='Cosine Similarity')
        plt.show()

        gt = np.eye(len(query_features), len(ref_features))
        gt = create_GTtol(gt, distance=1)
        N = [1,5,10,15,20,25] # N values to calculate
        # Calculate Recall@N
        R = []
        for n in N:
            R.append(round(recallAtK(similarity_matrix,gt,K=n),2))

        # Print the results

        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])


        # compare to sad
        sad_recall = run_sad(gt, query_images, ref_images)
        table.add_row(["SAD Recall", sad_recall[0], sad_recall[1], sad_recall[2], sad_recall[3], sad_recall[4], sad_recall[5]])
        print(table)