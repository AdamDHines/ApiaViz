# Imports
import math, torch, cv2, random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path
from matplotlib import cm
from typing import Optional
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from typing import Optional, List, Dict, Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score

import warnings
warnings.filterwarnings('ignore')

# ────────── k-Winner Takes All functions ──────────
class AdaptiveKWTA(nn.Module):
    def __init__(self, sparsity=0.05, momentum=0.9):
        super().__init__()
        self.sparsity = sparsity
        self.momentum = momentum
        self.register_buffer('running_mean', None)
        self.register_buffer('thresholds', None)
        
    def forward(self, x):
        if self.running_mean is None:
            self.running_mean = torch.zeros(x.size(1), device=x.device)
            self.thresholds = torch.ones(x.size(1), device=x.device)
        
        # Update running statistics
        with torch.no_grad():
            batch_mean = (x > 0).float().mean(dim=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            
            # Increase threshold for frequently active neurons
            self.thresholds = 1.0 + 2.0 * (self.running_mean - self.sparsity).clamp(min=0)
        
        # Apply adaptive thresholds
        x_adjusted = x / self.thresholds.unsqueeze(0)
        
        # Standard k-WTA on adjusted values
        k = max(1, int(self.sparsity * x.size(1)))
        topk, idx = torch.topk(x_adjusted, k, dim=1)
        mask = torch.zeros_like(x).scatter_(1, idx, 1.0)
        
        return x * mask
    
def k_wta(x, pct=.05):
    k = max(1, int(pct * x.size(1)))
    topk, idx = torch.topk(x, k, dim=1)
    mask = torch.zeros_like(x).scatter_(1, idx, 1.0)
    y = x * mask        # forward: hard sparsity
    # backward: pretend mask is constant
    return (y - x).detach() + x

# ────────── Augmentation helpers ──────────

class MaybeGray2Ch:                          # 50 % colour-drop
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, gb: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            g = gb.mean(0, keepdim=True)
            return torch.cat([g, g], dim=0)
        return gb

# ────────── Sparse linear function ──────────

class SparseLinear(nn.Module):
    def __init__(self, in_f, out_f, fan_in=7, bias=False):
        super().__init__()
        mask = torch.zeros(out_f, in_f, dtype=torch.bool)
        for o in range(out_f):
            mask[o, torch.randperm(in_f)[:fan_in]] = True
        self.register_buffer("mask", mask)

        # learnable weights only where mask == 1
        w = torch.zeros(out_f, in_f)
        w[self.mask] = torch.randn(self.mask.sum()) / math.sqrt(fan_in)
        self.weight = nn.Parameter(w)           # <-- now optimisable
        self.bias   = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        w = self.weight * self.mask            # keep zeros zero
        return F.linear(x, w, self.bias)

# ────────── Learning rule ────────── 
    
@staticmethod
def nt_xent(z1, z2, T: float = 0.07) -> torch.Tensor:
    """NT-Xent loss (SimCLR)"""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # 2B × D
    sim = (z @ z.t()) / T                    # cosine-scaled similarities
    sim.fill_diagonal_(-9e15)                # mask self-similarities
    pos = torch.cat([torch.arange(B, 2 * B),
                        torch.arange(0, B)]).to(z.device)
    return -sim.log_softmax(dim=1)[torch.arange(2 * B), pos].mean()

# ────────── Visualization functions ────────── 

"""
Comprehensive evaluation and visualization tool for insect-vision neural networks.

Requirements:
- scikit-learn >= 0.20.0 (for trustworthiness metric)
- matplotlib
- opencv-python (cv2)
- scipy
"""
class ModelEvaluator:
    """A comprehensive tool for evaluating and visualizing insect-vision neural networks."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', output_dir: Optional[Path] = None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.output_dir = output_dir or Path('./apiaviz/evaluation_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        self.activations = {}
        self.hooks = []
        
    # ============== Hook Management ==============
    
    def register_hooks(self, layer_names: List[str]):
        """Register forward hooks for specified layers."""
        self.remove_hooks()
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name in layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                h = layer.register_forward_hook(get_activation(name))
                self.hooks.append(h)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    # ============== Core Visualization Methods ==============
    
    def create_heatmap_overlay(self, image: np.ndarray, heatmap: Union[torch.Tensor, np.ndarray], 
                             alpha: float = 0.7, colormap: str = 'bwr') -> np.ndarray:
        """Creates a simple heatmap overlay."""
        if isinstance(image, torch.Tensor):
             image = image.cpu().numpy()
        if len(image.shape) == 4: image = image[0]
        if image.shape[0] in [1, 2, 3]: image = np.transpose(image, (1, 2, 0))
        if image.max() > 1: image = image / 255.0
            
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().detach().cpu().numpy()
        
        if len(heatmap.shape) == 3: heatmap = heatmap.mean(axis=0)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (9, 9), 0)
        
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_smooth)[:, :, :3]
        
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 2:
            image_rgb = np.zeros((h, w, 3))
            image_rgb[:, :, 1] = image[:, :, 0]
            image_rgb[:, :, 2] = image[:, :, 1]
        else:
            image_rgb = image
        
        overlay = alpha * heatmap_colored + (1 - alpha) * image_rgb
        return np.clip(overlay, 0, 1)

    def plot_full_layer_analysis(self, input_tensor: torch.Tensor, input_image: np.ndarray, 
                                sample_idx: int = 0, label: Optional[float] = None, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Creates a comprehensive 3x3 grid showing activations from all key network layers."""
        self.model.eval()
        
        layers_to_hook = ['opsin', 'lamina', 'med_c', 'med_a', 'lobula']
        self.register_hooks(layers_to_hook)
        
        with torch.no_grad():
            kc_output_sparse = self.model(input_tensor)
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Full Layer-by-Layer Analysis (Sample Index: {sample_idx})', fontsize=20, fontweight='bold')
        
        display_img_hwc = input_image
        if len(display_img_hwc.shape) == 3 and display_img_hwc.shape[0] in [1, 2, 3]:
            display_img_hwc = np.transpose(display_img_hwc, (1, 2, 0))

        rgb_display = np.zeros((display_img_hwc.shape[0], display_img_hwc.shape[1], 3))
        if display_img_hwc.shape[2] == 2:
            rgb_display[:, :, 1] = display_img_hwc[:, :, 0]
            rgb_display[:, :, 2] = display_img_hwc[:, :, 1]
        else:
            rgb_display = display_img_hwc

        ax = axes[0, 0]
        ax.imshow(rgb_display)
        title = 'Input Image'
        if label is not None: title += f'\nLabel: {label:.2f}'
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        
        def plot_layer(ax, layer_name, title_text, cmap='bwr'):
            if layer_name in self.activations:
                activation = self.activations[layer_name][0]
                overlay = self.create_heatmap_overlay(rgb_display, activation, colormap=cmap)
                ax.imshow(overlay)
                ax.set_title(title_text, fontsize=16, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Layer Not Found', ha='center', va='center')
            ax.axis('off')

        plot_layer(axes[0, 1], 'opsin', 'Opsin Response')
        plot_layer(axes[0, 2], 'lamina', 'Lamina Response')
        plot_layer(axes[1, 0], 'med_c', 'Medulla Chromatic')
        plot_layer(axes[1, 1], 'med_a', 'Medulla Achromatic')
        plot_layer(axes[1, 2], 'lobula', 'Lobula Response', cmap='jet')
        
        ax = axes[2, 0]
        kc_output = kc_output_sparse[0].cpu().numpy()
        active_indices = np.where(kc_output > 0)[0]
        grid_size = int(np.ceil(np.sqrt(len(kc_output))))
        kc_grid = np.zeros((grid_size, grid_size))
        kc_grid.flat[:len(kc_output)] = kc_output
        ax.imshow(kc_grid, cmap='bwr', interpolation='nearest')
        ax.set_title(f'Kenyon Cells\n({len(active_indices)}/{len(kc_output)} active)', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        if save_path:
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.remove_hooks()
        return fig

    # ============== Quantitative Methods (UPDATED Cosine Similarity) ==============
    
    def evaluate_representations(self, features: np.ndarray, labels: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        results = {}
        X_tr, X_te, y_tr, y_te = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_tr, y_tr)
        results['knn_accuracy'] = knn.score(X_te, y_te) * 100
        
        probe = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
        probe.fit(X_tr, y_tr)
        results['linear_probe_accuracy'] = probe.score(X_te, y_te) * 100
        
        results['silhouette_score'] = silhouette_score(features, labels)
        
        km = KMeans(n_clusters=len(np.unique(labels)), n_init='auto', random_state=42).fit(features)
        results['adjusted_rand_index'] = adjusted_rand_score(labels, km.labels_)
        return results

    def plot_class_similarity_matrix(self, features: np.ndarray, labels: np.ndarray, 
                                     class_names: Optional[List[str]] = None, 
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        ### NEW AND IMPROVED ###
        Calculates and plots the cosine similarity between the MEAN feature vectors of each class.
        """
        unique_labels = np.unique(labels)
        
        # Calculate the mean feature vector for each class
        mean_features = np.array([features[labels == l].mean(axis=0) for l in unique_labels])
        
        # Compute cosine similarity on the mean feature vectors
        cos_sim = cosine_similarity(mean_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_sim, cmap='viridis', vmin=0, vmax=1)
        
        # Use provided class names or default to label numbers
        if class_names is None:
            class_names = [f"Class {l}" for l in unique_labels]
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        
        # Annotate heatmap with similarity values
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text_color = "w" if cos_sim[i, j] < 0.5 else "k"
                ax.text(j, i, f"{cos_sim[i, j]:.2f}",
                       ha="center", va="center", color=text_color, fontsize=12)
        
        plt.colorbar(im, label='Cosine Similarity')
        ax.set_title('Mean Feature Cosine Similarity per Class', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            if not Path(save_path).is_absolute(): save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, perplexity: int = 30, save_path: Optional[str] = None) -> plt.Figure:
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
        tsne_features = tsne.fit_transform(features)
        trust = trustworthiness(features, tsne_features, n_neighbors=10)
        
        n_classes = len(np.unique(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for i in range(n_classes):
            mask = labels == i
            ax.scatter(tsne_features[mask, 0], tsne_features[mask, 1], c=[colors[i]], label=f'Class {i}', alpha=0.7, s=50)
        
        ax.set_title(f't-SNE Visualization (Trustworthiness: {trust:.3f})', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            if not Path(save_path).is_absolute(): save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    # ============== Full Pipeline ==============

    def denormalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return (img_tensor.clone() * 0.5 + 0.5).clamp(0, 1)

    def visualize_batch(self, dataloader, n_samples: int = 4, use_random_sampling: bool = True):
        """Processes and visualizes samples using either random or sequential sampling."""
        print(f"\n{'='*50}")
        sampling_mode = "Random" if use_random_sampling else "Sequential"
        print(f"Generating visualizations for {n_samples} samples (Mode: {sampling_mode})")
        print(f"{'='*50}\n")

        dataset = dataloader.dataset
        num_total_samples = len(dataset)
        if n_samples > num_total_samples:
            n_samples = num_total_samples

        if use_random_sampling:
            indices_to_process = np.random.choice(num_total_samples, n_samples, replace=False)
        else:
            indices_to_process = np.arange(n_samples)

        for i, sample_idx in enumerate(indices_to_process):
            print(f"Processing sample {i + 1}/{n_samples} (Dataset index: {sample_idx})...")
            
            img_tensor_single, label = dataset[sample_idx]
            img_tensor_batch = img_tensor_single.unsqueeze(0).to(self.device)
            img_numpy_denorm = self.denormalize_image(img_tensor_single).cpu().numpy()
            if isinstance(label, torch.Tensor): label = label.item()
            
            try:
                save_filename = f'sample_idx_{sample_idx}_analysis.png'
                fig = self.plot_full_layer_analysis(
                    img_tensor_batch, img_numpy_denorm, sample_idx=sample_idx, label=label,
                    save_path=save_filename
                )
                plt.close(fig)
                print(f"  ✓ Analysis plot saved to {self.output_dir / save_filename}")
            except Exception as e:
                print(f"  ✗ Error during analysis for sample index {sample_idx}: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\nVisualizations saved to: {self.output_dir}")

    def run_full_evaluation(self, dataloader, features: np.ndarray, labels: np.ndarray,
                          n_visualization_samples: int = 4, use_random_sampling: bool = True) -> Dict[str, any]:
        """Runs the complete evaluation pipeline."""
        print("\n" + "="*60)
        print("RUNNING FULL MODEL EVALUATION")
        print("="*60 + "\n")
        results = {}
        
        print("1. Quantitative Evaluation")
        print("-" * 30)
        eval_metrics = self.evaluate_representations(features, labels)
        results['metrics'] = eval_metrics
        for metric, value in eval_metrics.items():
            print(f"   {metric:<25}: {value:6.2f}" + ('%' if 'accuracy' in metric else ''))
        
        print("\n2. Generating Global Visualizations")
        print("-" * 30)
        
        # Get class names from the dataset if they exist, for prettier plots
        class_names = getattr(dataloader.dataset, 'class_names', None)
        
        # Use the new, more insightful class similarity plot
        fig_cos = self.plot_class_similarity_matrix(features, labels, class_names=class_names, save_path='class_cosine_similarity.png')
        plt.close(fig_cos)
        print("   ✓ Class similarity matrix")
        
        fig_tsne = self.plot_tsne(features, labels, save_path='tsne_visualization.png')
        plt.close(fig_tsne)
        print("   ✓ t-SNE visualization")
        
        print("\n3. Sample-wise Layer-by-Layer Analysis")
        print("-" * 30)
        self.visualize_batch(dataloader, n_samples=n_visualization_samples, use_random_sampling=use_random_sampling)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*60 + "\n")
        
        return results