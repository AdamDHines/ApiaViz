# Imports
import math, torch, cv2, random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.colors as colors
import matplotlib.colors as mcolors

from pathlib import Path
from matplotlib import cm
from typing import Optional
from sklearn.cluster import KMeans
from typing import Optional, List, Dict, Union
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, adjusted_rand_score

import warnings
warnings.filterwarnings('ignore')

# ────────── Augmentation helpers ──────────
class MaybeGray2Ch:                          # 50 % colour-drop
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, gb: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            g = gb.mean(0, keepdim=True)
            return torch.cat([g, g], dim=0)
        return gb
    
def absolute_threshold_sparsity(x, threshold=0.25):
    # Creates a mask where activation is above the threshold
    mask = (x > threshold).float()
    return x * mask

def generate_smooth_scan_path(num_steps: int, max_x: int, max_y: int, num_waypoints: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a 2D smooth scanning path within rectangular bounds (max_x, max_y).
    This version correctly handles non-square images.
    """
    # Generate random waypoints within the separate x and y boundaries
    waypoints_x = np.random.randint(0, max_x + 1, num_waypoints)
    waypoints_y = np.random.randint(0, max_y + 1, num_waypoints)
    
    # Create the time points for the waypoints and the full path
    control_points = np.linspace(0, num_steps - 1, num_waypoints)
    full_timeline = np.arange(num_steps)
    
    # Interpolate x and y paths independently
    path_x = np.interp(full_timeline, control_points, waypoints_x)
    path_y = np.interp(full_timeline, control_points, waypoints_y)

    return path_x.astype(np.int32), path_y.astype(np.int32)

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
class ModelEvaluator:
    """
    A comprehensive and refactored tool for evaluating and visualizing
    insect-vision neural networks.

    This class handles both Artificial Neural Networks (ANNs) and Spiking
    Neural Networks (SNNs). It also distinguishes between two data modes:
    - 'static': Processing single, static images or patches.
    - 'scanning': Processing a time-series of patches that scan across a larger image.
    """

    def __init__(self, model, logger, device, output_dir=None, snn_params=None, kc_decay_factor=0.9):
        """Initializes the evaluator."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.logger = logger

        self.output_dir = output_dir

        self.activations = {}
        self.hooks = []
        
        self.snn = snn_params is not None
        self.snn_params = snn_params or {}
        self.num_steps = self.snn_params.get('num_steps', 25)

        self.kc_decay_factor = kc_decay_factor

    def bernoulli_spikes(self, x: torch.Tensor, rate_scale: float = 1.5) -> torch.Tensor:
        """Converts a rate-coded tensor to Bernoulli spikes."""
        p = (x * rate_scale).clamp(0, 1)
        return (torch.rand_like(p) < p).float()

    def register_hooks(self, layer_names: List[str]):
        """
        Registers forward hooks to capture layer activations.
        - For SNNs, it sums spike activations over time.
        - For scanning ANNs, it captures the activation of each patch individually.
        - For static ANNs, it captures the single activation map.
        """
        self.remove_hooks()

        def get_activation(name: str):
            def hook(model, input, output):
                act = (output[0] if isinstance(output, tuple) else output).detach()
                
                if self.snn:
                    # Sum spike activations for SNNs
                    if name not in self.activations:
                        self.activations[name] = act.clone()
                    else:
                        self.activations[name] += act
                elif self.is_scanning:
                    # For ANN scanning, append each patch's activation to a list
                    if name not in self.activations:
                        self.activations[name] = []
                    self.activations[name].append(act)
                else:
                    # For static ANN, just store the single activation map
                    self.activations[name] = act
            return hook

        for name in layer_names:
            try:
                layer = getattr(self.model, name)
                h = layer.register_forward_hook(get_activation(name))
                self.hooks.append(h)
            except AttributeError:
                self.logger.info(f"Warning: Layer '{name}' not found in model. Skipping hook.")

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _convert_static_to_spiking_single(self, static_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to convert a single static image tensor into a spike train for SNNs."""
        if static_tensor.dim() == 3:
            static_tensor = static_tensor.unsqueeze(0)
        static_tensor = static_tensor.to(self.device)
        
        frames = [self.bernoulli_spikes(static_tensor) for _ in range(self.num_steps)]
        spiking_batch = torch.stack(frames, dim=1)

        return spiking_batch.permute(1, 0, 2, 3, 4)

    def _prepare_image_for_display(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Standardizes an image into a displayable RGB numpy array [0,1]."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4:
            image = image[0]
        if image.shape[0] in [1, 2, 3]:
            image = np.transpose(image, (1, 2, 0))
        if image.max() > 1.0:
            image = image / 255.0
        
        h, w = image.shape[:2]
        if len(image.shape) == 2 or image.shape[2] == 1:
            return np.stack([image.squeeze()] * 3, axis=-1)
        if image.shape[2] == 2:
            rgb_image = np.zeros((h, w, 3), dtype=np.float32)
            rgb_image[:, :, 1] = image[:, :, 0]
            rgb_image[:, :, 2] = image[:, :, 1]
            return rgb_image
            
        return image.astype(np.float32)

    def create_static_heatmap_overlay(self, image: np.ndarray, heatmap: torch.Tensor, alpha: float = 0.6, colormap: str = 'jet') -> np.ndarray:
        """Overlays a single heatmap on a background image, resizing to fit."""
        heatmap_np = heatmap.squeeze().detach().cpu().numpy()
        
        if len(heatmap_np.shape) == 3:
            heatmap_np = heatmap_np.max(axis=0)
            
        heatmap_norm = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)

        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        
        overlay = alpha * heatmap_colored + (1 - alpha) * image
        return np.clip(overlay, 0, 1)

    def create_scanning_heatmap_overlay(self, background_image: np.ndarray, patch_activations: List[torch.Tensor], scan_path, patch_size: int, colormap: str = 'jet') -> np.ndarray:
        """
        Generates an overlay by placing individual, transparent heatmaps for each patch
        activation along its scan path on the full background image.

        Only positive activations are shown. Zero or negative activations are transparent.
        """
        path_x, path_y = scan_path
        h, w, _ = background_image.shape
        
        # Create a transparent RGBA canvas to "paint" the heatmaps on
        heatmap_canvas = np.zeros((h, w, 4), dtype=np.float32)
        
        cmap = cm.get_cmap(colormap)

        for i, patch_act in enumerate(patch_activations):
            # Process the activation for this patch
            act = patch_act.squeeze().cpu().numpy()
            if len(act.shape) == 3:
                act = act.mean(axis=0)
            
            # --- Key change: Only plot positive activations ---
            act[act < 0] = 0
            
            # Skip if there are no positive activations in this patch
            max_val = act.max()
            if max_val <= 1e-8:
                continue
                
            # Normalize only the positive values to [0, 1]
            act_norm = act / max_val
            
            # Apply colormap to get RGBA values
            colored_heatmap = cmap(act_norm)
            
            # --- Key change: Make zero values transparent ---
            colored_heatmap[act_norm == 0, 3] = 0.0 # Set alpha channel to 0
            
            # Get patch coordinates
            x0, y0 = int(path_x[i]), int(path_y[i])
            y1, x1 = y0 + patch_size, x0 + patch_size
            
            # Ensure the patch fits within the canvas bounds
            if y1 > h or x1 > w: continue
                
            # "Paint" this patch's heatmap onto the canvas
            heatmap_canvas[y0:y1, x0:x1] = colored_heatmap

        # Blend the heatmap canvas over the background image
        background_rgba = np.concatenate([background_image, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
        
        # Alpha compositing formula: a*A + b*B*(1-a)
        overlay_alpha = heatmap_canvas[:, :, 3:]
        overlay_rgb = heatmap_canvas[:, :, :3]
        
        final_image = overlay_rgb + background_image[:,:,:3] * (1 - overlay_alpha)

        return np.clip(final_image, 0, 1)


    def plot_full_layer_analysis(self, input_tensor: torch.Tensor, display_image: np.ndarray,
                                 sample_idx: int = 0, label: Optional[float] = None,
                                 scan_path = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Creates a comprehensive grid showing activations from key network layers.
        This version uses a leaky integrator for scanning ANN Kenyon Cell outputs.
        """
        self.model.eval(); self.activations = {}

        if self.snn:
            layers_to_hook = ['retina_lif', 'lamina_lif', 'med_lif', 'lobula_lif', 'asot_lif', 'aiot_lif', 'lot_lif']
            model_input = input_tensor.unsqueeze(1).to(self.device) if self.is_scanning else self._convert_static_to_spiking_single(input_tensor)
        else:
            layers_to_hook = ['retina', 'lamina', 'lam_c', 'lam_a', 'med_leaky', 'lobula']
            model_input = input_tensor.to(self.device)

        self.register_hooks(layers_to_hook)
        
        # resize model input to 75x75 for everything
        model_input = F.interpolate(model_input.unsqueeze(0), size=(75, 75), mode='bilinear', align_corners=False).squeeze(0)

        # --- Model Forward Pass ---
        with torch.no_grad():
            if self.snn:
                output = self.model(model_input, num_steps=self.num_steps)
                kc_output_sparse = output.sum(dim=0) # Sum over time
            
            elif not self.is_scanning: # Static ANN
                kc_output_sparse = self.model(model_input.unsqueeze(0))
            
            else: # ANN Scanning with Leaky Integrator
                
                # Initialize the accumulator for the KC output
                accumulated_kc_output = None
                
                # Loop through each patch in the time-series
                for t in range(model_input.shape[0]):
                    # Get the output for the current patch
                    # This call also triggers the hooks to save intermediate layer activations
                    current_output = self.model(model_input[t].unsqueeze(0))
                    
                    if accumulated_kc_output is None:
                        # For the first patch, the accumulator is just the output
                        accumulated_kc_output = current_output
                    else:
                        # For subsequent patches, decay the old value and add the new one
                        accumulated_kc_output = (self.kc_decay_factor * accumulated_kc_output) + current_output
                
                kc_output_sparse = accumulated_kc_output
        
        # --- Plotting ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle(f'Full Layer Analysis (Sample: {sample_idx}, SNN: {self.snn}, Scanning: {self.is_scanning})', fontsize=20, fontweight='bold')
        
        display_img_rgb = self._prepare_image_for_display(display_image)
        
        # Plot Input Image
        ax = axes[0, 0]
        ax.imshow(display_img_rgb)
        title_text = 'Input Image' if not self.is_scanning else 'Full Scene'
        if label is not None: title_text += f'\nLabel: {label:.2f}'
        ax.set_title(title_text, fontsize=16, fontweight='bold')
        ax.axis('off')

        def plot_layer(ax, layer_name, title_text, cmap='jet'):
            ax.axis('off')
            if layer_name in self.activations:
                # --- CHOOSE VISUALIZATION STRATEGY ---
                if self.is_scanning and not self.snn:
                    # New path for ANN scanning: use the list of activations
                    patch_activations = self.activations[layer_name]
                    patch_size = model_input.shape[2] # Get H from (T, C, H, W)
                    overlay = self.create_scanning_heatmap_overlay(
                        display_img_rgb, patch_activations, scan_path, patch_size, colormap=cmap
                    )
                else:
                    # Old path for Static ANN and all SNNs (which use averaged maps)
                    activation = self.activations[layer_name]
                    if self.snn: # For SNN, activation is summed; avg for display
                        activation = activation / self.num_steps
                    overlay = self.create_static_heatmap_overlay(display_img_rgb, activation[0], colormap=cmap)

                ax.imshow(overlay)
                ax.set_title(title_text, fontsize=16, fontweight='bold')

                # Create a colorbar representing the colormap scale
                norm = mcolors.Normalize(vmin=0, vmax=1) # Normalized from 0 to 1
                mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(mappable, ax=ax, shrink=0.75, aspect=20, label="Normalized Activation")
            else:
                ax.text(0.5, 0.5, 'Layer Not Found', ha='center', va='center', color='red')

        # --- Map layers to plot locations ---
        if self.snn:
            plot_layer(axes[0, 1], 'retina_lif', 'Retina (Avg. Spikes)')
            plot_layer(axes[0, 2], 'lamina_lif', 'Lamina (Avg. Spikes)')
            plot_layer(axes[1, 0], 'med_lif', 'Medulla (Avg. Spikes)')
            plot_layer(axes[1, 1], 'lobula_lif', 'Lobula (Avg. Spikes)')
            plot_layer(axes[1, 2], 'asot_lif', 'ASOT (Avg. Spikes)')
            plot_layer(axes[2, 0], 'aiot_lif', 'AIOT (Avg. Spikes)')
            plot_layer(axes[2, 1], 'lot_lif', 'LOT (Avg. Spikes)')
        else: # ANN
            plot_layer(axes[0, 1], 'retina', 'Retina Response')
            plot_layer(axes[0, 2], 'lamina', 'Lamina Response')
            plot_layer(axes[1, 0], 'lam_c', 'Lamina Chromatic')
            plot_layer(axes[1, 1], 'lam_a', 'Lamina Achromatic')
            plot_layer(axes[1, 2], 'med_leaky', 'Medulla activation Response')
            plot_layer(axes[2, 0], 'lobula', 'Lobula Response')

        # Get the GridSpec from the existing axes
        gs = axes[0, 0].get_gridspec()

        # Remove the old axes to make space for the new one
        if axes[2, 1] in fig.axes:
            axes[2, 1].remove()
        if axes[2, 2] in fig.axes:
            axes[2, 2].remove()

        # Create a new subplot that spans the desired columns
        ax = fig.add_subplot(gs[2, 1:])

        # Prepare data for plotting
        kc_output = kc_output_sparse[0].cpu().numpy()
        num_kc_total = len(kc_output)
        all_kc_indices = np.arange(num_kc_total)

        # **FILTER STEP**: Select only the cells with activity > 0
        active_mask = kc_output > 0
        kc_indices_active = all_kc_indices[active_mask]
        kc_output_active = kc_output[active_mask]
        num_active = len(kc_indices_active)

        # Set title - this will display even if there is no activity
        ax.set_title(f'Kenyon Cell Activity\n({num_active}/{num_kc_total} active)', fontsize=16, fontweight='bold')

        # Only proceed with plotting if there are active cells
        if num_active > 0:
            # Normalize colors based on the range of *active* cells
            norm = colors.Normalize(vmin=kc_output_active.min(), vmax=kc_output_active.max())
            cmap = cm.jet

            # Create the stem plot using only the active data
            # Hide default markers ('') as we will draw our own with scatter
            markerline, stemlines, baseline = ax.stem(
                kc_indices_active, kc_output_active, linefmt='-', markerfmt='', basefmt=' '
            )

            # Apply the colormap to the entire collection of stems
            stem_colors = cmap(norm(kc_output_active))
            stemlines.set_color(stem_colors)

            # Use scatter to plot the markers, which handles individual colors correctly
            ax.scatter(kc_indices_active, kc_output_active, c=kc_output_active, cmap=cmap, s=25, zorder=3)
            ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

        # Set labels and limits for the plot
        ax.set_xlabel('Kenyon Cell Index', fontsize=12)
        ax.set_ylabel('Activity Level', fontsize=12)
        # Set x-axis to the full range to correctly show the sparse locations
        ax.set_xlim([0, num_kc_total])
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        if save_path:
            full_save_path = Path(save_path)
            if not full_save_path.is_absolute():
                full_save_path = self.output_dir / save_path
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        
        self.remove_hooks()
        return fig

    def evaluate_representations(self, features: np.ndarray, labels: np.ndarray, groups, test_size: float = 0.2) -> Dict[str, float]:
        """Performs a quantitative evaluation of feature representations."""
        results = {}
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X=np.zeros(len(labels)), y=labels, groups=groups))

        X_tr, X_te = features[train_idx], features[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]

        # quick leakage check:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "LEAKAGE: groups overlap!"
        
        # 1. K-Nearest Neighbors Classifier
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_tr, y_tr)
        results['knn_accuracy'] = knn.score(X_te, y_te) * 100

        # 2. Linear Probe (Logistic Regression)
        probe = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, random_state=42)
        probe.fit(X_tr, y_tr)
        results['linear_probe_accuracy'] = probe.score(X_te, y_te) * 100

        # 3. Silhouette Score for clustering quality
        # Requires more than 1 label and less than n_samples-1 labels
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(features):
            results['silhouette_score'] = silhouette_score(features, labels, metric='cosine')
        else:
            results['silhouette_score'] = 0.0

        # 4. K-Means Clustering + Adjusted Rand Index
        if len(np.unique(labels)) > 1:
            km = KMeans(n_clusters=len(np.unique(labels)), n_init='auto', random_state=42).fit(features)
            results['adjusted_rand_index'] = adjusted_rand_score(labels, km.labels_)
        else:
            results['adjusted_rand_index'] = 0.0

        return results

    def plot_class_similarity_matrix(self, features: np.ndarray, labels: np.ndarray, class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> plt.Figure:
        """Plots the cosine similarity between mean feature vectors of each class."""
        unique_labels = sorted(np.unique(labels))
        
        # Calculate mean feature vector for each class
        mean_features = np.array([features[labels == l].mean(axis=0) for l in unique_labels])
        
        # Compute cosine similarity between these mean vectors
        cos_sim = cosine_similarity(mean_features)

        # # Run an additional cosine similarity within each class to get intra-class similarity
        # class_sims = []
        # for l in unique_labels:
        #     class_feats = features[labels == l]
        #     if len(class_feats) > 1:
        #         class_sims.append(cosine_similarity(class_feats))

        # # Plot each class sim
        # for i, sim in enumerate(class_sims):
        #     fig, ax = plt.subplots(figsize=(6, 5))
        #     im = ax.imshow(sim, cmap='viridis', vmin=0, vmax=1)
        #     ax.set_title(f'Intra-class Cosine Similarity (Class {unique_labels[i]})', fontsize=14, fontweight='bold')
        #     plt.colorbar(im, label='Cosine Similarity')
        #     plt.show()
        #     plt.close(fig)

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_sim, cmap='viridis', vmin=0, vmax=1)
        
        # Configure ticks and labels
        if class_names is None:
            class_names = [f"Class {l}" for l in unique_labels]
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text_color = "w" if cos_sim[i, j] < 0.5 else "k"
                ax.text(j, i, f"{cos_sim[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=12)
        
        # Final touches
        plt.colorbar(im, label='Cosine Similarity')
        ax.set_title('Mean Feature Cosine Similarity per Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            full_save_path = Path(save_path)
            if not full_save_path.is_absolute():
                 full_save_path = self.output_dir / save_path
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, perplexity: int = 30, save_path: Optional[str] = None) -> plt.Figure:
        """Generates and plots a t-SNE visualization of the features."""
        n_classes = len(np.unique(labels))
        if len(features) <= perplexity:
            self.logger.info(f"Warning: Perplexity ({perplexity}) is too high for the number of samples ({len(features)}). Skipping t-SNE plot.")
            return plt.figure() # Return an empty figure

        # 1. Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42, metric='cosine')
        tsne_features = tsne.fit_transform(features)
        
        # 2. Calculate Trustworthiness
        trust = trustworthiness(features, tsne_features, n_neighbors=10, metric='cosine')
        
        # 3. Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        
        # 4. Scatter plot for each class
        for i in range(n_classes):
            class_mask = (labels == i)
            ax.scatter(tsne_features[class_mask, 0], tsne_features[class_mask, 1],
                       c=[colors[i]], label=f'Class {i}', alpha=0.8, s=50)
                       
        # 5. Final touches
        ax.set_title(f't-SNE Visualization (Trustworthiness: {trust:.3f})', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.axis('off')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        if save_path:
            full_save_path = Path(save_path)
            if not full_save_path.is_absolute():
                 full_save_path = self.output_dir / save_path
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def visualize_batch(self, dataloader, indices_to_process: List[int]):
        """
        Generates and saves layer analysis plots for a specific list of sample indices.
        """
        n_samples = len(indices_to_process)
        self.logger.info(f"\n{'='*50}\nGenerating visualizations for {n_samples} specific samples (one per class)\n{'='*50}\n")
        
        dataset = dataloader.dataset

        for i, sample_idx in enumerate(indices_to_process):
            self.logger.info(f"Processing sample {i + 1}/{n_samples} (Dataset index: {sample_idx})...")
            
            # --- Unpack data from dataset ---
            # This handles both (patch, label, _, _) and (timeseries, label, full_img, scan_path)
            item = dataset[sample_idx]
            input_tensor, label, full_image, scan_path = item[0], item[1], item[2], item[3]
            
            # Use the patch itself as the display image if no full image is available
            display_image = full_image if (isinstance(full_image, torch.Tensor) and full_image.numel() > 0) else input_tensor

            if isinstance(label, torch.Tensor):
                label = label.item()

            try:
                save_filename = f'sample_idx_{sample_idx}_class_{label}_analysis.pdf'
                fig = self.plot_full_layer_analysis(
                    input_tensor=input_tensor,
                    display_image=display_image,
                    sample_idx=sample_idx,
                    label=label,
                    scan_path=scan_path if self.is_scanning else None,
                    save_path=save_filename
                )
                plt.close(fig) # Prevent plots from displaying in notebooks
                self.logger.info(f"  ✓ Analysis plot saved to {self.output_dir / save_filename}")
            except Exception as e:
                import traceback
                self.logger.info(f"  ✗ Error during analysis for sample index {sample_idx}: {e}")
                traceback.print_exc()

        self.logger.info(f"\nVisualizations saved to: {self.output_dir}")

    def run_full_evaluation(self, dataloader, features: np.ndarray, labels: np.ndarray, groups, is_scanning: bool = False):
        """
        Runs the complete evaluation pipeline. This version safely handles cases where
        classes have only one sample by skipping the quantitative evaluation.

        Args:
            dataloader: The DataLoader for fetching samples for visualization.
            features: Pre-extracted features from the model for the entire dataset.
            labels: Corresponding labels for the features.
            is_scanning: Boolean flag to indicate if data is from a scanning process.
        """
        self.logger.info("\n" + "="*60 + "\nRUNNING FULL MODEL EVALUATION\n" + "="*60 + "\n")
        
        # Set the data mode for this evaluation run
        self.is_scanning = is_scanning
        results = {}

        # --- Pre-computation Check for Quantitative Evaluation ---
        # Count samples per class to see if splitting is possible.
        unique_labels, counts = np.unique(labels, return_counts=True)
        can_run_quantitative_eval = all(counts >= 2)

        # 1. Quantitative Evaluation
        self.logger.info("1. Quantitative Evaluation\n" + "-" * 30)
        
        if can_run_quantitative_eval:
            self.logger.info("   All classes have 2 or more samples. Running quantitative metrics...")
            eval_metrics = self.evaluate_representations(features, labels, groups)
            results['metrics'] = eval_metrics
            for metric, value in eval_metrics.items():
                unit = '%' if 'accuracy' in metric else ''
                self.logger.info(f"   {metric:<25}: {value:6.2f}{unit}")
        else:
            self.logger.info("   SKIPPING quantitative metrics (KNN, Linear Probe, etc.).")
            self.logger.info("   Reason: At least one class has fewer than 2 samples, which is required for data splitting.")

        # 2. Representation-level Visualizations (These run regardless)
        self.logger.info("\n2. Representation Space Analysis\n" + "-" * 30)
        class_names = getattr(dataloader.dataset, 'class_names', None)
        
        fig_cos = self.plot_class_similarity_matrix(
            features, labels, class_names=class_names, save_path='class_cosine_similarity.png'
        )
        plt.close(fig_cos)
        self.logger.info("   ✓ Class similarity matrix saved.")
        
        fig_tsne = self.plot_tsne(features, labels, save_path='tsne_visualization.png')
        plt.close(fig_tsne)
        self.logger.info("   ✓ t-SNE visualization saved.")
        
        # 3. Sample-wise Qualitative Analysis (One sample per class)
        self.logger.info("\n3. Sample-wise Layer-by-Layer Analysis\n" + "-" * 30)
        
        # This logic for selecting one sample per class remains robust.
        indices_to_visualize = []
        for label_val in unique_labels:
            try:
                first_occurrence_idx = np.where(labels == label_val)[0][0]
                indices_to_visualize.append(int(first_occurrence_idx))
            except IndexError:
                self.logger.info(f"Warning: Could not find any samples for label {label_val} to visualize.")
        
        if indices_to_visualize:
            self.logger.info(f"   Selected one sample from each of the {len(unique_labels)} classes for visualization.")
            self.logger.info(f"   Dataset indices to be plotted: {indices_to_visualize}")
            self.visualize_batch(dataloader, indices_to_process=indices_to_visualize)
        
        self.logger.info("\n" + "="*60 + "\nEVALUATION COMPLETE\n" + f"Results saved to: {self.output_dir}\n" + "="*60 + "\n")
        return results