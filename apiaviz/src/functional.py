# Imports
import math, torch, cv2, random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path
from typing import Optional
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, List, Union
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
    """Comprehensive evaluation and visualization tool for insect-vision neural networks."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda', output_dir: Optional[Path] = None):
        """
        Initialize the evaluator with a model.
        
        Args:
            model: The neural network model to evaluate
            device: Device to run computations on ('cuda' or 'cpu')
            output_dir: Directory to save visualizations (creates 'evaluation_outputs' if None)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Setup output directory
        self.output_dir = output_dir or Path('evaluation_outputs')
        self.output_dir.mkdir(exist_ok=True)
        
        # Storage for hooks and activations
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
    # ============== Hook Management ==============
    
    def register_hooks(self):
        """Register forward and backward hooks for all major layers."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, input, output):
                self.gradients[name] = output[0].detach()
            return hook
        
        # Register hooks for each layer - check if they exist in the model
        layer_mapping = {
            'opsin': 'opsin',
            'lamina': 'lamina', 
            'med_c': 'med_c',
            'med_a': 'med_a',
            'lobula': 'lobula',
            'asot': 'asot',
            'aiot': 'aiot',
            'lot': 'lot',
        }
        
        for name, attr_name in layer_mapping.items():
            if hasattr(self.model, attr_name):
                layer = getattr(self.model, attr_name)
                # Handle Sequential layers
                if isinstance(layer, nn.Sequential):
                    # Register hook on the whole sequential
                    h = layer.register_forward_hook(get_activation(name))
                    self.hooks.append(h)
                else:
                    h = layer.register_forward_hook(get_activation(name))
                    self.hooks.append(h)
                    h = layer.register_backward_hook(get_gradient(name))
                    self.hooks.append(h)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    # ============== Attention/Gradient Methods ==============
    
    def get_saliency_map(self, input_tensor: torch.Tensor, target_neuron: Optional[int] = None) -> torch.Tensor:
        """Compute gradient-based saliency map."""
        input_tensor.requires_grad_()

        # Forward pass
        output = self.model(input_tensor)
        
        # Select target for backprop
        if target_neuron is None:
            target = output.max()
        else:
            target = output[0, target_neuron]
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

        saliency = input_tensor.grad.data.abs()
        return saliency
    
    def guided_backpropagation(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """Guided backpropagation for cleaner gradients."""
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        
        # Store original ReLU
        def relu_hook_function(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)
        
        # Register hooks on all ReLU activations
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_backward_hook(relu_hook_function))
        
        # Forward and backward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target = output.max()
        else:
            target = output[0, target_class]
        
        self.model.zero_grad()
        target.backward()
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        return input_tensor.grad.data
    
    def smooth_grad(self, input_tensor: torch.Tensor, n_samples: int = 50, noise_level: float = 0.15) -> torch.Tensor:
        """SmoothGrad: average gradients with noise for smoother visualization."""
        input_tensor = input_tensor.to(self.device)
        smooth_grad = torch.zeros_like(input_tensor)
        
        for _ in range(n_samples):
            noise = torch.randn_like(input_tensor) * noise_level
            noisy_input = input_tensor + noise
            noisy_input.requires_grad_()
            
            output = self.model(noisy_input)
            target = output.max()
            
            self.model.zero_grad()
            target.backward()
            
            smooth_grad += noisy_input.grad.data
        
        return smooth_grad / n_samples
    
    def integrated_gradients(self, input_tensor: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """Compute integrated gradients for better attribution."""
        input_tensor = input_tensor.to(self.device)
        baseline = torch.zeros_like(input_tensor)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        integrated_grads = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_()
            
            output = self.model(interpolated)
            target = output.max()
            
            self.model.zero_grad()
            target.backward()
            
            integrated_grads += interpolated.grad.data / steps
        
        return integrated_grads.abs()
    
    def multi_scale_attention(self, input_tensor: torch.Tensor, scales: List[float] = [0.5, 1.0, 1.5]) -> torch.Tensor:
        """Compute attention at multiple scales for robustness."""
        input_tensor = input_tensor.to(self.device)
        original_size = input_tensor.shape[-2:]
        multi_scale_grad = torch.zeros_like(input_tensor)
        
        for scale in scales:
            # Resize input
            scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_input = F.interpolate(input_tensor, size=scaled_size, mode='bilinear', align_corners=False)
            scaled_input.requires_grad_()
            
            # Forward and backward
            output = self.model(scaled_input)
            target = output.max()
            
            self.model.zero_grad()
            target.backward()
            
            # Resize gradient back
            grad_resized = F.interpolate(scaled_input.grad.data, size=original_size, 
                                        mode='bilinear', align_corners=False)
            multi_scale_grad += grad_resized
        
        return multi_scale_grad / len(scales)
    
    # ============== Visualization Methods ==============
    
    def create_heatmap_overlay(self, image: np.ndarray, heatmap: Union[torch.Tensor, np.ndarray], 
                             alpha: float = 0.6, colormap: str = 'bwr', 
                             blur_sigma: float = 1.5, threshold: float = 0.3) -> np.ndarray:
        """Create a beautiful heatmap overlay with advanced blending."""
        # Handle different input image formats
        if len(image.shape) == 4:  # Batch dimension (B, C, H, W)
            image = image[0]  # Take first image
        
        if len(image.shape) == 3 and image.shape[0] in [1, 2, 3, 4]:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
        
        # Process heatmap
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().detach().cpu().numpy()
        
        if len(heatmap.shape) == 3:
            heatmap = heatmap.mean(axis=0)
        
        # Normalize heatmap
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_norm[heatmap_norm < threshold] *= 0.5  # Dim unimportant areas
        
        # Smooth the heatmap
        heatmap_smooth = gaussian_filter(heatmap_norm, sigma=blur_sigma)
        
        # Resize to match image
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap_smooth, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)
        
        # Prepare image (handle green-blue format)
        if len(image.shape) == 2:
            image_rgb = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 2:  # Green-blue
            image_rgb = np.zeros((h, w, 3))
            image_rgb[:, :, 1] = image[:, :, 0]  # Green
            image_rgb[:, :, 2] = image[:, :, 1]  # Blue
        else:
            image_rgb = image.copy()
        
        # Advanced blending
        alpha_channel = heatmap_resized ** 0.8  # Power for non-linear blending
        alpha_expanded = np.expand_dims(alpha_channel, axis=-1)
        
        overlay = image_rgb * (1 - alpha * alpha_expanded) + \
                  heatmap_colored[:, :, :3] * alpha * alpha_expanded
        
        return np.clip(overlay, 0, 1)
    
    def plot_attention_summary(self, input_tensor: torch.Tensor, input_image: np.ndarray, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive attention summary with multiple techniques.
        
        Args:
            input_tensor: Input tensor for gradient computation (B, C, H, W) or (C, H, W)
            input_image: Numpy array for display. Can be (C, H, W) or (H, W, C) format
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Handle different input image formats
        if len(input_image.shape) == 4:  # Batch dimension (B, C, H, W)
            input_image = input_image[0]  # Take first image
        
        if len(input_image.shape) == 3 and input_image.shape[0] in [1, 2, 3, 4]:  # (C, H, W)
            input_image = np.transpose(input_image, (1, 2, 0))  # Convert to (H, W, C)
        
        # Prepare display image
        if input_image.shape[2] == 2:
            display_img = np.zeros((input_image.shape[0], input_image.shape[1], 3))
            display_img[:, :, 1] = input_image[:, :, 0]
            display_img[:, :, 2] = input_image[:, :, 1]
        else:
            display_img = input_image
        
        # 1. Original image
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(display_img)
        ax1.set_title('Original Input', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Standard gradient
        standard_grad = self.get_saliency_map(input_tensor.clone())
        ax2 = plt.subplot(2, 4, 2)
        overlay = self.create_heatmap_overlay(input_image, standard_grad[0], colormap='bwr')
        ax2.imshow(overlay)
        ax2.set_title('Standard Gradient', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 3. Guided backprop
        guided_grad = self.guided_backpropagation(input_tensor.clone())
        ax3 = plt.subplot(2, 4, 3)
        overlay = self.create_heatmap_overlay(input_image, guided_grad[0].abs(), colormap='bwr')
        ax3.imshow(overlay)
        ax3.set_title('Guided Backpropagation', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 4. SmoothGrad
        smooth_grad = self.smooth_grad(input_tensor.clone())
        ax4 = plt.subplot(2, 4, 4)
        overlay = self.create_heatmap_overlay(input_image, smooth_grad[0].abs(), colormap='bwr')
        ax4.imshow(overlay)
        ax4.set_title('SmoothGrad', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # 5. Multi-scale attention
        multi_scale = self.multi_scale_attention(input_tensor.clone())
        ax5 = plt.subplot(2, 4, 5)
        overlay = self.create_heatmap_overlay(input_image, multi_scale[0].abs(), colormap='bwr')
        ax5.imshow(overlay)
        ax5.set_title('Multi-Scale Attention', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # 6. Integrated gradients
        int_grads = self.integrated_gradients(input_tensor.clone())
        ax6 = plt.subplot(2, 4, 6)
        overlay = self.create_heatmap_overlay(input_image, int_grads[0].abs(), colormap='bwr')
        ax6.imshow(overlay)
        ax6.set_title('Integrated Gradients', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # 7. Combined attention (weighted average)
        combined = 0.3 * standard_grad[0].abs() + \
                  0.3 * guided_grad[0].abs() + \
                  0.2 * smooth_grad[0].abs() + \
                  0.2 * multi_scale[0].abs()
        
        ax7 = plt.subplot(2, 4, 7)
        overlay = self.create_heatmap_overlay(input_image, combined, alpha=0.7, 
                                            colormap='bwr', blur_sigma=2.0)
        ax7.imshow(overlay)
        ax7.set_title('Combined Attention', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        # 8. 3D surface plot
        ax8 = plt.subplot(2, 4, 8, projection='3d')
        attention_2d = combined.mean(0).detach().cpu().numpy()
        X, Y = np.meshgrid(np.arange(attention_2d.shape[1]), np.arange(attention_2d.shape[0]))
        
        # Downsample for cleaner visualization
        step = 4
        X_down = X[::step, ::step]
        Y_down = Y[::step, ::step]
        Z_down = attention_2d[::step, ::step]
        
        surf = ax8.plot_surface(X_down, Y_down, Z_down, cmap='bwr', 
                               linewidth=0, antialiased=True, alpha=0.8)
        ax8.set_title('3D Attention Surface', fontsize=14, fontweight='bold')
        ax8.set_xlabel('X')
        ax8.set_ylabel('Y')
        ax8.set_zlabel('Attention')
        
        plt.suptitle('Advanced Attention Visualization Techniques', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            # Handle relative paths by prepending output_dir
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_layer_analysis(self, input_tensor: torch.Tensor, input_image: np.ndarray, 
                           save_path: Optional[str] = None) -> plt.Figure:
        """Analyze and visualize layer-wise activations and attention."""
        self.register_hooks()
        
        # Forward pass to collect activations
        output = self.model(input_tensor.to(self.device))
        output = F.normalize(output, dim=1)
        
        # Extract KC output early for use in multiple plots
        kc_output = output[0].detach().cpu().numpy()
        active_kc = np.where(kc_output > 0)[0]
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        
        # Handle different input image formats
        if len(input_image.shape) == 4:  # Batch dimension (B, C, H, W)
            input_image = input_image[0]  # Take first image
        
        if len(input_image.shape) == 3 and input_image.shape[0] in [1, 2, 3, 4]:  # (C, H, W)
            input_image = np.transpose(input_image, (1, 2, 0))  # Convert to (H, W, C)
        
        # Prepare display image
        if input_image.shape[2] == 2:
            display_img = np.zeros((input_image.shape[0], input_image.shape[1], 3))
            display_img[:, :, 1] = input_image[:, :, 0]
            display_img[:, :, 2] = input_image[:, :, 1]
        else:
            display_img = input_image
        
        # 1. Original image
        ax1 = plt.subplot(4, 5, 1)
        ax1.imshow(display_img)
        ax1.set_title('Input Image\n(Green & Blue channels)', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Plot layer activations
        layer_configs = [
            ('opsin', 4, 'Opsin Response\n(Photoreceptor)', 'bwr'),
            ('lamina', 5, 'Lamina\n(Local Motion/Contrast)', 'bwr'),
            ('med_c', 9, 'Medulla Chromatic\n(Color Processing)', 'bwr'),
            ('med_a', 10, 'Medulla Achromatic\n(Luminance Edges)', 'bwr'),
            ('lobula', 14, 'Lobula\n(Feature Integration)', 'bwr'),
        ]
        
        for layer_name, position, title, cmap in layer_configs:
            if layer_name in self.activations:
                ax = plt.subplot(4, 5, position)
                act = self.activations[layer_name]
                
                # Handle different activation shapes
                if len(act.shape) == 4:  # Conv layer
                    attention = act[0].mean(dim=0, keepdim=True)
                else:
                    attention = act[0].unsqueeze(0)
                
                overlay = self.create_heatmap_overlay(input_image, attention, colormap=cmap)
                ax.imshow(overlay)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.axis('off')
        
        # 2. Gradient saliency  
        ax_grad = plt.subplot(4, 5, 2)
        with torch.enable_grad():
            saliency = self.get_saliency_map(input_tensor.clone())
        overlay = self.create_heatmap_overlay(input_image, saliency[0], colormap='bwr')
        ax_grad.imshow(overlay)
        ax_grad.set_title('Gradient Saliency', fontsize=12, fontweight='bold')
        ax_grad.axis('off')
        
        # 3. Combined attention visualization
        if 'lobula' in self.activations:
            ax_combined = plt.subplot(4, 5, 3)
            lobula_act = self.activations['lobula'][0].mean(dim=0).cpu().numpy()
            saliency_np = saliency[0].mean(dim=0).cpu().numpy()
            
            # Normalize both
            lobula_norm = (lobula_act - lobula_act.min()) / (lobula_act.max() - lobula_act.min() + 1e-8)
            saliency_norm = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
            
            # Combined attention
            combined_attention = 0.4 * saliency_norm + 0.6 * lobula_norm
            overlay = self.create_heatmap_overlay(input_image, combined_attention, 
                                                colormap='bwr', alpha=0.7, blur_sigma=2.0)
            ax_combined.imshow(overlay)
            ax_combined.set_title('Combined Attention', fontsize=12, fontweight='bold')
            ax_combined.axis('off')
        
        for layer_name, position, title, cmap in layer_configs:
            if layer_name in self.activations:
                ax = plt.subplot(4, 5, position)
                act = self.activations[layer_name]
                
                # Handle different activation shapes
                if len(act.shape) == 4:  # Conv layer
                    attention = act[0].mean(dim=0, keepdim=True)
                else:
                    attention = act[0].unsqueeze(0)
                
                overlay = self.create_heatmap_overlay(input_image, attention, colormap=cmap)
                ax.imshow(overlay)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.axis('off')
        
        # VPN pathway responses (if present)
        vpn_layers = ['asot', 'aiot', 'lot']
        vpn_titles = ['ASOT Pathway\n(Anterior Superior)', 
                      'AIOT Pathway\n(Anterior Inferior)', 
                      'LOT Pathway\n(Lateral)']
        
        for i, (layer, title) in enumerate(zip(vpn_layers, vpn_titles)):
            if layer in self.activations:
                ax = plt.subplot(4, 5, 6 + i)
                act = self.activations[layer][0]
                # These are after global pooling, so visualize as heatmap
                response = act.mean(dim=0, keepdim=True)
                overlay = self.create_heatmap_overlay(input_image, response, 
                                                    colormap='bwr', alpha=0.8)
                ax.imshow(overlay)
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.axis('off')
        
        # Feature importance across layers
        ax_feat = plt.subplot(4, 5, 13)
        layer_names = ['Opsin', 'Lamina', 'Med-C', 'Med-A', 'Lobula']
        importances = []
        
        for name in ['opsin', 'lamina', 'med_c', 'med_a', 'lobula']:
            if name in self.activations:
                act = self.activations[name][0]
                importance = act.abs().mean().item()
                importances.append(importance)
        
        if importances:
            ax_feat.bar(layer_names[:len(importances)], importances, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax_feat.set_title('Layer-wise Feature Importance', fontsize=12, fontweight='bold')
            ax_feat.set_xlabel('Layer')
            ax_feat.set_ylabel('Mean Activation')
            ax_feat.grid(True, alpha=0.3)
        else:
            ax_feat.text(0.5, 0.5, 'No Layer Data', ha='center', va='center', transform=ax_feat.transAxes)
            ax_feat.axis('off')
        
        # Sparse KC histogram
        ax_hist = plt.subplot(4, 5, 15)
        if len(kc_output[kc_output > 0]) > 0:
            ax_hist.hist(kc_output[kc_output > 0], bins=30, color='darkred', alpha=0.7, edgecolor='black')
            ax_hist.set_title('Active Kenyon Cell Distribution', fontsize=12, fontweight='bold')
            ax_hist.set_xlabel('Activation Value')
            ax_hist.set_ylabel('Count')
            ax_hist.grid(True, alpha=0.3)
        else:
            ax_hist.text(0.5, 0.5, 'No Active KCs', ha='center', va='center', transform=ax_hist.transAxes)
            ax_hist.set_title('Active Kenyon Cell Distribution', fontsize=12, fontweight='bold')
            ax_hist.axis('off')
        
        # KC activation pattern
        ax_kc = plt.subplot(4, 5, 11)
        
        # Create 2D representation
        kc_dim = len(kc_output)
        grid_size = int(np.ceil(np.sqrt(kc_dim)))
        kc_grid = kc_output.reshape(-1, 1).repeat(10, axis=1).reshape(grid_size, -1)[:grid_size, :grid_size]
        
        im = ax_kc.imshow(kc_grid, cmap='bwr', interpolation='nearest')
        ax_kc.set_title(f'Kenyon Cells\n({len(active_kc)} active / {kc_dim} total)', 
                       fontsize=10, fontweight='bold')
        ax_kc.axis('off')
        
        # Add a 3D visualization of attention
        ax_3d = plt.subplot(4, 5, 17, projection='3d')
        if 'lobula' in self.activations:
            attention_2d = self.activations['lobula'][0].mean(dim=0).cpu().numpy()
            h, w = attention_2d.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Downsample for cleaner visualization
            step = max(1, min(h, w) // 20)
            X_down = X[::step, ::step]
            Y_down = Y[::step, ::step]
            Z_down = attention_2d[::step, ::step]
            
            # Normalize Z values
            Z_down = (Z_down - Z_down.min()) / (Z_down.max() - Z_down.min() + 1e-8)
            
            surf = ax_3d.plot_surface(X_down, Y_down, Z_down, cmap='bwr', 
                                    linewidth=0, antialiased=True, alpha=0.8)
            ax_3d.set_title('3D Attention Surface', fontsize=10, fontweight='bold')
            ax_3d.set_xlabel('X', fontsize=8)
            ax_3d.set_ylabel('Y', fontsize=8)
            ax_3d.set_zlabel('Activation', fontsize=8)
            ax_3d.view_init(elev=30, azim=45)
        
        plt.suptitle('Neural Network Layer Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            # Handle relative paths by prepending output_dir
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.remove_hooks()
        return fig
    
    # ============== Quantitative Evaluation Methods ==============
    
    def evaluate_representations(self, features: np.ndarray, labels: np.ndarray, 
                                test_size: float = 0.2) -> Dict[str, float]:
        """Comprehensive evaluation of learned representations."""
        results = {}
        
        # Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42)
        
        # k-NN accuracy
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(X_tr, y_tr)
        results['knn_accuracy'] = knn.score(X_te, y_te) * 100
        
        # Linear probe accuracy
        probe = LogisticRegression(max_iter=10_000, solver="saga", n_jobs=-1)
        probe.fit(X_tr, y_tr)
        results['linear_probe_accuracy'] = probe.score(X_te, y_te) * 100
        
        # Unsupervised metrics
        results['silhouette_score'] = silhouette_score(features, labels)
        
        km = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=42).fit(features)
        results['adjusted_rand_index'] = adjusted_rand_score(labels, km.labels_)
        
        # Cosine similarity statistics
        cos_sim = cosine_similarity(features)
        results['mean_cosine_similarity'] = cos_sim.mean()
        results['std_cosine_similarity'] = cos_sim.std()
        
        return results
    
    def plot_cosine_similarity_matrix(self, features: np.ndarray, save_path: Optional[str] = None) -> plt.Figure:
        """Plot cosine similarity matrix."""
        cos_sim = cosine_similarity(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cos_sim, cmap='viridis', norm=Normalize(vmin=0, vmax=1))
        plt.colorbar(im, label='Cosine Similarity')
        ax.set_title('Feature Cosine Similarity Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        
        if save_path:
            # Handle relative paths by prepending output_dir
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_tsne(self, features: np.ndarray, labels: np.ndarray, 
                  perplexity: int = 30, save_path: Optional[str] = None) -> plt.Figure:
        """Generate t-SNE visualization."""
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
        tsne_features = tsne.fit_transform(features)
        
        # Calculate trustworthiness
        trust = trustworthiness(features, tsne_features, n_neighbors=10)
        
        # Create color palette
        n_classes = len(np.unique(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i in range(n_classes):
            mask = labels == i
            ax.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                      c=[colors[i]], label=f'Class {i}', alpha=0.7, s=50)
        
        ax.set_title(f't-SNE Visualization (Trustworthiness: {trust:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            # Handle relative paths by prepending output_dir
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_top_k_features(self, features: np.ndarray, labels: np.ndarray, 
                           top_k: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize top-k activated features per class."""
        # Compute mean features per class
        unique_labels = np.unique(labels)
        mean_features = np.array([features[labels == l].mean(axis=0) for l in unique_labels])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        x = np.arange(mean_features.shape[1])
        
        for i, label in enumerate(unique_labels):
            vec = mean_features[i]
            top_idx = np.argsort(vec)[-top_k:]
            
            # Plot stems
            for idx in top_idx:
                ax.vlines(x=idx, ymin=0, ymax=vec[idx],
                         color=colors[i], linewidth=3, alpha=0.8)
            
            # Plot markers
            ax.scatter(top_idx, vec[top_idx],
                      color=colors[i], edgecolor='black', s=80, 
                      label=f'Class {label}', zorder=3)
        
        ax.set_title(f'Top {top_k} Feature Activations per Class', fontsize=16, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=14)
        ax.set_ylabel('Mean Activation', fontsize=14)
        ax.legend(frameon=False, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Handle relative paths by prepending output_dir
            if not Path(save_path).is_absolute():
                save_path = self.output_dir / save_path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_sample(self, input_tensor: torch.Tensor, input_image: Optional[np.ndarray] = None,
                        sample_name: str = "sample") -> Dict[str, plt.Figure]:
        """
        Convenient method to visualize a single sample with all techniques.
        
        Args:
            input_tensor: Input tensor (B, C, H, W) or (C, H, W)
            input_image: Optional numpy array for display. If None, will use input_tensor
            sample_name: Name prefix for saving files
            
        Returns:
            Dictionary of figure objects
        """
        # Ensure batch dimension and gradient computation
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Clone tensor to avoid modifying original
        input_tensor = input_tensor.clone().to(self.device).requires_grad_(True)
        
        # Convert to numpy for display if not provided
        if input_image is None:
            input_image = input_tensor[0].cpu().numpy()
        
        # Ensure tensor is on device
        input_tensor = input_tensor.to(self.device)
        
        figures = {}
        
        # Attention summary
        fig1 = self.plot_attention_summary(
            input_tensor, 
            input_image,
            save_path=f'{sample_name}_attention_summary.png'
        )
        figures['attention_summary'] = fig1
        
        # Layer analysis
        fig2 = self.plot_layer_analysis(
            input_tensor,
            input_image,
            save_path=f'{sample_name}_layer_analysis.png'
        )
        figures['layer_analysis'] = fig2
        
        return figures
    
    # ============== Batch Processing Methods ==============
    
    def visualize_batch(self, dataloader, n_samples: int = 4, save_individual: bool = True):
        """Process and visualize a batch of samples."""
        print(f"\n{'='*50}")
        print(f"Generating visualizations for {n_samples} samples...")
        print(f"{'='*50}\n")
        
        # Get first batch
        imgs, labels = next(iter(dataloader))
        
        for idx in range(min(n_samples, imgs.shape[0])):
            print(f"Processing sample {idx + 1}/{n_samples}...")
            
            # Get single image with batch dimension
            img_tensor = imgs[idx:idx+1].to(self.device)
            
            # Get numpy array (C, H, W) format - the methods will handle conversion
            img_numpy = imgs[idx].cpu().numpy()
            
            # Generate visualizations
            try:
                # Use the convenient visualize_sample method
                figures = self.visualize_sample(
                    img_tensor,
                    img_numpy,
                    sample_name=f'sample_{idx+1}'
                )
                
                # Close figures to free memory
                for fig in figures.values():
                    plt.close(fig)
                
                print(f"  ✓ Sample {idx + 1} completed")
                
            except Exception as e:
                print(f"  ✗ Error with sample {idx + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\nVisualizations saved to: {self.output_dir}")
    
    def denormalize_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize image from [-1,1] to [0,1] for visualization."""
        img = img_tensor.clone()
        img = img * 0.5 + 0.5  # Reverse normalization
        return img.clamp(0, 1)
    
    # ============== Full Evaluation Pipeline ==============
    
    def run_full_evaluation(self, dataloader, features: np.ndarray, labels: np.ndarray, 
                          n_visualization_samples: int = 4) -> Dict[str, any]:
        """Run complete evaluation pipeline."""
        print("\n" + "="*60)
        print("RUNNING FULL MODEL EVALUATION")
        print("="*60 + "\n")
        
        results = {}
        
        # 1. Quantitative evaluation
        print("1. Quantitative Evaluation")
        print("-" * 30)
        eval_metrics = self.evaluate_representations(features, labels)
        for metric, value in eval_metrics.items():
            if 'accuracy' in metric:
                print(f"   {metric:<25}: {value:6.2f}%")
            else:
                print(f"   {metric:<25}: {value:6.3f}")
        results['metrics'] = eval_metrics
        
        # 2. Visualizations
        print("\n2. Generating Visualizations")
        print("-" * 30)
        
        # Cosine similarity
        sorted_features = features[np.argsort(labels)]
        fig_cos = self.plot_cosine_similarity_matrix(
            sorted_features,  # Limit for visibility
            save_path='cosine_similarity.png'
        )
        plt.close(fig_cos)
        print("   ✓ Cosine similarity matrix")
        
        # t-SNE
        fig_tsne = self.plot_tsne(
            features, labels,
            save_path='tsne_visualization.png'
        )
        plt.close(fig_tsne)
        print("   ✓ t-SNE visualization")
        
        # Top-k features
        fig_topk = self.plot_top_k_features(
            features, labels,
            save_path='top_k_features.png'
        )
        plt.close(fig_topk)
        print("   ✓ Top-k feature analysis")
        
        # 3. Sample visualizations
        print("\n3. Sample-wise Attention Analysis")
        print("-" * 30)
        self.visualize_batch(dataloader, n_samples=n_visualization_samples)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*60 + "\n")
        
        return results