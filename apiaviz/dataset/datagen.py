# Imports
import os
import cv2
import math
import torch
import random

import numpy as np

from pathlib import Path
from enum import Enum, auto
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset
from apiaviz.src.functional import generate_smooth_scan_path

class TinyImageNetPairDataset(Dataset):
    """
    Returns two independent augmentations of the same Tiny-ImageNet image.
    
    In SNN mode, this class generates two temporally-linked spike trains by
    applying a single, shared, smooth scanning path to both augmented views.
    This is suitable for contrastive learning frameworks like SimCLR.
    """
    def __init__(self, root: str, transform: transforms.Compose, 
                 snn_mode: bool = False, 
                 num_steps: int = 50):
        
        super().__init__()
        self.root = Path(root)
        self.base_transform = transform
        self.snn_mode = snn_mode

        if self.snn_mode:
            self.num_steps = num_steps
        
        self.images = sorted(list(self.root.glob("**/*.jpg")))
        if len(self.images) == 0:
            raise RuntimeError(f"No JPEGs found in {self.root} or its subdirectories")
        random.shuffle(self.images)

    def __len__(self) -> int:
        return len(self.images)
    
    def bernoulli_spikes(self, x, rate_scale=1.5):
        # x ∈ [0,1]; optional scale controls average firing
        p = (x * rate_scale).clamp_(0, 1)
        return (torch.rand_like(p) < p).float()

    def _create_spike_train(self, static_tensor) -> torch.Tensor:
        """Generates a spike train from a static tensor using a pre-defined path."""
        frames = []
        for _ in range(self.num_steps):
            frames.append(self.bernoulli_spikes(static_tensor))
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Create two different augmented views of the same image
        v1_static = self.base_transform(img)
        v2_static = self.base_transform(img)

        if self.snn_mode:
            v1 = self._create_spike_train(v1_static)
            v2 = self._create_spike_train(v2_static)
        else:
            # In ANN mode, just return the static augmented tensors
            v1 = v1_static
            v2 = v2_static
            
        return v1, v2

class DataMode(Enum):
    """Defines the operating mode for the dataset."""
    STATIC_FULL = auto()      # Returns a single, full-size static image.
    STATIC_PATCH = auto()     # Returns a single, random static patch from an image.
    SCANNING_PATCH = auto()   # Returns a time-series of patches scanning across an image.

class InsectVisionDataset(Dataset):
    """
    A unified dataset for loading insect vision data in various formats.

    This class can operate in three modes, configurable via the `mode` parameter:
    1.  STATIC_FULL:    Yields the entire source image, resized if necessary.
    2.  STATIC_PATCH:   Yields a single random patch from a source image.
    3.  SCANNING_PATCH: Yields a time-series of patches that form a smooth
                        scan path across the source image.
    """
    def __init__(self, root, dataset, mode = DataMode, patch_size = None, samples_per_image = 100, num_steps = None):
        """
        Args:
            root (str): The root directory of the dataset.
            mode (DataMode): The operating mode (STATIC_FULL, STATIC_PATCH, or SCANNING_PATCH).
            patch_size (Optional[int]): The size (height and width) of the patches.
                                        Required for STATIC_PATCH and SCANNING_PATCH modes.
            samples_per_image (int): How many samples (patches or scan paths) to generate
                                     per source image. Not used in STATIC_FULL mode.
            num_steps (Optional[int]): The number of steps in a scanning time-series.
                                       Required for SCANNING_PATCH mode.
        """
        combined_root = os.path.join(root, dataset)
        self.root = Path(combined_root)
        self.dataset = dataset
        self.mode = mode
        self.patch_size = patch_size
        self.samples_per_image = samples_per_image
        self.num_steps = num_steps

        # --- Validate mode-specific arguments ---
        if self.mode in [DataMode.STATIC_PATCH, DataMode.SCANNING_PATCH]:
            if self.patch_size is None:
                raise ValueError("`patch_size` must be provided for patch-based modes.")
        if self.mode == DataMode.SCANNING_PATCH:
            if self.num_steps is None:
                raise ValueError("`num_steps` must be provided for SCANNING_PATCH mode.")

        # --- Load image paths and class names ---
        self.source_images = []
        if self.dataset == "flowers":
            self.class_names = ['lavender-resized', 'sunflower-resized', 'rose-resized']
        else:
            self.class_names = ['goldfish1','goldfish2','ball','roads','car','fruit','bird']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.to_tensor = transforms.ToTensor()

        print("Loading dataset...")
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            print(class_dir)
            if not class_dir.is_dir():
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            for fp in sorted(class_dir.glob('*.*')):
                if fp.name.startswith('.'): continue
                try:
                    # Keep images as PIL objects to save memory, convert to tensor on the fly
                    img = Image.open(fp).convert('RGB')
                    self.source_images.append((img, class_idx))
                except Exception as e:
                    print(f"Warning: Could not load image {fp}. Error: {e}")

        if not self.source_images:
            raise RuntimeError(f"No images found under the specified root: {self.root}")

        print(f"Dataset loaded. Total source images: {len(self.source_images)}.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        if self.mode == DataMode.STATIC_FULL:
            return len(self.source_images)
        else:
            return len(self.source_images) * self.samples_per_image

    def __getitem__(self, idx):
        """
        Returns a single data sample, formatted consistently across all modes.

        Returns:
            Tuple[torch.Tensor, int, torch.Tensor, list]: A tuple containing:
            - input_tensor: The data to be fed into the model (full image, patch, or time-series).
            - label: The integer class label.
            - source_image: The original full-resolution source image (as a tensor).
            - scan_path: A tuple of (path_x, path_y) for scanning mode, otherwise an empty list.
        """
        # Determine which source image to use
        if self.mode == DataMode.STATIC_FULL:
            source_idx = idx
        else:
            source_idx = idx // self.samples_per_image

        source_img_pil, label = self.source_images[source_idx]
        source_img_tensor = self.to_tensor(source_img_pil)
        
        # --- Mode-specific logic ---

        if self.mode == DataMode.STATIC_FULL:
            # The input is the full image itself (GB channels only)
            input_tensor = source_img_tensor[1:3]
            scan_path = []
            return input_tensor, label, source_img_tensor, scan_path

        if self.mode == DataMode.STATIC_PATCH:
            _, H, W = source_img_tensor.shape
            max_y = H - self.patch_size
            max_x = W - self.patch_size
            
            # Generate a single random crop
            y0 = random.randint(0, max_y)
            x0 = random.randint(0, max_x)
            patch = source_img_tensor[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size]
            
            # Input is the patch (GB channels)
            input_tensor = patch[1:3]
            scan_path = []
            return input_tensor, label, source_img_tensor, scan_path

        if self.mode == DataMode.SCANNING_PATCH:
            _, H, W = source_img_tensor.shape
            max_y = H - self.patch_size
            max_x = W - self.patch_size
            
            # Generate a smooth path for the patch
            path_x, path_y = generate_smooth_scan_path(self.num_steps, max_x, max_y)
            
            patches = []
            for i in range(self.num_steps):
                y0, x0 = path_y[i], path_x[i]
                crop = source_img_tensor[:, y0:y0 + self.patch_size, x0:x0 + self.patch_size]
                patches.append(crop[1:3]) # Keep only Green and Blue channels
            
            # Input is the stacked time-series of patches
            input_tensor = torch.stack(patches)
            scan_path = (path_x, path_y)
            return input_tensor, label, source_img_tensor, scan_path
            
        raise NotImplementedError(f"Mode {self.mode} is not implemented.")

class SyntheticDataset(Dataset):
    """
    Synthetic dataset generator - currently set up for RandomDot

    Args
    ----
    num_samples       : number of samples in the dataset
    image_transform   : torchvision transform for images
    eval              : if True, return single view + label
    no_inner_symbols  : omit shapes/dots inside reward zone
    green_pct         : percentage of green dots in eval scenes
    """
    # ────────── constructor ──────────
    def __init__(self,
                 num_samples: int,
                 image_transform=None,
                 eval: bool = False,
                 no_inner_symbols: bool = False,
                 green_pct_high: int = 80,
                 green_pct_low: int = 20):
        super().__init__()

        self.num_samples      = num_samples
        self.image_transform  = image_transform
        self.eval             = eval
        self.no_inner_symbols = no_inner_symbols
        self.green_hi        = green_pct_high
        self.green_lo        = green_pct_low

        # arena geometry
        self.img_size      = 900
        self.patch_size    = 75
        self.center        = (self.img_size // 2, self.img_size // 2)
        self.outer_radius  = 200
        self.small_radius  = 37.5
        self.buffer        = 5
        self.border_width  = 2

        # stimulus parameters
        self.min_distance  = 10
        self.num_shapes    = 1800
        self.shape_types   = ['circle', 'square', 'triangle', 'cross']
        self.shape_area_px = 80
        self.dot_area_px   = 80
        self.shape_radius  = self._rad_from_area(self.shape_area_px)
        self.dot_radius    = self._rad_from_area(self.dot_area_px)
        self.num_dots      = 400

    # ────────── helpers ──────────
    def __len__(self): return self.num_samples

    @staticmethod
    def _rad_from_area(area_px: float) -> float:
        return float(np.sqrt(area_px / math.pi))

    # placement utilities ------------------------------------------------
    def _rand_positions(self, n, min_dist, exclude_r, max_r):
        xs, ys = [], []
        attempts, limit = 0, n * 100
        while len(xs) < n and attempts < limit:
            φ, ρ = random.random() * 2 * math.pi, random.uniform(exclude_r, max_r)
            x, y = ρ * math.cos(φ), ρ * math.sin(φ)
            if self.no_inner_symbols and math.hypot(x, y) < self.small_radius:
                attempts += 1; continue
            if all(math.hypot(x - xi, y - yi) >= min_dist for xi, yi in zip(xs, ys)):
                xs.append(x); ys.append(y)
            attempts += 1
        return np.array(xs), np.array(ys)

    def _rand_dots(self, n, green_pct, min_dist, exclude_r, max_r):
        xs, ys = self._rand_positions(n, min_dist, exclude_r, max_r)
        n_g = int(n * green_pct / 100)
        colours = ['green'] * n_g + ['blue'] * (n - n_g)
        random.shuffle(colours)
        return xs, ys, colours

    # drawing helpers ----------------------------------------------------
    def _draw_circle (self, d, cx, cy, r, col): d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)
    def _draw_square (self, d, cx, cy, r, col):
        d.polygon([(cx-r,cy-r),(cx+r,cy-r),(cx+r,cy+r),(cx-r,cy+r)], fill=col)
    def _draw_triangle(self, d, cx, cy, r, col):
        h = r * math.sqrt(3)
        d.polygon([(cx,cy-2*h/3),(cx-r,cy+h/3),(cx+r,cy+h/3)], fill=col)
    def _draw_cross (self, d, cx, cy, r, col):
        t = max(1,int(r*0.3))
        d.line([(cx-r,cy-r),(cx+r,cy+r)], fill=col, width=t)
        d.line([(cx-r,cy+r),(cx+r,cy-r)], fill=col, width=t)
    def _dispatch(self, d, typ, *args): getattr(self, f"_draw_{typ}")(d, *args)

    # pink-dot background ------------------------------------------------
    def _background(self):
        bg = np.full((self.img_size, self.img_size, 3), 255, np.uint8)
        for _ in range(4000):
            px, py = random.randint(0, self.img_size-1), random.randint(0, self.img_size-1)
            if (px - self.center[0])**2 + (py - self.center[1])**2 < self.outer_radius**2:
                continue
            cv2.circle(bg, (px, py), 3, (230, 161, 161), -1)
        return Image.fromarray(bg)

    def _region(self, x, y):
        """
        Region code for a patch whose top-left corner is (x,y):
            0  = centre disk
            1  = ring           (60 ≤ radius ≤ 170 px)
            2  = background
        """
        cx = x + self.patch_size / 2 - self.center[0]
        cy = y + self.patch_size / 2 - self.center[1]
        d  = math.hypot(cx, cy)
        if d <= self.small_radius:     # inner black disk
            return 0
        if 60 <= d <= 170:             # annulus
            return 1
        return 2                       # everything else

    # Add this to your class (outside any method) to track saves
    _saved_examples = { 'green_hi': False, 'green_lo': False }

    # Modify the _full_stimulus method:
    def _full_stimulus(self, return_green_pct=False):
        img = Image.new("RGBA", (self.img_size, self.img_size))
        img.paste(self._background().convert("RGBA"), (0, 0))
        d = ImageDraw.Draw(img)

        # arena rings
        d.ellipse([self.center[0]-self.outer_radius, self.center[1]-self.outer_radius,
                self.center[0]+self.outer_radius, self.center[1]+self.outer_radius],
                fill=(255,255,255,255), outline="black")
        d.ellipse([self.center[0]-self.small_radius, self.center[1]-self.small_radius,
                self.center[0]+self.small_radius, self.center[1]+self.small_radius],
                fill=(0,0,0,255), outline="black")

        # eval mode: coloured dots
        green_pct   = random.choice([self.green_hi, self.green_lo])
        colour_idx  = 0 if green_pct == self.green_hi else 1
        if self.eval:
            excl = self.small_radius + self.buffer + self.dot_radius + self.border_width
            maxR = self.outer_radius - self.dot_radius - self.border_width
            xs, ys, cols = self._rand_dots(self.num_dots,
                                        green_pct,
                                        self.min_distance,
                                        excl, maxR)
            for x_off, y_off, col in zip(xs, ys, cols):
                colour = (0,255,0,255) if col == 'green' else (0,0,255,255)
                bbox = [self.center[0]+x_off-self.dot_radius,
                        self.center[1]+y_off-self.dot_radius,
                        self.center[0]+x_off+self.dot_radius,
                        self.center[1]+y_off+self.dot_radius]
                d.ellipse(bbox, fill=colour)

            rgb = img.convert("RGB")

            # Save image once for each green condition
            if not self._saved_examples['green_hi'] and green_pct == self.green_hi:
                rgb.save("example_high_green.png")
                self._saved_examples['green_hi'] = True
            elif not self._saved_examples['green_lo'] and green_pct == self.green_lo:
                rgb.save("example_high_blue.png")
                self._saved_examples['green_lo'] = True

            return (rgb, green_pct) if return_green_pct else (rgb, int(green_pct == self.green_hi))

        # training mode: random shapes
        else:
            R,G,B = [random.randint(0,255) for _ in range(3)]
            label = int(np.argmax([R,G,B]))
            colour = (R,G,B,255)
            excl = self.small_radius + self.buffer + self.shape_radius + self.border_width
            maxR = self.outer_radius - self.shape_radius - self.border_width
            xs, ys = self._rand_positions(self.num_shapes,
                                        self.min_distance,
                                        excl, maxR)
            for x_off, y_off in zip(xs, ys):
                cx, cy = self.center[0]+x_off, self.center[1]+y_off
                typ = random.choice(self.shape_types)
                self._dispatch(d, typ, cx, cy, self.shape_radius, colour)

        return img.convert("RGB"), colour_idx

    # ────────── __getitem__ ──────────
    def __getitem__(self, _):
        full_img, colour_idx = self._full_stimulus()

        # random crop
        x0 = random.randint(0, self.img_size - self.patch_size)
        y0 = random.randint(0, self.img_size - self.patch_size)
        patch = full_img.crop((x0, y0,
                               x0 + self.patch_size,
                               y0 + self.patch_size))

        if not self.eval:
            v1 = self.image_transform(patch) if self.image_transform else patch
            v2 = self.image_transform(patch) if self.image_transform else patch
            return v1, v2

        # region label from coordinates
        region = self._region(x0, y0)           # 0 / 1 / 2
        combo  = region * 2 + colour_idx        # 0-5 as before
        patch  = self.image_transform(patch) if self.image_transform else patch
        return patch, combo
    
class BalancedEvalVisionDataset(SyntheticDataset):
    def __init__(self, num_samples: int, *args, **kwargs):
        self.samples_per_region = num_samples
        self.image_transform = kwargs.get("image_transform", None)

        # request many extra samples so we can cherry-pick region + green pct
        super().__init__(num_samples=100 * num_samples, *args, eval=True, **kwargs)

        self._generate_balanced_samples()

    def _generate_balanced_samples(self):
        self.balanced_patches = []
        region_green_counts = {(r, g): 0 for r in range(3) for g in range(2)}  # (region, green_level): count

        while min(region_green_counts.values()) < self.samples_per_region:
            full_img, green_pct = self._full_stimulus(return_green_pct=True)
            green_level = 1 if green_pct == self.green_hi else 0  # 1 = high, 0 = low

            for _ in range(50):  # Try several patches from this stimulus
                x0 = random.randint(0, self.img_size - self.patch_size)
                y0 = random.randint(0, self.img_size - self.patch_size)
                region = self._region(x0, y0)
                key = (region, green_level)

                if region_green_counts[key] < self.samples_per_region:
                    label = region * 2 + green_level
                    patch = full_img.crop((x0, y0, x0 + self.patch_size, y0 + self.patch_size))
                    self.balanced_patches.append((patch, label))
                    region_green_counts[key] += 1
                    break  # Go to next image after successful add

        assert all(v == self.samples_per_region for v in region_green_counts.values()), "Unbalanced region/green counts"

    def __len__(self):
        return len(self.balanced_patches)

    def __getitem__(self, idx):
        patch, label = self.balanced_patches[idx]
        if self.image_transform:
            patch = self.image_transform(patch)
        return patch, label