# Imports
import cv2
import math
import torch
import random

import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset

class KeepGB(torch.nn.Module):
    def forward(self, t):
        return t[1:3]          # keep G & B channels

class TinyImageNetPairDataset(Dataset):
    """
    Returns two independent augmentations of the *same* Tiny-ImageNet image,
    ready for SimCLR-style contrastive learning.

    Each item:  (view_1, view_2)   where view_i == transform(original_img)
    """
    def __init__(self, root, transform):
        self.root      = Path(root)
        self.transform = transform

        # tiny-imagenet-200/train/<wnid>/images/*.JPEG
        self.images = []
        self.images.extend(self.root.glob("*.jpg"))

        if len(self.images) == 0:
            raise RuntimeError(f"No JPEGs found in {self.root}")

        random.shuffle(self.images)

    def __len__(self) -> int:
        return len(self.images)      # 100 000 for Tiny-ImageNet train split

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")  # Tiny-IN files are RGB

        # two stochastic views
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2

class FacePatchDataset(Dataset):
    def __init__(self, root, patch=75, patches_per_file=3000):
        self.patch = patch
        self.imgs  = []   # list of (tensor, label)
        self.names = []

        t_img = transforms.ToTensor()
        root  = Path(root)
        for sub in ['female', 'male']:
            for fp in sorted((root/sub).glob('*.*')):
                # ignore .DS_Store files on macOS
                if fp.name.startswith('.DS_Store'):
                    continue
                img = Image.open(fp).convert('L')
                self.imgs.append((t_img(img), len(self.names)))
                self.names.append(fp.stem)

        if not self.imgs:
            raise RuntimeError("No images found under female/ or male/")

        self.per_file = patches_per_file
        self.total    = self.per_file * len(self.imgs)

    def __len__(self): return self.total

    def __getitem__(self, idx):
        img_t, label = self.imgs[idx // self.per_file]
        _, H, W = img_t.shape
        cx, cy  = W/2, H/2
        max_r   = 0.7 * min(W, H) / 2

        for _ in range(1000):
            x0 = random.randint(0, W - self.patch)
            y0 = random.randint(0, H - self.patch)
            if math.hypot(x0+self.patch/2 - cx, y0+self.patch/2 - cy) <= max_r:
                break

        patch = img_t[:, y0:y0+self.patch, x0:x0+self.patch]
        patch = patch.repeat(2,1,1)  # duplicate channel → 2×75×75
        return patch, label
    
class FlowerPatchDataset(Dataset):
    def __init__(self, root='./apiaviz/dataset/natural-scenes', patch=75, patches_per_file=3000):
        self.patch = patch
        self.per_file = patches_per_file
        
        # This list will hold tuples of (tensor, class_label)
        self.source_images = []
        
        # This dictionary will map folder names to integer labels
        self.class_names = ['summer', 'spring', 'fall']
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        t_img = transforms.ToTensor()
        root_path = Path(root)

        print("Loading dataset...")
        # Iterate through the defined classes to assign consistent labels
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = root_path / class_name
            if not class_dir.exists():
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Find all image files in the class directory
            image_files = [fp for fp in sorted(class_dir.glob('*.*')) if not fp.name.startswith('.')]
            print(f"Found {len(image_files)} images in '{class_name}' (Label: {class_idx})")

            for fp in image_files:
                try:
                    img = Image.open(fp).convert('RGB')
                    # Store the full image tensor along with its correct class label
                    self.source_images.append((t_img(img), class_idx))
                except Exception as e:
                    print(f"Warning: Could not load image {fp}. Error: {e}")

        if not self.source_images:
            raise RuntimeError(f"No images found under the specified root: {root}")

        # The total number of patches is the number of source images * patches per image
        self.total = self.per_file * len(self.source_images)
        print(f"Dataset loaded. Total source images: {len(self.source_images)}. Total patches: {self.total}.")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Determine which source image to use based on the index
        source_idx = idx // self.per_file
        
        # Retrieve the full image tensor and its correct class label
        img_tensor, label = self.source_images[source_idx]
        
        # Perform the random crop on the full image
        _, H, W = img_tensor.shape
        y0 = random.randint(0, H - self.patch)
        x0 = random.randint(0, W - self.patch)
        crop = img_tensor[:, y0:y0+self.patch, x0:x0+self.patch]
        
        # Keep only the Green and Blue channels
        crop_gb = crop[1:3]
        
        # Return the cropped patch and the correct folder-level label
        return crop_gb, label

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