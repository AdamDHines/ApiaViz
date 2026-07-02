"""Panoramic image renderer matching ImgGrabber.m.

Renders a panoramic view from a position in a 3D world made of polygon patches,
using matplotlib for rasterization with painter's algorithm depth ordering.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def _pi2pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle to [-pi, pi). Matches pi2pi helper in ImgGrabber.m."""
    x = np.mod(x, 2 * np.pi)
    x[x > np.pi] -= 2 * np.pi
    return x


def render_panorama(
    x: float,
    y: float,
    z: float,
    th: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    colp: np.ndarray,
    hfov: float = 296.0,
    resolution: float = 4.0,
) -> np.ndarray:
    """Render a panoramic image from a world model.

    Faithfully matches ImgGrabber.m:
    1. Convert world vertices to spherical coords relative to camera
    2. Rotate by heading
    3. Render patches with painter's algorithm (z-buffering by distance)
    4. Use green-only colormap (64 levels)
    5. Extract green channel, trim to FOV, subsample at resolution

    Args:
        x, y, z: Camera position in meters.
        th: Heading direction in degrees (0 = +x axis).
        X, Y, Z: World polygon vertices, shape (numPolygons, vertsPerPoly).
        colp: Vertex color values, shape (numPolygons, vertsPerPoly).
        hfov: Horizontal field of view in degrees.
        resolution: Image resolution in degrees/pixel.

    Returns:
        Grayscale image (uint8), shape depends on world/hfov/resolution.
    """
    z0 = 0.0

    # Convert to spherical coordinates relative to camera
    dx = X - x
    dy = Y - y
    dz = np.abs(Z) - z - z0

    R = np.sqrt(dx**2 + dy**2 + dz**2)
    TH = np.arctan2(dy, dx)  # azimuth
    PHI = np.arctan2(dz, np.sqrt(dx**2 + dy**2))  # elevation

    # Rotate by heading
    TH = _pi2pi(TH - np.radians(th))

    # Split polygons crossing the ±π boundary
    # MATLAB: ind = (max(TH') - min(TH') < pi)
    # This checks if the angular span of each polygon is < π
    th_max = np.max(TH, axis=1)
    th_min = np.min(TH, axis=1)
    ind = (th_max - th_min) < np.pi

    # Non-wrapping polygons
    A1 = TH[ind]
    E1 = PHI[ind]
    D1 = R[ind]
    c1 = colp[ind]

    # Wrapping polygons — need duplicates
    A2 = TH[~ind]
    E2 = PHI[~ind]
    D2 = R[~ind]
    c2 = colp[~ind]

    # Create two copies of wrapping polygons shifted by ±2π
    A3 = A2.copy()
    A3[A3 <= 0] += 2 * np.pi
    A4 = A2.copy()
    A4[A4 > 0] -= 2 * np.pi

    # Render at 1°/pixel, full 360° panorama → 360 × 75 pixels
    imgwidth = 360
    imgheight = 75
    dpi = 1  # 1 pixel per point for exact sizing

    fig, ax = plt.subplots(
        figsize=(imgwidth, imgheight), dpi=dpi, frameon=False
    )
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi / 12, np.pi / 3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("cyan")

    # Green colormap: 64 levels from black to green
    # MATLAB: grasscolormap(:,2) = linspace(0,1,64)
    # colp values map to this colormap
    # We normalize colp to [0, 1] range for the green channel
    colp_min = np.min(colp)
    colp_max = np.max(colp)
    if colp_max > colp_min:
        colp_range = colp_max - colp_min
    else:
        colp_range = 1.0

    def _make_patches_and_sort(azimuth, elevation, distance, colors):
        """Create polygon patches sorted by distance (far to near)."""
        if len(azimuth) == 0:
            return [], []

        mean_dist = np.mean(distance, axis=1)
        # Sort far to near (painter's algorithm: draw far first)
        order = np.argsort(-mean_dist)

        patches = []
        facecolors = []
        for idx in order:
            verts = np.column_stack(
                [azimuth[idx], elevation[idx]]
            )
            # Mean color for the polygon
            mean_color = np.mean(colors[idx])
            # Map through green colormap
            green_val = (mean_color - colp_min) / colp_range
            green_val = np.clip(green_val, 0, 1)
            facecolors.append((0, green_val, 0))
            patches.append(MplPolygon(verts, closed=True))

        return patches, facecolors

    # Draw ground plane
    # MATLAB: Xp = [-10 -10 10.5 10.5]', Yp = [-pi pi pi -pi]'
    # Zp = atan2(-z-z0, Xp), groundcolor = [229 183 90]/255
    ground_az = np.array([-np.pi, np.pi, np.pi, -np.pi])
    ground_el = np.array(
        [
            np.arctan2(-z - z0, -10),
            np.arctan2(-z - z0, -10),
            np.arctan2(-z - z0, 10.5),
            np.arctan2(-z - z0, 10.5),
        ]
    )
    ground_verts = np.column_stack([ground_az, ground_el])
    ground_color = np.array([229, 183, 90]) / 255.0
    ax.add_patch(
        MplPolygon(ground_verts, closed=True, facecolor=ground_color, edgecolor="none")
    )

    # Draw all polygon sets
    all_polygon_sets = [
        (A1, E1, D1, c1),
        (A3, E2, D2, c2),
        (A4, E2, D2, c2),
    ]

    for az, el, dist, cols in all_polygon_sets:
        patches, facecolors = _make_patches_and_sort(az, el, dist, cols)
        if patches:
            pc = PatchCollection(patches, facecolors=facecolors, edgecolors="none")
            ax.add_collection(pc)

    # Render to array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    plt.close(fig)

    # Extract green channel (matching MATLAB: img = img(:,:,2))
    img = img_array[:, :, 1]  # green channel (RGBA index 1)

    # Trim to FOV
    # MATLAB: img = img(:, 181-hfov/2 : 180+hfov/2)
    # For 360-wide image, center is at pixel 180 (0-indexed: 179)
    if hfov != 360:
        half_fov = int(hfov / 2)
        left = 180 - half_fov  # MATLAB 1-indexed: 181 - hfov/2
        right = 180 + half_fov  # MATLAB 1-indexed: 180 + hfov/2
        # In 0-indexed Python, MATLAB col 181-hfov/2 → col (180-hfov/2)
        # MATLAB col 180+hfov/2 → col (179+hfov/2)
        # Total width: hfov pixels
        img = img[:, left:right]

    # Subsample at resolution
    # MATLAB: blockproc(img, [4 4], fun) where fun takes mean of center 2x2
    if resolution == 4 and hfov != 360:
        block_h, block_w = 4, 4
        h, w = img.shape
        out_h = h // block_h
        out_w = w // block_w
        result = np.zeros((out_h, out_w), dtype=np.uint8)
        for bh in range(out_h):
            for bw in range(out_w):
                block = img[
                    bh * block_h : (bh + 1) * block_h,
                    bw * block_w : (bw + 1) * block_w,
                ]
                # MATLAB: mean(mean(block_struct.data(2:3, 2:3)))
                # 1-indexed rows 2:3, cols 2:3 → 0-indexed rows 1:3, cols 1:3
                center = block[1:3, 1:3].astype(np.float64)
                result[bh, bw] = np.uint8(np.mean(center))
        img = result

    return img
