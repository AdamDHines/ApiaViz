"""Shared Torch utilities for rendered ant-route navigation."""

from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from apiaviz.mbant.config import NavigationConfig

from skimage.exposure import equalize_adapthist as _skimage_equalize_adapthist
from skimage.transform import resize as _skimage_resize

def normalize_codes(codes: torch.Tensor) -> torch.Tensor:
    x = codes.detach().float().clamp_min(0.0)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)

class CosineRouteMemory(torch.nn.Module):
    def __init__(
        self,
        route_codes: torch.Tensor,
        mode: str = "max_cosine",
        topk: int = 5,
        softmax_temperature: float = 0.05,
    ):
        super().__init__()
        self.mode = str(mode)
        self.topk = int(topk)
        self.softmax_temperature = float(softmax_temperature)
        self.register_buffer("memory", normalize_codes(route_codes))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        x = normalize_codes(codes).to(self.memory.device)
        similarity = x @ self.memory.T
        if self.mode == "max_cosine":
            familiarity = similarity.max(dim=1).values
        elif self.mode == "topk_cosine":
            k = min(max(1, self.topk), similarity.size(1))
            familiarity = similarity.topk(k, dim=1).values.mean(dim=1)
        elif self.mode == "softmax_cosine":
            temperature = max(self.softmax_temperature, 1e-6)
            familiarity = temperature * torch.logsumexp(similarity / temperature, dim=1)
        else:
            raise ValueError(f"Unsupported memory mode: {self.mode}")
        return -familiarity

def render_route_centers(
    renderer: TorchWorldRenderer,
    img_pos: np.ndarray,
    heading: np.ndarray,
    eye_height: float,
) -> torch.Tensor:
    positions = torch.as_tensor(img_pos[:-1], dtype=torch.float32, device=renderer.device)
    headings = torch.as_tensor(heading, dtype=torch.float32, device=renderer.device)
    return renderer.render_batch(positions, headings, eye_height=eye_height)


def corridor_offsets(n_viewpoints: int, halfwidth: float) -> np.ndarray:
    """Lateral offsets (m) for a corridor of parallel walks either side of the centreline.

    ``n_viewpoints == 1`` -> ``[0.0]`` (the classic single-viewpoint centreline memory).
    ``n_viewpoints  > 1`` -> ``n_viewpoints`` offsets evenly spaced across
    ``[-halfwidth, +halfwidth]`` (the centreline included when the count is odd).
    """
    n = max(1, int(n_viewpoints))
    if n == 1:
        return np.asarray([0.0], dtype=np.float32)
    return np.linspace(-float(halfwidth), float(halfwidth), n, dtype=np.float32)


def _lateral_position(pos: np.ndarray, heading_deg: float, lateral_m: float) -> np.ndarray:
    """Shift ``pos`` by ``lateral_m`` along the left-hand normal of ``heading_deg``."""
    rad = math.radians(float(heading_deg))
    normal = np.array([-math.sin(rad), math.cos(rad)], dtype=np.float32)
    return np.asarray(pos, dtype=np.float32) + normal * float(lateral_m)


def route_corridor_bank(
    img_pos: np.ndarray,
    heading: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Corridor memory positions/headings, ordered route-index major, lateral-walk minor.

    Each route index shares its heading across all lateral walks; the offset shifts the position
    along the route normal. The route-index-major ordering means an ``MBONPopulation`` with a
    contiguous partition packs every lateral viewpoint of a route stretch into the same MBON, so a
    single readout neuron learns the whole corridor cross-section it is responsible for.
    """
    positions: list[np.ndarray] = []
    headings: list[float] = []
    for idx in range(len(heading)):
        h = float(heading[idx])
        for offset in offsets:
            positions.append(_lateral_position(img_pos[idx], h, float(offset)))
            headings.append(h)
    return np.asarray(positions, dtype=np.float32), np.asarray(headings, dtype=np.float32)


def render_route_corridor(
    renderer: TorchWorldRenderer,
    img_pos: np.ndarray,
    heading: np.ndarray,
    offsets: np.ndarray,
    eye_height: float,
) -> torch.Tensor:
    """Render the corridor memory bank -> raw images ``[len(offsets) * num_pos, ...]``.

    With a single ``[0.0]`` offset this is identical to :func:`render_route_centers`.
    """
    positions, headings = route_corridor_bank(img_pos, heading, offsets)
    pos_t = torch.as_tensor(positions, dtype=torch.float32, device=renderer.device)
    head_t = torch.as_tensor(headings, dtype=torch.float32, device=renderer.device)
    return renderer.render_batch(pos_t, head_t, eye_height=eye_height)

def navigation_route_index_summary(nav: dict[str, Any]) -> dict[str, Any]:
    positions = np.asarray(nav["current_position"], dtype=np.float32)
    route = np.asarray(nav["trained_route"], dtype=np.float32)
    nearest = [int(np.linalg.norm(route - pos, axis=1).argmin()) for pos in positions]
    return {
        "nearest_route_index_min": int(min(nearest)) if nearest else 0,
        "nearest_route_index_max": int(max(nearest)) if nearest else 0,
        "nearest_route_index_final": int(nearest[-1]) if nearest else 0,
        "route_last_index": int(route.shape[0] - 1),
        "last20_nearest_route_indices": nearest[-20:],
    }

def selected_offset_summary(nav: dict[str, Any], nav_config: NavigationConfig) -> dict[str, Any]:
    indices = np.asarray(nav["selected_scan_indices"], dtype=int)
    if indices.size == 0:
        return {}
    offsets = nav_config.scan_range / 2.0 - indices.astype(np.float32) * nav_config.scan_step
    return {
        "selected_abs_offset_mean": float(np.abs(offsets).mean()),
        "selected_abs_offset_median": float(np.median(np.abs(offsets))),
        "selected_center_fraction": float((np.abs(offsets) <= 1e-6).mean()),
        "selected_within_10deg_fraction": float((np.abs(offsets) <= 10.0).mean()),
        "selected_offsets_last20": [float(v) for v in offsets[-20:].tolist()],
    }

def nav_result_summary(nav: dict[str, Any], nav_config: NavigationConfig) -> dict[str, Any]:
    return {
        **nav_summary(nav),
        **selected_offset_summary(nav, nav_config),
        "route_index": navigation_route_index_summary(nav),
    }

def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class Timer:
    def __init__(self):
        self.items: dict[str, float] = {}

    def add(self, key: str, elapsed: float) -> None:
        self.items[key] = self.items.get(key, 0.0) + float(elapsed)


class TorchWorldRenderer:
    """Torch renderer for the MBANT triangle world."""

    # Sky and ground RGB colors for chromatic rendering
    _SKY_RGB = (135.0, 206.0, 235.0)
    _GROUND_RGB = (80.0, 140.0, 60.0)

    def __init__(
        self,
        world_data: dict,
        device: torch.device,
        hfov: float = 296.0,
        resolution: float = 1.0,
        chunk_size: int = 256,
        color: bool = False,
        render_mode: str = "normal",
        triangle_color: "np.ndarray | torch.Tensor | None" = None,
    ):
        self.device = device
        self.hfov = float(hfov)
        self.resolution = float(resolution)
        self.chunk_size = int(chunk_size)
        self.color = bool(color)
        self.render_mode = str(render_mode)
        if self.render_mode not in {"normal", "binary_objects"}:
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        self.X = torch.as_tensor(world_data["X"], dtype=torch.float32, device=device)
        self.Y = torch.as_tensor(world_data["Y"], dtype=torch.float32, device=device)
        self.Z = torch.as_tensor(world_data["Z"], dtype=torch.float32, device=device)
        self.colp = torch.as_tensor(world_data["colp"], dtype=torch.float32, device=device)
        self.colp_min = self.colp.min()
        self.colp_range = (self.colp.max() - self.colp_min).clamp_min(1e-6)

        # Per-triangle grey value the colour renderer would paint (0-255), exposed so callers can
        # build colour landmarks on top of the exact grey baseline. Optional `triangle_color`
        # [n_tri, 3] overrides the grey obstacle fill with an explicit RGB per triangle when
        # color=True (e.g. isoluminant colour landmarks); None keeps the default grey behaviour.
        self.triangle_grey = (
            ((self.colp.mean(dim=1) - self.colp_min) / self.colp_range).clamp(0.0, 1.0) * 255.0
        )
        if triangle_color is None:
            self.triangle_color = None
        else:
            tc = torch.as_tensor(triangle_color, dtype=torch.float32, device=device)
            if tc.shape != (self.colp.shape[0], 3):
                raise ValueError(
                    f"triangle_color must be [n_tri, 3]=[{self.colp.shape[0]}, 3], got {tuple(tc.shape)}"
                )
            self.triangle_color = tc

        full_height = 75.0
        self.height = max(1, int(full_height / self.resolution))
        self.width = max(1, int(self.hfov / self.resolution))

        x = torch.linspace(
            -math.radians(self.hfov / 2.0),
            math.radians(self.hfov / 2.0),
            self.width,
            device=device,
        )
        y = torch.linspace(math.pi / 3.0, -math.pi / 12.0, self.height, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.grid_x = xx.flatten()
        self.grid_y = yy.flatten()

    @staticmethod
    def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi

    def _point_in_tri(self, ax, ay, bx, by, cx, cy) -> torch.Tensor:
        px = self.grid_x.unsqueeze(0)
        py = self.grid_y.unsqueeze(0)
        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        valid = denom.abs() > 1e-8
        denom = torch.where(valid, denom, torch.ones_like(denom))
        w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
        w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
        w3 = 1.0 - w1 - w2
        return valid & (w1 >= 0.0) & (w2 >= 0.0) & (w3 >= 0.0)

    def render_single(self, x: float, y: float, z: float, heading_deg: float) -> torch.Tensor:
        dx = self.X - float(x)
        dy = self.Y - float(y)
        dz = self.Z.abs() - float(z)

        radius = torch.sqrt(dx.square() + dy.square() + dz.square()).clamp_min(1e-6)
        az = self._wrap_pi(torch.atan2(dy, dx) - math.radians(float(heading_deg)))
        el = torch.atan2(dz, torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6))

        span = az.max(dim=1).values - az.min(dim=1).values
        nonwrap = span < math.pi

        az_parts = [az[nonwrap]]
        el_parts = [el[nonwrap]]
        radius_parts = [radius[nonwrap]]
        color_parts = [self.colp[nonwrap]]
        use_tri_rgb = self.color and self.triangle_color is not None
        rgb_parts = [self.triangle_color[nonwrap]] if use_tri_rgb else None

        if bool((~nonwrap).any()):
            az_wrap = az[~nonwrap]
            el_wrap = el[~nonwrap]
            radius_wrap = radius[~nonwrap]
            color_wrap = self.colp[~nonwrap]

            az_pos = az_wrap.clone()
            az_pos[az_pos <= 0.0] += 2.0 * math.pi
            az_neg = az_wrap.clone()
            az_neg[az_neg > 0.0] -= 2.0 * math.pi
            az_parts.extend([az_pos, az_neg])
            el_parts.extend([el_wrap, el_wrap])
            radius_parts.extend([radius_wrap, radius_wrap])
            color_parts.extend([color_wrap, color_wrap])
            if use_tri_rgb:
                rgb_wrap = self.triangle_color[~nonwrap]
                rgb_parts.extend([rgb_wrap, rgb_wrap])

        draw_az = torch.cat(az_parts, dim=0)
        draw_el = torch.cat(el_parts, dim=0)
        draw_radius = torch.cat(radius_parts, dim=0)
        draw_color_raw = torch.cat(color_parts, dim=0).mean(dim=1)
        if self.render_mode == "binary_objects":
            draw_color = torch.full_like(draw_color_raw, 255.0)
        else:
            draw_color = ((draw_color_raw - self.colp_min) / self.colp_range).clamp(0.0, 1.0) * 255.0

        order = torch.argsort(draw_radius.mean(dim=1), descending=True)
        draw_az = draw_az[order]
        draw_el = draw_el[order]
        draw_color = draw_color[order]
        draw_rgb = torch.cat(rgb_parts, dim=0)[order] if use_tri_rgb else None

        ground_horizon = math.atan2(-float(z), 10.5)
        n = self.grid_x.numel()

        if self.render_mode == "binary_objects":
            if self.color:
                image = torch.zeros((3, n), dtype=torch.float32, device=self.device)
            else:
                image = torch.zeros((n,), dtype=torch.float32, device=self.device)
        elif self.color:
            # 3-channel output: sky=blue, ground=green, triangles=grey
            image = torch.empty((3, n), dtype=torch.float32, device=self.device)
            for c, (sv, gv) in enumerate(zip(self._SKY_RGB, self._GROUND_RGB)):
                image[c].fill_(sv)
                image[c, self.grid_y <= ground_horizon] = gv
        else:
            image = torch.full((n,), 255.0, dtype=torch.float32, device=self.device)
            image[self.grid_y <= ground_horizon] = 183.0

        for start in range(0, draw_az.size(0), self.chunk_size):
            end = min(start + self.chunk_size, draw_az.size(0))
            a = draw_az[start:end]
            e = draw_el[start:end]
            inside = self._point_in_tri(
                a[:, 0:1],
                e[:, 0:1],
                a[:, 1:2],
                e[:, 1:2],
                a[:, 2:3],
                e[:, 2:3],
            )
            if not bool(inside.any()):
                continue
            local_order = torch.arange(
                1,
                inside.size(0) + 1,
                dtype=torch.int64,
                device=self.device,
            ).unsqueeze(1)
            winner = torch.where(inside, local_order, torch.zeros_like(local_order))
            winner_idx = winner.max(dim=0).values
            mask = winner_idx > 0
            cv = draw_color[start:end][winner_idx[mask] - 1]
            if self.color:
                if use_tri_rgb:
                    # Explicit per-triangle RGB (e.g. colour landmarks).
                    image[:, mask] = draw_rgb[start:end][winner_idx[mask] - 1].t()
                else:
                    # Triangles are grey: same value in all channels
                    image[:, mask] = cv.unsqueeze(0).expand(3, -1)
            else:
                image[mask] = cv

        if self.color:
            return image.view(3, self.height, self.width)
        return image.view(self.height, self.width)

    def render_batch(
        self,
        positions: torch.Tensor,
        headings_deg: torch.Tensor,
        eye_height: float,
    ) -> torch.Tensor:
        outputs = []
        for pos, heading in zip(positions.detach().cpu(), headings_deg.detach().cpu()):
            outputs.append(
                self.render_single(
                    float(pos[0]),
                    float(pos[1]),
                    float(eye_height),
                    float(heading),
                )
            )
        return torch.stack(outputs, dim=0)


def preprocess_original_torch(
    raw_images: torch.Tensor,
    c_i_pn_var: float,
    target_shape: tuple[int, int] = (10, 36),
) -> torch.Tensor:
    """
    Preprocess route images for original mbant retinotopic encoding.

    Matches the local upstream mbant pipeline:
        resize to 10x36, invert, adaptive histogram equalization, flatten,
        per-image L2 normalize, and scale.

    Accepts:
        [N, H, W]       grayscale batch
        [N, C, H, W]    channel-first batch
        [H, W]          single grayscale image
        [C, H, W]       single channel-first image

    Returns:
        [N, target_shape[0] * target_shape[1]]
    """

    input_device = raw_images.device
    images = raw_images.detach().float()

    # Single grayscale image: [H, W] -> [1, 1, H, W]
    if images.ndim == 2:
        images = images.unsqueeze(0).unsqueeze(0)

    # Either [N, H, W] grayscale batch OR [C, H, W] single image
    elif images.ndim == 3:
        # Treat [C, H, W] as a single image if C looks like channels
        if images.shape[0] in (1, 3, 4):
            images = images.unsqueeze(0)  # [1, C, H, W]
        else:
            images = images.unsqueeze(1)  # [N, 1, H, W]

    # Batched channel-first image: [N, C, H, W]
    elif images.ndim == 4:
        pass

    else:
        raise ValueError(
            f"Expected raw_images to have shape [H,W], [N,H,W], [C,H,W], "
            f"or [N,C,H,W], but got {tuple(raw_images.shape)}"
        )

    # Convert multi-channel images to grayscale.
    # Output becomes [N, 1, H, W].
    if images.shape[1] == 3:
        # Simple average is fine unless you specifically want RGB luminance weights.
        images = images.mean(dim=1, keepdim=True)

    elif images.shape[1] == 4:
        # Drop alpha, then average RGB.
        images = images[:, :3].mean(dim=1, keepdim=True)

    elif images.shape[1] != 1:
        raise ValueError(
            f"Expected channel dimension to be 1, 3, or 4, but got shape {tuple(images.shape)}"
        )

    images_np = images.squeeze(1).cpu().numpy()
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    rows = []
    for image in images_np:
        if _skimage_resize is not None and _skimage_equalize_adapthist is not None:
            resized = _skimage_resize(
                image,
                (target_h, target_w),
                order=1,
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float64)
            inverted = np.clip(1.0 - resized / 255.0, 0.0, 1.0)
            equalized = _skimage_equalize_adapthist(inverted).astype(np.float32)
        else:
            interpolation = (
                cv2.INTER_AREA
                if image.shape[0] > target_h or image.shape[1] > target_w
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(
                image.astype(np.float32),
                (target_w, target_h),
                interpolation=interpolation,
            )
            inverted = np.clip(1.0 - resized / 255.0, 0.0, 1.0)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(np.round(inverted * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
        rows.append(equalized.reshape(-1))

    flat = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=input_device)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return flat * float(c_i_pn_var)


def preprocess_apiaviz_torch(raw_images: torch.Tensor) -> torch.Tensor:
    """Pre-process rendered images for the ApiaViz backbone.

    Accepts:
        [N, H, W]    greyscale 0-255 — duplicates to 2 channels (G, B identical).
        [N, 3, H, W] RGB 0-255       — extracts G (channel 1) and B (channel 2).

    Returns:
        [N, 2, H, W] normalised to [-1, 1].
    """
    if raw_images.dim() == 3:
        img = (raw_images.float() / 255.0).clamp(0.0, 1.0)
        img = img.unsqueeze(1).repeat(1, 2, 1, 1)
    else:
        img = (raw_images.float() / 255.0).clamp(0.0, 1.0)
        img = img[:, 1:3, :, :]  # extract G and B channels
    return (img - 0.5) / 0.5


def navigate_torch(
    img_pos_np: np.ndarray,
    heading_np: np.ndarray,
    scorer,
    nav_config: NavigationConfig,
    max_navigation_steps: int | None = None,
    progress_interval: int = 0,
) -> dict[str, Any]:
    device = scorer.renderer.device
    img_pos = torch.as_tensor(img_pos_np, dtype=torch.float32, device=device)
    heading = torch.as_tensor(heading_np, dtype=torch.float32, device=device)
    num_pos = int(heading.numel())
    route_length = int(math.ceil((len(img_pos_np) - 1) * 10))
    if max_navigation_steps is not None:
        route_length = min(route_length, max(1, int(max_navigation_steps)))
    storage_length = route_length * 2 + 3

    current_position = torch.zeros((storage_length, 2), dtype=torch.float32, device=device)
    step_record = torch.zeros((3, storage_length), dtype=torch.float32, device=device)
    perf_measure = torch.zeros((storage_length,), dtype=torch.float32, device=device)
    current_position[0] = img_pos[0]

    feeder = img_pos[0].clone()
    nest = img_pos[-1].clone()
    record_pos: list[int] = []
    error_locations: list[torch.Tensor] = []
    en_responses: list[np.ndarray] = []
    selected_indices: list[int] = []
    moving = False
    current_pos_idx = 0
    step_count = 0
    last_filled_position = 0

    scan_offsets = (
        nav_config.scan_range / 2.0
        - torch.arange(nav_config.num_scan_img, dtype=torch.float32, device=device) * nav_config.scan_step
    )

    for _ in range(route_length):
        if step_count + 2 >= storage_length:
            break
        center_heading = heading[current_pos_idx] if not moving else step_record[0, step_count - 1]

        scan_headings = center_heading + scan_offsets
        scan = scorer.score(current_position[step_count], scan_headings, device)
        en_scores = scan["en"].detach()
        en_responses.append(en_scores.detach().cpu().numpy())
        winner = int(torch.argmin(en_scores).item())
        selected_indices.append(winner)
        selected_heading = scan_headings[winner]

        step_record[0, step_count] = selected_heading
        step_record[1, step_count] = en_scores[winner]
        step_record[2, step_count] = 1.0

        heading_rad = torch.deg2rad(selected_heading)
        move = torch.stack([torch.cos(heading_rad), torch.sin(heading_rad)]) * nav_config.step_size
        current_position[step_count + 1] = current_position[step_count] + move
        last_filled_position = max(last_filled_position, step_count + 1)
        moving = True

        if torch.linalg.norm(nest - current_position[step_count + 1]) <= nav_config.dis_threshold:
            break

        route_distance = torch.linalg.norm(img_pos[:num_pos] - current_position[step_count + 1], dim=1)
        dis_value, ind_pos_t = torch.min(route_distance, dim=0)
        ind_pos = int(ind_pos_t.item())
        if progress_interval > 0 and len(en_responses) % int(progress_interval) == 0:
            print(
                "navigation progress: "
                f"scan={len(en_responses)} step={step_count} route_idx={ind_pos}/{num_pos - 1} "
                f"error_count={len(error_locations)} en_min={float(en_scores.min().item()):.3f} "
                f"en_range={float((en_scores.max() - en_scores.min()).item()):.3f}",
                flush=True,
            )
        record_pos.append(ind_pos)

        if ind_pos >= num_pos - 1:
            break

        if float(dis_value.item()) > nav_config.dis_threshold:
            max_record = max(record_pos) if record_pos else 0
            if max_record < num_pos - 2:
                if ind_pos < max_record:
                    next_pos = min(max_record + 1, num_pos - 1)
                    record_pos.append(max_record + 1)
                else:
                    next_pos = min(ind_pos + round(nav_config.step_size * 10), num_pos - 1)
                    record_pos.append(ind_pos + 1)
                current_position[step_count + 2] = img_pos[next_pos]
                last_filled_position = max(last_filled_position, step_count + 2)
                current_pos_idx = next_pos
                perf_measure[step_count] = 1.0
                error_locations.append(current_position[step_count + 2].clone())
                step_count += 2
                moving = False
            else:
                break
        else:
            step_count += 1

    crop_len = max(last_filled_position + 1, 1)
    final_positions = current_position[:crop_len].detach().cpu().numpy()
    final_step_record = step_record[:, : max(step_count + 1, 1)].detach().cpu().numpy()
    error_np = (
        torch.stack(error_locations).detach().cpu().numpy()
        if error_locations
        else np.empty((0, 2), dtype=np.float32)
    )
    error_rate = float(perf_measure.sum().item() / max(step_count, 1))
    reached_nest = bool(np.linalg.norm(final_positions[-1] - img_pos_np[-1]) <= nav_config.dis_threshold)
    return {
        "step_record": final_step_record,
        "current_position": final_positions,
        "error_rate": error_rate,
        "error_location": error_np,
        "perf_measure": perf_measure.detach().cpu().numpy(),
        "EN_response": en_responses,
        "selected_scan_indices": selected_indices,
        "trained_route": img_pos_np,
        "feeder": feeder.detach().cpu().numpy(),
        "nest": nest.detach().cpu().numpy(),
        "reached_nest": reached_nest,
        "step_count": int(step_count),
    }


def nav_summary(nav: dict[str, Any]) -> dict[str, Any]:
    en = np.array(nav["EN_response"], dtype=object)
    scan_mins = np.array([float(np.min(row)) for row in en]) if len(en) else np.array([])
    scan_ranges = np.array([float(np.max(row) - np.min(row)) for row in en]) if len(en) else np.array([])
    scan_margins = []
    for row in en:
        scores = np.asarray(row, dtype=np.float32)
        if scores.size < 2:
            scan_margins.append(0.0)
        else:
            best_two = np.partition(scores, 1)[:2]
            scan_margins.append(float(best_two[1] - best_two[0]))
    scan_margins = np.asarray(scan_margins, dtype=np.float32)
    perf = np.asarray(nav["perf_measure"], dtype=np.float32)[: scan_margins.size]
    error_margins = scan_margins[perf > 0.0] if scan_margins.size and perf.size == scan_margins.size else np.array([])
    return {
        "error_rate": float(nav["error_rate"]),
        "error_count": int(nav["error_location"].shape[0]),
        "step_count": int(nav["step_count"]),
        "path_points": int(nav["current_position"].shape[0]),
        "reached_nest": bool(nav["reached_nest"]),
        "scan_count": int(len(nav["EN_response"])),
        "scan_min_mean": float(scan_mins.mean()) if scan_mins.size else 0.0,
        "scan_range_mean": float(scan_ranges.mean()) if scan_ranges.size else 0.0,
        "scan_margin_mean": float(scan_margins.mean()) if scan_margins.size else 0.0,
        "scan_margin_median": float(np.median(scan_margins)) if scan_margins.size else 0.0,
        "scan_margin_min": float(scan_margins.min()) if scan_margins.size else 0.0,
        "scan_margin_error_mean": float(error_margins.mean()) if error_margins.size else 0.0,
    }
