"""Deterministic image perturbations shared by every representation."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

CONDITIONS = ("clean", "gray", "luminance", "chromatic", "noise")


def sample_key(position, heading):
    data = np.asarray([*position, heading], dtype="<f4").tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def perturb(images, condition="clean", severity=0., seed=0, keys=None):
    """[B,RGB,H,W] in [0,1]. Noise is indexed by stimulus, not batch order.

    Common-mode addition preserves channel differences only before clipping.
    Return measured clipping and opponent changes with the corrupted images.
    """
    if condition not in CONDITIONS or severity < 0:
        raise ValueError("Invalid corruption condition or severity")
    original = images
    if condition == "gray":
        images = images[:, -2:].mean(1, keepdim=True).expand_as(images)
    elif condition != "clean" and severity > 0:
        keys = range(len(images)) if keys is None else keys
        if len(keys) != len(images):
            raise ValueError("One perturbation key is required per image")
        fields = []
        for key in keys:
            gen = torch.Generator().manual_seed((seed + int(key)) % (2**63 - 1))
            h, w = images.shape[-2:]
            if condition == "noise":
                field = torch.randn(1, images.shape[1], h, w, generator=gen) * .15
            else:
                field = sum(F.interpolate(torch.randn(1, 1, max(1, h // scale), max(1, w // scale),
                                                      generator=gen), size=(h, w), mode="bilinear",
                                           align_corners=False) for scale in (2, 4, 8, 16))
                field = .2 * field / field.std(unbiased=False).clamp_min(1e-6)
                if condition == "chromatic":
                    weights = torch.zeros(1, images.shape[1], 1, 1)
                    weights[:, -2] = 1
                    weights[:, -1] = -1
                    field = field * weights
            fields.append(field[0])
        images = images + severity * torch.stack(fields).to(images.device)
    clipped = float(((images < 0) | (images > 1)).float().mean())
    images = images.clamp(0, 1)
    opponent_delta = ((images[:, -2] - images[:, -1]) -
                      (original[:, -2] - original[:, -1])).abs().mean()
    return images, {"clipped_fraction": clipped, "opponent_change_mae": float(opponent_delta)}


def flower_views(paths, obj=75, canvas=135):
    if not 0 < obj <= canvas:
        raise ValueError("Require 0 < object size <= canvas size")
    margin = (canvas - obj) // 2
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB").resize((obj, obj)), dtype=np.float32) / 255
        view = np.full((canvas, canvas, 3), .5, dtype=np.float32)
        view[margin:margin + obj, margin:margin + obj] = image
        images.append(torch.from_numpy(view).permute(2, 0, 1))
    return torch.stack(images)
