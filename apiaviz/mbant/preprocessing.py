"""Image preprocessing pipeline matching ant_route_following_test.m."""

import numpy as np
import torch

try:
    from skimage.exposure import equalize_adapthist
    from skimage.transform import resize as skimage_resize
except ImportError:
    import cv2

    equalize_adapthist = None
    skimage_resize = None


def preprocess_image(
    raw_img: np.ndarray,
    target_shape: tuple[int, int] = (10, 36),
) -> np.ndarray:
    """Preprocess a single raw image to a normalized vector.

    Matches ant_route_following_test.m lines 98-103:
        1. Resize to [10, 36]
        2. Invert: 1 - img/255
        3. Adaptive histogram equalization (CLAHE)
        4. Reshape to (360, 1)

    Args:
        raw_img: Raw grayscale image (uint8).
        target_shape: (height, width) resize target.

    Returns:
        Preprocessed image as float64 array, shape (360,).
    """
    # Resize to target shape
    # skimage resize returns float in [0, 1] by default;
    # we need uint8 values for the inversion step, so we match MATLAB's imresize
    if skimage_resize is not None:
        img_resized = skimage_resize(
            raw_img, target_shape, order=1, preserve_range=True, anti_aliasing=True
        ).astype(np.float64)
    else:
        interpolation = (
            cv2.INTER_AREA
            if raw_img.shape[0] > target_shape[0] or raw_img.shape[1] > target_shape[1]
            else cv2.INTER_LINEAR
        )
        img_resized = cv2.resize(
            raw_img.astype(np.float32),
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=interpolation,
        ).astype(np.float64)

    # Invert: 1 - double(img)/255
    img_inverted = 1.0 - img_resized / 255.0

    # Adaptive histogram equalization (MATLAB: adapthisteq)
    # Clip to [0,1] for equalize_adapthist
    img_clipped = np.clip(img_inverted, 0, 1)
    if equalize_adapthist is not None:
        img_equalized = equalize_adapthist(img_clipped)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_equalized = clahe.apply(np.round(img_clipped * 255.0).astype(np.uint8)).astype(np.float64) / 255.0

    # Reshape to vector
    img_vector = img_equalized.reshape(-1)

    return img_vector


def normalize_and_scale(
    vectors: np.ndarray,
    C_I_PN_var: float = 5250.0,
) -> np.ndarray:
    """L2-normalize and scale input vectors.

    Matches ant_route_following_test.m lines 107-109:
        training_inputs = training_inputs ./ repmat(sqrt(sum(training_inputs.^2)), numPN, 1)
        training_inputs = training_inputs * C_I_PN_var

    Args:
        vectors: Input vectors, shape (numPN, numImages) or (numPN,).
        C_I_PN_var: Scaling parameter.

    Returns:
        Normalized and scaled vectors, same shape as input.
    """
    if vectors.ndim == 1:
        norm = np.sqrt(np.sum(vectors**2))
        if norm > 0:
            vectors = vectors / norm
        return vectors * C_I_PN_var

    # Column-wise L2 normalization
    norms = np.sqrt(np.sum(vectors**2, axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors * C_I_PN_var


def preprocess_single_for_network(
    raw_img: np.ndarray,
    C_I_PN_var: float = 5250.0,
    target_shape: tuple[int, int] = (10, 36),
) -> np.ndarray:
    """Full preprocessing for a single image (used during navigation).

    Matches the navigation test loop preprocessing:
        1. Resize to [10, 36]
        2. Invert: 1 - double(img)/255
        3. adapthisteq
        4. Reshape to (360, 1)
        5. Per-image L2 normalize: img / sqrt(sum(img^2))
        6. Scale by C_I_PN_var
    """
    vec = preprocess_image(raw_img, target_shape)
    return normalize_and_scale(vec, C_I_PN_var)


def preprocess_training_images(
    raw_images: list[np.ndarray],
    C_I_PN_var: float = 5250.0,
    target_shape: tuple[int, int] = (10, 36),
) -> np.ndarray:
    """Preprocess all training images and batch-normalize.

    Matches ant_route_following_test.m lines 96-109:
    Each image preprocessed individually, then batch L2-normalized.

    Args:
        raw_images: List of raw grayscale images (uint8).
        C_I_PN_var: Input scaling parameter.
        target_shape: (height, width) resize target.

    Returns:
        training_inputs as float64 array, shape (numPN, numImages).
    """
    numPN = target_shape[0] * target_shape[1]
    num_images = len(raw_images)

    training_inputs = np.zeros((numPN, num_images), dtype=np.float64)
    for i, img in enumerate(raw_images):
        training_inputs[:, i] = preprocess_image(img, target_shape)

    # Batch L2 normalization and scaling
    return normalize_and_scale(training_inputs, C_I_PN_var)


def to_torch_input(
    img_vector: np.ndarray,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert preprocessed image vector to torch tensor.

    Args:
        img_vector: Preprocessed image, shape (numPN,) or (numPN, numImages).
        device: Target torch device.

    Returns:
        Float32 tensor on specified device.
    """
    if device is None:
        device = torch.device("cpu")
    return torch.tensor(img_vector, dtype=torch.float32, device=device)
