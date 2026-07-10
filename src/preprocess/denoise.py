"""
Post-enhancement denoising.

Amplifying a near-zero-signal PSR patch amplifies its noise floor along with
it. Rather than trying to fix this inside the network (which risks
hallucinating detail -- the exact failure mode zero-reference approaches are
meant to avoid), denoising is done as a classical, deterministic
post-process: Non-Local Means for texture-preserving noise removal, then
CLAHE for local contrast recovery. Both are standard OpenCV operations;
this module just wraps them with sane defaults for this data.
"""
from __future__ import annotations

import cv2
import numpy as np


def nlm_denoise(
    image_u8: np.ndarray,
    h: float = 10.0,
    template_window: int = 7,
    search_window: int = 21,
) -> np.ndarray:
    """
    Non-Local Means denoising on an 8-bit image.

    Args:
        image_u8: (H, W) or (H, W, 3) uint8 image.
        h: filter strength; higher removes more noise but can smear texture.
        template_window: size of the patch used to compute weights.
        search_window: size of the neighborhood searched for similar patches.
    """
    if image_u8.ndim == 2:
        return cv2.fastNlMeansDenoising(image_u8, None, h, template_window, search_window)
    return cv2.fastNlMeansDenoisingColored(
        image_u8, None, h, h, template_window, search_window
    )


def clahe_contrast(
    image_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization, applied on the L
    channel in LAB space so color balance from `ColorConstancyLoss` isn't
    disturbed.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image_u8.ndim == 2:
        return clahe.apply(image_u8)

    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def postprocess(
    enhanced_float: np.ndarray,
    nlm_h: float = 10.0,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Full post-process chain: float [0,1] -> denoise -> local contrast ->
    float [0,1] output.
    """
    u8 = np.clip(enhanced_float * 255.0, 0, 255).astype(np.uint8)
    denoised = nlm_denoise(u8, h=nlm_h)
    contrasted = clahe_contrast(denoised, clip_limit=clip_limit)
    return contrasted.astype(np.float32) / 255.0
