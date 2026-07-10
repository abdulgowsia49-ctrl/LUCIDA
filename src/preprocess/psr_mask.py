"""
Permanently-Shadowed-Region (PSR) masking utilities.

Identifies pixels/patches likely to be inside a permanently shadowed region
by thresholding raw OHRC digital numbers and rejecting patches that don't
have enough remaining structure to be useful (e.g. sensor cutoffs, dead
pixels). This is a lightweight, independent implementation -- geometric PSR
ground truth (from illumination modeling / SPICE) is preferred when
available; this module is a fast fallback based purely on pixel statistics.
"""
from __future__ import annotations

import numpy as np


def shadow_mask(image: np.ndarray, dn_threshold: float = 12.0) -> np.ndarray:
    """
    Boolean mask, True where a pixel is dark enough to plausibly be inside
    a PSR (near-zero digital number in the raw OHRC frame).

    Args:
        image: (H, W) or (H, W, C) array of raw digital numbers.
        dn_threshold: pixels at or below this DN value are marked shadowed.
    """
    gray = image if image.ndim == 2 else image.mean(axis=-1)
    return gray <= dn_threshold


def is_useful_patch(
    patch: np.ndarray,
    dn_threshold: float = 12.0,
    min_shadow_fraction: float = 0.6,
    min_nonzero_std: float = 0.5,
) -> bool:
    """
    Decide whether a patch is worth keeping for training/inference:
    mostly shadowed (so the model actually learns PSR statistics), but not
    a flat dead region with essentially no secondary-scatter signal.

    Args:
        patch: (H, W) or (H, W, C) raw DN patch.
        dn_threshold: see `shadow_mask`.
        min_shadow_fraction: minimum fraction of pixels that must be
            "shadowed" for the patch to count as PSR-representative.
        min_nonzero_std: minimum pixel-value std-dev required, to reject
            flat/saturated/dead-pixel patches.
    """
    mask = shadow_mask(patch, dn_threshold)
    shadow_fraction = float(mask.mean())
    if shadow_fraction < min_shadow_fraction:
        return False

    gray = patch if patch.ndim == 2 else patch.mean(axis=-1)
    return float(gray.std()) >= min_nonzero_std


def synthetic_darken(
    image: np.ndarray,
    gamma_range: tuple[float, float] = (2.0, 5.0),
    poisson_scale: float = 0.02,
    gaussian_sigma: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthetically darken a sunlit patch and inject mixed Poisson-Gaussian
    noise, for Phase-1 supervised pretraining where a "bright" reference
    exists (the original sunlit patch) but real PSR patches have none.

    Args:
        image: (H, W, C) float image in [0, 1].
        gamma_range: exponent applied to darken (`image ** gamma`); higher
            gamma = darker.
        poisson_scale: relative scale of shot noise injected after darkening.
        gaussian_sigma: std-dev of additive read noise.
        rng: optional numpy Generator for reproducibility.
    """
    rng = rng or np.random.default_rng()
    gamma = rng.uniform(*gamma_range)
    darkened = np.clip(image, 0.0, 1.0) ** gamma

    # shot noise scales with signal, read noise is signal-independent
    shot = rng.poisson(darkened / poisson_scale) * poisson_scale - darkened
    read = rng.normal(0.0, gaussian_sigma, size=image.shape)

    noisy = np.clip(darkened + shot + read, 0.0, 1.0)
    return noisy.astype(np.float32)
