"""
Tiled inference over full-size OHRC frames.

Full OHRC strips are far too large to run through the network in one pass,
so this splits the frame into overlapping tiles, runs LiteCurveNet on each,
blends overlaps with a cosine-tapered window (to avoid visible tile seams),
then hands the stitched result to the classical denoiser/contrast chain.
"""
from __future__ import annotations

import numpy as np
import torch

from src.model.curve_net import LiteCurveNet
from src.preprocess.denoise import postprocess


def _cosine_window(size: int) -> np.ndarray:
    """1D raised-cosine taper, used to blend overlapping tile edges."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / max(size - 1, 1))
    return w


def _tile_weight(tile_size: int, overlap: int) -> np.ndarray:
    """2D blend weight for a tile: 1 in the interior, cosine-tapered over
    the overlap region at each edge."""
    ramp = _cosine_window(overlap * 2)
    w1d = np.ones(tile_size, dtype=np.float32)
    w1d[:overlap] = ramp[:overlap]
    w1d[-overlap:] = ramp[overlap:]
    return np.outer(w1d, w1d)


class TiledInferencer:
    def __init__(
        self,
        model: LiteCurveNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        tile_size: int = 256,
        overlap: int = 32,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self._weight = _tile_weight(tile_size, overlap)

    @torch.no_grad()
    def run(self, image: np.ndarray, apply_postprocess: bool = True) -> np.ndarray:
        """
        Args:
            image: (H, W, 3) float array in [0, 1].
            apply_postprocess: run NLM + CLAHE after stitching.

        Returns:
            (H, W, 3) float array in [0, 1].
        """
        h, w, c = image.shape
        stride = self.tile_size - self.overlap

        out_accum = np.zeros((h, w, c), dtype=np.float32)
        weight_accum = np.zeros((h, w, 1), dtype=np.float32)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y1, x1 = min(y + self.tile_size, h), min(x + self.tile_size, w)
                y0, x0 = max(0, y1 - self.tile_size), max(0, x1 - self.tile_size)

                tile = image[y0:y1, x0:x1]
                th, tw = tile.shape[:2]

                tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
                enhanced, _ = self.model(tensor)
                enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()

                weight = self._weight[:th, :tw][..., None]
                out_accum[y0:y1, x0:x1] += enhanced_np * weight
                weight_accum[y0:y1, x0:x1] += weight

        stitched = out_accum / np.clip(weight_accum, 1e-6, None)

        return postprocess(stitched) if apply_postprocess else stitched
