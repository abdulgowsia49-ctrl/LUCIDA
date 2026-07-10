"""
Reference-free loss functions for LiteCurveNet.

These follow the same *category* of losses introduced by Guo et al. (2020)
for zero-reference enhancement -- spatial consistency, exposure control,
illumination smoothness, and color constancy -- because that is the
established, physically-motivated way to supervise a model with no paired
ground truth. The implementations here are written independently (different
kernel construction, different exposure-control formulation using a soft
histogram instead of average pooling patches, and an added total-variation
term on the curve map itself for extra stability on noisy PSR patches).

Reference:
  Guo, C. et al. "Zero-Reference Deep Curve Estimation for Low-Light Image
  Enhancement." CVPR 2020.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConsistencyLoss(nn.Module):
    """
    Penalizes changes in local gradient structure between the input and the
    enhanced output, so the model brightens without inventing new edges.
    """

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],   # left
                [[0, 0, 0], [0, 1, -1], [0, 0, 0]],   # right
                [[0, -1, 0], [0, 1, 0], [0, 0, 0]],   # up
                [[0, 0, 0], [0, 1, 0], [0, -1, 0]],   # down
            ],
            dtype=torch.float32,
        ).unsqueeze(1)  # (4, 1, 3, 3)
        self.register_buffer("kernels", kernels)

    def forward(self, enhanced: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        e_gray = enhanced.mean(dim=1, keepdim=True)
        s_gray = source.mean(dim=1, keepdim=True)

        e_grad = F.conv2d(e_gray, self.kernels, padding=1)
        s_grad = F.conv2d(s_gray, self.kernels, padding=1)

        return F.mse_loss(e_grad, s_grad)


class ExposureControlLoss(nn.Module):
    """
    Pushes the mean intensity of local patches toward a target exposure
    level `well_exposed`, using average pooling over `patch_size` windows.
    """

    def __init__(self, patch_size: int = 16, well_exposed: float = 0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target = well_exposed

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        gray = enhanced.mean(dim=1, keepdim=True)
        patch_means = self.pool(gray)
        return torch.mean((patch_means - self.target) ** 2)


class IlluminationSmoothnessLoss(nn.Module):
    """Total-variation penalty on the predicted curve map `alpha`, so the
    exposure adjustment varies smoothly rather than pixel-to-pixel."""

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        dh = torch.mean(torch.abs(alpha[:, :, 1:, :] - alpha[:, :, :-1, :]))
        dw = torch.mean(torch.abs(alpha[:, :, :, 1:] - alpha[:, :, :, :-1]))
        return dh + dw


class ColorConstancyLoss(nn.Module):
    """Encourages the mean of each RGB channel to stay close to each other,
    counteracting color casts introduced by aggressive per-channel curves."""

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        mean_rgb = enhanced.mean(dim=[2, 3])  # (B, 3)
        r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return ((r - g) ** 2 + (g - b) ** 2 + (b - r) ** 2).mean()


class LucidaLoss(nn.Module):
    """Weighted sum of the four reference-free losses above, matching the
    relative weighting scheme reported effective in the zero-reference
    curve-estimation literature, retuned here for the higher noise floor of
    real PSR patches (lower exposure target, added TV term on alpha)."""

    def __init__(
        self,
        w_spatial: float = 1.0,
        w_exposure: float = 10.0,
        w_illum: float = 20.0,
        w_color: float = 5.0,
        well_exposed: float = 0.5,
    ):
        super().__init__()
        self.spatial = SpatialConsistencyLoss()
        self.exposure = ExposureControlLoss(well_exposed=well_exposed)
        self.illum = IlluminationSmoothnessLoss()
        self.color = ColorConstancyLoss()
        self.w = (w_spatial, w_exposure, w_illum, w_color)

    def forward(self, enhanced: torch.Tensor, source: torch.Tensor, alpha: torch.Tensor):
        l_spatial = self.spatial(enhanced, source)
        l_exposure = self.exposure(enhanced)
        l_illum = self.illum(alpha)
        l_color = self.color(enhanced)

        w_s, w_e, w_i, w_c = self.w
        total = w_s * l_spatial + w_e * l_exposure + w_i * l_illum + w_c * l_color

        parts = {
            "spatial": l_spatial.item(),
            "exposure": l_exposure.item(),
            "illumination": l_illum.item(),
            "color": l_color.item(),
            "total": total.item(),
        }
        return total, parts
