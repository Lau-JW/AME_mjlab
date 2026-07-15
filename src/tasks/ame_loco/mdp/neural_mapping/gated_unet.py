"""AME-2 gated U-Net elevation mapper (paper Fig 8, Sec V-B3).

Input: 1-channel local height grid from projected depth (holes = sentinel).
Outputs: raw elevation, log-variance, gate.
Final elevation: gate * raw + (1 - gate) * input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedElevationUNet(nn.Module):
    """Shallow U-Net with gated residual (AME-2 Fig 8)."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.enc1 = _ConvBlock(1, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _ConvBlock(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = _ConvBlock(c * 2, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, 3, 1)  # raw, log_var, gate_logit

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) local height input
        Returns:
            mu: (B, 1, H, W) gated elevation
            log_var: (B, 1, H, W)
            gate: (B, 1, H, W) in (0, 1)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        raw, log_var, gate_logit = self.head(d1).split(1, dim=1)
        gate = torch.sigmoid(gate_logit)
        # Clamp log-variance for numerical stability.
        log_var = torch.clamp(log_var, min=-8.0, max=4.0)
        mu = gate * raw + (1.0 - gate) * x
        return mu, log_var, gate


def beta_nll_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """β-NLL / L0.5 loss (AME-2 Eq. 9), beta=0.5."""
    var = torch.exp(log_var).clamp_min(1e-6)
    sigma = torch.sqrt(var)
    nll = 0.5 * log_var + 0.5 * (target - mu).pow(2) / var
    # stop-grad on sigma^beta
    weight = sigma.detach().pow(beta)
    per = weight * nll
    if sample_weights is not None:
        # sample_weights: (B,) normalized
        per = per.flatten(1).mean(dim=1) * sample_weights
        return per.mean()
    return per.mean()


def total_variation_weights(target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample TV weights (AME-2 Eq. 10). target: (B, 1, H, W)."""
    dx = (target[:, :, :, 1:] - target[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
    dy = (target[:, :, 1:, :] - target[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
    tv = dx + dy
    w = tv / (tv.sum() + eps)
    return w
