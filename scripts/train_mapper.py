"""Train AME-2 elevation U-Net mapper (paper Sec V-B).

Synthesizes local height grids from the locomotion terrain raycaster, applies
paper augmentations, and optimizes β-NLL (Eq. 9) with TV sample weights (Eq. 10).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tasks.ame_loco.mdp.neural_mapping.gated_unet import (
    GatedElevationUNet,
    beta_nll_loss,
    total_variation_weights,
)
from src.tasks.ame_loco.mdp.neural_mapping.pipeline import LOCAL_H, LOCAL_W


def _augment(gt: torch.Tensor) -> torch.Tensor:
    """Paper V-B2 augmentations on (B,1,H,W) GT → noisy/partial input."""
    x = gt.clone()
    B, _, H, W = x.shape
    # additive uniform noise with random magnitude
    mag = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.0, 0.15)
    x = x + mag * torch.empty_like(x).uniform_(-1.0, 1.0)
    # random border crop → fill empty
    for b in range(B):
        top = int(torch.randint(0, max(1, H // 5), (1,)).item())
        bottom = int(torch.randint(0, max(1, H // 5), (1,)).item())
        left = int(torch.randint(0, max(1, W // 5), (1,)).item())
        right = int(torch.randint(0, max(1, W // 5), (1,)).item())
        if top:
            x[b, :, :top, :] = -1.0
        if bottom:
            x[b, :, H - bottom :, :] = -1.0
        if left:
            x[b, :, :, :left] = -1.0
        if right:
            x[b, :, :, W - right :] = -1.0
    # random missing
    miss = torch.rand_like(x) < 0.15
    x = torch.where(miss, torch.full_like(x, -1.0), x)
    # outliers
    out = torch.rand_like(x) < 0.02
    x = torch.where(out, x + torch.empty_like(x).uniform_(-1.0, 1.0), x)
    return x


def _synthetic_batch(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Procedural terrains when full sim sampling is unavailable."""
    H, W = LOCAL_H, LOCAL_W
    ys = torch.linspace(-1, 1, H, device=device).view(H, 1)
    xs = torch.linspace(-1, 1, W, device=device).view(1, W)
    gt = torch.zeros(batch, 1, H, W, device=device)
    for b in range(batch):
        mode = int(torch.randint(0, 4, (1,)).item())
        if mode == 0:
            # flat + noise
            gt[b] = 0.05 * torch.randn(1, H, W, device=device)
        elif mode == 1:
            # stairs along x
            step = (ys + 1) * 3
            gt[b, 0] = torch.floor(step) * 0.12
        elif mode == 2:
            # boxes
            gt[b] = 0.0
            for _ in range(5):
                i0 = int(torch.randint(0, H - 5, (1,)).item())
                j0 = int(torch.randint(0, W - 5, (1,)).item())
                h = float(torch.empty(1).uniform_(0.1, 0.6))
                gt[b, 0, i0 : i0 + 5, j0 : j0 + 5] = h
        else:
            # smooth heightfield
            gt[b, 0] = 0.25 * torch.sin(3 * ys) * torch.cos(3 * xs)
    return _augment(gt), gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AME-2 gated elevation U-Net.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("logs/mappers/g1_elevation_unet.pt"))
    args = parser.parse_args()

    device = args.device
    model = GatedElevationUNet(base_channels=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = _synthetic_batch(args.batch_size, device)
        mu, log_var, _ = model(x)
        w = total_variation_weights(y)
        loss = beta_nll_loss(mu, log_var, y, beta=0.5, sample_weights=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                mae = (mu - y).abs().mean().item()
                mean_std = torch.exp(0.5 * log_var).mean().item()
            print(f"[mapper] step={step}/{args.steps} loss={loss.item():.4f} mae={mae:.4f} std={mean_std:.3f}")

    torch.save({"model": model.state_dict(), "local_hw": (LOCAL_H, LOCAL_W)}, args.out)
    print(f"[mapper] saved → {args.out}")


if __name__ == "__main__":
    main()
