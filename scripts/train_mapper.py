"""Train AME-2 elevation U-Net mapper (paper Sec V-B).

Synthesizes local height grids (stairs/boxes/slopes/holes), applies paper-style
augmentations, and optimizes β-NLL (Eq. 9) with TV sample weights (Eq. 10).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

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
    mag = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.0, 0.20)
    x = x + mag * torch.empty_like(x).uniform_(-1.0, 1.0)
    for b in range(B):
        top = int(torch.randint(0, max(1, H // 4), (1,)).item())
        bottom = int(torch.randint(0, max(1, H // 4), (1,)).item())
        left = int(torch.randint(0, max(1, W // 4), (1,)).item())
        right = int(torch.randint(0, max(1, W // 4), (1,)).item())
        if top:
            x[b, :, :top, :] = -1.0
        if bottom:
            x[b, :, H - bottom :, :] = -1.0
        if left:
            x[b, :, :, :left] = -1.0
        if right:
            x[b, :, :, W - right :] = -1.0
    miss = torch.rand_like(x) < 0.20
    x = torch.where(miss, torch.full_like(x, -1.0), x)
    out = torch.rand_like(x) < 0.03
    x = torch.where(out, x + torch.empty_like(x).uniform_(-1.0, 1.0), x)
    return x


def _synthetic_batch(batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Richer procedural terrains (stairs 0.1–0.25 m, pits, slopes, boxes)."""
    H, W = LOCAL_H, LOCAL_W
    ys = torch.linspace(-1, 1, H, device=device).view(H, 1)
    xs = torch.linspace(-1, 1, W, device=device).view(1, W)
    gt = torch.zeros(batch, 1, H, W, device=device)
    for b in range(batch):
        mode = int(torch.randint(0, 6, (1,)).item())
        if mode == 0:
            gt[b] = 0.05 * torch.randn(1, H, W, device=device)
        elif mode == 1:
            # stairs along x (matches G1 rough curriculum step heights)
            step_h = float(torch.empty(1).uniform_(0.10, 0.25))
            n_steps = float(torch.empty(1).uniform_(2.0, 5.0))
            step = (ys + 1) * n_steps
            gt[b, 0] = torch.floor(step) * step_h
        elif mode == 2:
            gt[b] = 0.0
            for _ in range(int(torch.randint(3, 8, (1,)).item())):
                i0 = int(torch.randint(0, H - 6, (1,)).item())
                j0 = int(torch.randint(0, W - 6, (1,)).item())
                hi = int(torch.randint(3, 8, (1,)).item())
                wj = int(torch.randint(3, 8, (1,)).item())
                h = float(torch.empty(1).uniform_(0.08, 0.55))
                gt[b, 0, i0 : i0 + hi, j0 : j0 + wj] = h
        elif mode == 3:
            gt[b, 0] = 0.25 * torch.sin(3 * ys) * torch.cos(3 * xs)
            gt[b, 0] += 0.15 * torch.sin(7 * xs)
        elif mode == 4:
            # ramp / slope
            sx = float(torch.empty(1).uniform_(-0.35, 0.35))
            sy = float(torch.empty(1).uniform_(-0.25, 0.25))
            gt[b, 0] = sx * ys + sy * xs
        else:
            # pit / hole
            gt[b] = 0.05 * torch.randn(1, H, W, device=device)
            i0 = int(torch.randint(4, H - 10, (1,)).item())
            j0 = int(torch.randint(4, W - 10, (1,)).item())
            depth = float(torch.empty(1).uniform_(0.15, 0.50))
            gt[b, 0, i0 : i0 + 6, j0 : j0 + 6] = -depth
    return _augment(gt), gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AME-2 gated elevation U-Net.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=Path, default=Path("logs/mappers/g1_elevation_unet.pt"))
    args = parser.parse_args()

    device = args.device
    model = GatedElevationUNet(base_channels=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-5)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        x, y = _synthetic_batch(args.batch_size, device)
        mu, log_var, _ = model(x)
        w = total_variation_weights(y)
        loss = beta_nll_loss(mu, log_var, y, beta=0.5, sample_weights=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                mae = (mu - y).abs().mean().item()
                mean_std = torch.exp(0.5 * log_var).mean().item()
            print(
                f"[mapper] step={step}/{args.steps} loss={loss.item():.4f} "
                f"mae={mae:.4f} std={mean_std:.3f} lr={sched.get_last_lr()[0]:.2e}"
            )
            if mae < best_mae:
                best_mae = mae
                torch.save(
                    {
                        "model": model.state_dict(),
                        "local_hw": (LOCAL_H, LOCAL_W),
                        "mae": best_mae,
                        "step": step,
                    },
                    args.out,
                )
                print(f"[mapper] best mae={best_mae:.4f} → {args.out}")

    # final save (may overwrite best if last is worse — keep best already on disk)
    torch.save(
        {
            "model": model.state_dict(),
            "local_hw": (LOCAL_H, LOCAL_W),
            "mae": best_mae,
            "step": args.steps,
        },
        args.out.with_name(args.out.stem + "_last.pt"),
    )
    print(f"[mapper] done best_mae={best_mae:.4f} best→{args.out}")


if __name__ == "__main__":
    main()
