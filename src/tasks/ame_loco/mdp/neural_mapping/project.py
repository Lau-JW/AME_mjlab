"""Project depth point clouds into local elevation grids (AME-2 Sec V-A)."""

from __future__ import annotations

import torch


def project_points_to_height_grid(
    points_b: torch.Tensor,
    valid: torch.Tensor,
    grid_h: int,
    grid_w: int,
    resolution: float,
    center_x: float,
    center_y: float = 0.0,
    empty_value: float = -1.0,
) -> torch.Tensor:
    """Max-z projection of base-frame points into a local height grid.

    Args:
        points_b: (B, N, 3) points in robot base/yaw frame
        valid: (B, N) bool mask of valid hits
        grid_h, grid_w: grid shape
        resolution: meters / cell
        center_x, center_y: grid center in base frame
        empty_value: fill for empty cells (paper: fixed minimum)

    Returns:
        height: (B, 1, H, W)
    """
    B, N, _ = points_b.shape
    device = points_b.device
    half_h = 0.5 * (grid_h - 1) * resolution
    half_w = 0.5 * (grid_w - 1) * resolution
    x0 = center_x - half_h
    y0 = center_y - half_w

    x = points_b[..., 0]
    y = points_b[..., 1]
    z = points_b[..., 2]
    ix = torch.floor((x - x0) / resolution).long()
    iy = torch.floor((y - y0) / resolution).long()
    in_b = (
        valid
        & (ix >= 0) & (ix < grid_h)
        & (iy >= 0) & (iy < grid_w)
    )

    # Flatten batch for scatter
    flat_idx = (
        torch.arange(B, device=device).unsqueeze(1) * (grid_h * grid_w)
        + ix * grid_w
        + iy
    )
    flat_idx = flat_idx.clamp(min=0)
    z_flat = z.clone()
    z_flat = torch.where(in_b, z_flat, torch.full_like(z_flat, -1e9))

    out = torch.full((B * grid_h * grid_w,), -1e9, device=device, dtype=z.dtype)
    out.scatter_reduce_(0, flat_idx.reshape(-1), z_flat.reshape(-1), reduce="amax", include_self=True)
    height = out.view(B, grid_h, grid_w)
    height = torch.where(height < -1e8, torch.full_like(height, empty_value), height)
    return height.unsqueeze(1)


def make_xy_grid(
    grid_h: int,
    grid_w: int,
    resolution: float,
    center_x: float,
    center_y: float = 0.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return meshgrid x,y of shape (H, W) in base frame."""
    xs = center_x + (torch.arange(grid_h, device=device, dtype=torch.float32) - 0.5 * (grid_h - 1)) * resolution
    ys = center_y + (torch.arange(grid_w, device=device, dtype=torch.float32) - 0.5 * (grid_w - 1)) * resolution
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return grid_x, grid_y
