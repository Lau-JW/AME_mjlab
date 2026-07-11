"""Neural mapping pipeline for AME-2 student (Section V).

Converts depth clouds into uncertainty-aware elevation maps via a lightweight
U-Net, then fuses local predictions into a global map using odometry poses.

Paper:
- Local grids: TRON1 31x31 @ 4cm, centered at (0.6, 0.0) in base frame.
- Ego map for controller: TRON1 18x13 @ 8cm, centered at (0.32, 0.0).
- Loss: beta-NLL (beta=0.5) + per-sample TV reweighting.
- Fusion: Probabilistic Winner-Take-All with variance lower/upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Local grid geometry helpers
# ---------------------------------------------------------------------------


def project_depth_to_local_grid(
    points: torch.Tensor,
    grid_size: tuple[int, int],
    resolution: float,
    center: tuple[float, float] = (0.0, 0.0),
    max_height: float = 2.0,
    min_height: float = -2.0,
) -> torch.Tensor:
    """Project base-frame depth points into a 2D local height grid.

    Args:
        points: (B, N, 3) base-frame point cloud (x forward, y left, z up).
        grid_size: (H, W) local grid resolution in cells.
        resolution: cell size in meters.
        center: (x, y) offset of grid center in base frame.
        max_height: clamp maximum z value.
        min_height: value for empty cells.

    Returns:
        local_grid: (B, 1, H, W) height map with max z per cell, empty cells = min_height.
    """
    B, N, _ = points.shape
    H, W = grid_size
    device = points.device
    cx, cy = center

    # Grid bounds: x in [cx - (W-1)*res/2, cx + (W-1)*res/2], y same with H
    half_x = (W - 1) * resolution / 2.0
    half_y = (H - 1) * resolution / 2.0

    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]

    # Normalize to grid indices [0, W-1] and [0, H-1]
    ix = ((x - (cx - half_x)) / resolution).long()
    iy = ((y - (cy - half_y)) / resolution).long()

    in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (z > -10.0)

    grid = torch.full((B, 1, H, W), min_height, device=device, dtype=torch.float32)

    for b in range(B):
        valid = in_bounds[b]
        if not valid.any():
            continue
        ix_b = ix[b][valid].clamp(0, W - 1)
        iy_b = iy[b][valid].clamp(0, H - 1)
        z_b = z[b][valid]
        flat_idx = (iy_b * W + ix_b).long()
        grid_flat = grid[b, 0].view(-1)
        grid_flat.scatter_reduce_(0, flat_idx, z_b, reduce="amax", include_self=False)

    grid.clamp_(min_height, max_height)
    return grid


# ---------------------------------------------------------------------------
# Local elevation/uncertainty predictor (U-Net with gated residual)
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv + ELU + Conv + ELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LocalMapPredictor(nn.Module):
    """Lightweight U-Net predicting local elevation and log-variance.

    Paper Fig. 8: 2x CNN encoder -> maxpool -> upsample -> concat -> 2x CNN,
    then three 1x1 heads for uncertainty, raw estimation, and gating map.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.enc2 = ConvBlock(c, c)

        # Decoder
        self.up = nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2, c)
        self.dec2 = ConvBlock(c, c)

        # Output heads
        self.uncertainty_head = nn.Conv2d(c, 1, kernel_size=1)
        self.raw_head = nn.Conv2d(c, 1, kernel_size=1)
        self.gate_head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args: x (B, 1, H, W). Returns dict with keys 'elevation', 'log_var'."""
        e1 = self.enc1(x)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        d2 = self.up(e2)
        # Pad d2 to match e1 spatial dims if necessary
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        d0 = self.dec2(d1)

        log_var = self.uncertainty_head(d0)
        raw = self.raw_head(d0)
        gate = torch.sigmoid(self.gate_head(d0))

        # Gated residual: preserve input where input is valid, use raw elsewhere
        out = gate * raw + (1.0 - gate) * x

        return {"elevation": out, "log_var": log_var}


# ---------------------------------------------------------------------------
# Global map fusion and query
# ---------------------------------------------------------------------------


def rotation_matrix_2d(yaw: torch.Tensor) -> torch.Tensor:
    """Build 2D rotation matrices (B, 2, 2) from yaw angles (B,)."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    R = torch.zeros(yaw.shape[0], 2, 2, device=yaw.device, dtype=yaw.dtype)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R


@dataclass
class NeuralMapCfg:
    """Configuration for the neural global map."""

    local_grid_size: tuple[int, int] = (31, 31)  # (H, W)
    local_resolution: float = 0.04
    local_center: tuple[float, float] = (0.6, 0.0)
    ego_grid_size: tuple[int, int] = (18, 13)  # (H, W) for controller
    ego_resolution: float = 0.08
    ego_center: tuple[float, float] = (0.32, 0.0)
    global_size: float = 8.0  # meters per side
    global_resolution: float = 0.04
    device: str = "cuda:0"
    max_variance: float = 1.0


class NeuralMapManager(nn.Module):
    """Maintains a global elevation+uncertainty map and queries ego maps.

    For use in simulation: all maps are batched (B, ...).
    On real hardware this class would run with B=1.
    """

    def __init__(self, num_envs: int, cfg: NeuralMapCfg):
        super().__init__()
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = cfg.device

        self.global_size_m = cfg.global_size
        self.global_res = cfg.global_resolution
        self.global_cells = int(round(cfg.global_size / cfg.global_resolution))

        # Global map state: (B, 2, H, W) channel 0 = elevation, 1 = variance
        self.register_buffer(
            "global_map",
            torch.zeros(num_envs, 2, self.global_cells, self.global_cells, device=self.device),
        )
        self.register_buffer(
            "origin_xy",
            torch.zeros(num_envs, 2, device=self.device),
        )
        self.reset()

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset global map to flat ground with large variance."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Flat ground elevation at standing height relative to base
        self.global_map[env_ids, 0] = 0.0
        self.global_map[env_ids, 1] = self.cfg.max_variance

    @torch.no_grad()
    def update(self, local_elev: torch.Tensor, local_var: torch.Tensor, base_pos: torch.Tensor, base_yaw: torch.Tensor):
        """Fuse local prediction into global map using Probabilistic Winner-Take-All.

        Args:
            local_elev: (B, 1, H, W) base-frame local elevation.
            local_var: (B, 1, H, W) base-frame local variance.
            base_pos: (B, 3) world position.
            base_yaw: (B,) world yaw.
        """
        B = local_elev.shape[0]
        H, W = self.cfg.local_grid_size
        res = self.cfg.local_resolution
        cx, cy = self.cfg.local_center

        half_x = (W - 1) * res / 2.0
        half_y = (H - 1) * res / 2.0

        xs = torch.linspace(cx - half_x, cx + half_x, W, device=self.device)
        ys = torch.linspace(cy - half_y, cy + half_y, H, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        local_xy = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

        # Rotate local points to world frame and add base position
        R = rotation_matrix_2d(base_yaw)  # (B, 2, 2)
        local_xy_flat = local_xy.view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # (B, H*W, 2)
        world_xy = torch.bmm(local_xy_flat, R.transpose(1, 2)) + base_pos[:, :2].unsqueeze(1)

        # Convert world xy to global grid indices
        world_xy_rel = world_xy - self.origin_xy.unsqueeze(1)  # (B, N, 2)
        gi = (world_xy_rel[:, :, 0] / self.global_res).long()
        gj = (world_xy_rel[:, :, 1] / self.global_res).long()

        elev_flat = local_elev.view(B, -1)
        var_flat = local_var.view(B, -1).clamp_min(1e-4)

        half_cells = self.global_cells // 2
        in_bounds = (gi >= -half_cells) & (gi < half_cells) & (gj >= -half_cells) & (gj < half_cells)

        # Recenter indices to [0, global_cells)
        gi = gi + half_cells
        gj = gj + half_cells

        for b in range(B):
            valid = in_bounds[b]
            if not valid.any():
                continue
            gi_b = gi[b][valid]
            gj_b = gj[b][valid]
            elev_b = elev_flat[b][valid]
            var_b = var_flat[b][valid]

            # Prior
            prior_var = self.global_map[b, 1, gj_b, gi_b].clamp_min(1e-4)
            prior_elev = self.global_map[b, 0, gj_b, gi_b]

            # Effective measurement variance lower-bounded by prior
            eff_var = torch.maximum(var_b, 0.5 * prior_var)

            # Valid update: not too much larger than prior, or low absolute uncertainty
            valid_update = (eff_var < 1.5 * prior_var) | (eff_var < 0.22)
            if not valid_update.any():
                continue

            gi_b = gi_b[valid_update]
            gj_b = gj_b[valid_update]
            elev_b = elev_b[valid_update]
            eff_var = eff_var[valid_update]
            prior_var = prior_var[valid_update]

            # Win probability
            p_win = (1.0 / eff_var) / ((1.0 / eff_var) + (1.0 / prior_var))
            rand = torch.rand_like(p_win)
            take_new = rand < p_win

            self.global_map[b, 0, gj_b, gi_b] = torch.where(
                take_new, elev_b, self.global_map[b, 0, gj_b, gi_b]
            )
            self.global_map[b, 1, gj_b, gi_b] = torch.where(
                take_new, eff_var, self.global_map[b, 1, gj_b, gi_b]
            )

    @torch.no_grad()
    def query(self, base_pos: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
        """Query an ego-centric elevation map from the global map.

        Args:
            base_pos: (B, 3) world position.
            base_yaw: (B,) world yaw.

        Returns:
            ego_map: (B, 4, H, W) with channels [x, y, z, u].
        """
        B = base_pos.shape[0]
        H, W = self.cfg.ego_grid_size
        res = self.cfg.ego_resolution
        cx, cy = self.cfg.ego_center

        half_x = (W - 1) * res / 2.0
        half_y = (H - 1) * res / 2.0

        xs = torch.linspace(cx - half_x, cx + half_x, W, device=self.device)
        ys = torch.linspace(cy - half_y, cy + half_y, H, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        local_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W, 2)

        # Rotate to world and add base pos
        R = rotation_matrix_2d(base_yaw)
        local_xy_b = local_xy.unsqueeze(0).expand(B, -1, -1)
        world_xy = torch.bmm(local_xy_b, R.transpose(1, 2)) + base_pos[:, :2].unsqueeze(1)

        world_xy_rel = world_xy - self.origin_xy.unsqueeze(1)
        gi = world_xy_rel[:, :, 0] / self.global_res
        gj = world_xy_rel[:, :, 1] / self.global_res
        half_cells = self.global_cells // 2
        gi_norm = (gi + half_cells) / self.global_cells * 2.0 - 1.0
        gj_norm = (gj + half_cells) / self.global_cells * 2.0 - 1.0
        grid = torch.stack([gi_norm, gj_norm], dim=-1).view(B, H, W, 2)  # (B, H, W, 2)

        # Bilinear sample global map
        global_map_expanded = self.global_map.unsqueeze(1)  # (B, 1, 2, G, G)
        sampled = F.grid_sample(
            global_map_expanded,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # (B, 2, H, W)

        elev = sampled[:, 0:1]
        var = sampled[:, 1:2].clamp_min(1e-4)

        # Build x,y channels (base-frame grid coordinates)
        x_grid = grid_x.view(1, 1, H, W).expand(B, -1, -1, -1)
        y_grid = grid_y.view(1, 1, H, W).expand(B, -1, -1, -1)

        return torch.cat([x_grid, y_grid, elev, var], dim=1)

    def recompute_origin(self, base_pos: torch.Tensor):
        """Recenter global map around robot when approaching boundary."""
        # Simple recenter: if robot gets within 1m of edge, shift origin and map
        half_m = self.global_size_m / 2.0 - 1.0
        rel = base_pos[:, :2] - self.origin_xy
        need_recenter = (rel.abs() > half_m).any(dim=1)
        if not need_recenter.any():
            return
        # Re-centering implementation: shift origin and resample global map
        # (omitted for simplicity; would require bilinear resampling of the global map)
        # For now we just move the origin and reset the map for those envs
        self.origin_xy[need_recenter] = base_pos[need_recenter, :2].detach()
        self.reset(torch.arange(self.num_envs, device=self.device)[need_recenter])


# ---------------------------------------------------------------------------
# Wrapper used inside the environment / observation function
# ---------------------------------------------------------------------------


class StudentMappingState:
    """Holds per-environment neural mapping state and predictor.

    This is attached to the environment so the observation function can
    produce 4-channel ego maps without changing the environment class.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        cfg: NeuralMapCfg | None = None,
        checkpoint_path: str | None = None,
    ):
        self.cfg = cfg or NeuralMapCfg(device=device)
        self.device = device
        self.predictor = LocalMapPredictor().to(device)
        self.map_manager = NeuralMapManager(num_envs, self.cfg).to(device)
        self.num_envs = num_envs
        if checkpoint_path is not None:
            print(f"[AME] Loading neural mapping predictor from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            self.predictor.load_state_dict(ckpt["model"])

    def reset(self, env_ids: torch.Tensor | None = None):
        self.map_manager.reset(env_ids)

    @torch.no_grad()
    def update_from_depth(
        self,
        depth_points: torch.Tensor,
        base_pos: torch.Tensor,
        base_yaw: torch.Tensor,
    ) -> torch.Tensor:
        """Run one mapping step: depth -> local grid -> UNet -> fuse -> query ego map.

        Args:
            depth_points: (B, N, 3) base-frame point cloud.
            base_pos: (B, 3) world position.
            base_yaw: (B,) world yaw.

        Returns:
            ego_map: (B, 4, H, W) for student observation.
        """
        B = depth_points.shape[0]
        local_grid = project_depth_to_local_grid(
            depth_points,
            grid_size=self.cfg.local_grid_size,
            resolution=self.cfg.local_resolution,
            center=self.cfg.local_center,
        )
        pred = self.predictor(local_grid)
        elev = pred["elevation"]
        var = torch.exp(pred["log_var"]).clamp_min(1e-4)
        self.map_manager.update(elev, var, base_pos, base_yaw)
        self.map_manager.recompute_origin(base_pos)
        return self.map_manager.query(base_pos, base_yaw)

    def state_dict(self) -> dict:
        return {
            "predictor": self.predictor.state_dict(),
            "map_manager": self.map_manager.global_map.detach().cpu(),
            "origin_xy": self.map_manager.origin_xy.detach().cpu(),
        }

    def load_state_dict(self, state: dict):
        if "predictor" in state:
            self.predictor.load_state_dict(state["predictor"])
        if "map_manager" in state:
            self.map_manager.global_map.copy_(state["map_manager"].to(self.device))
        if "origin_xy" in state:
            self.map_manager.origin_xy.copy_(state["origin_xy"].to(self.device))


# ---------------------------------------------------------------------------
# Training loss for the local map predictor
# ---------------------------------------------------------------------------


def beta_nll_loss(
    pred_elev: torch.Tensor,
    log_var: torch.Tensor,
    target_elev: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Beta-NLL loss (Eq. 9). Standard-deviation weight is stop-gradient."""
    sigma = torch.exp(0.5 * log_var).clamp_min(1e-4)
    loss = sigma.detach() * (0.5 * log_var + 0.5 * (target_elev - pred_elev) ** 2 / sigma ** 2)
    return loss.mean()


def tv_reweight_loss(
    pred_elev: torch.Tensor,
    log_var: torch.Tensor,
    target_elev: torch.Tensor,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Beta-NLL with per-sample total-variation reweighting (Eq. 10)."""
    B = pred_elev.shape[0]
    sigma = torch.exp(0.5 * log_var).clamp_min(1e-4)
    per_pixel_loss = sigma.detach() * (0.5 * log_var + 0.5 * (target_elev - pred_elev) ** 2 / sigma ** 2)

    # Total variation of ground truth
    grad_x = torch.abs(target_elev[:, :, :, 1:] - target_elev[:, :, :, :-1]).sum(dim=(1, 2, 3))
    grad_y = torch.abs(target_elev[:, :, 1:, :] - target_elev[:, :, :-1, :]).sum(dim=(1, 2, 3))
    H, W = target_elev.shape[-2:]
    tv = (grad_x + grad_y) / (H * W)
    weights = tv / (tv.sum() + eps) * B

    # Average per sample, then reweight
    per_sample = per_pixel_loss.view(B, -1).mean(dim=1)
    return (weights * per_sample).mean()
