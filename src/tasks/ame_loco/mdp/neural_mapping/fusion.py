"""Global elevation map with Probabilistic Winner-Take-All fusion (AME-2 Eqs 6-8)."""

from __future__ import annotations

import math

import torch

from .project import make_xy_grid


class GlobalElevationMap:
    """Per-env egocentric-recentering global map (elevation + variance)."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        map_size_m: float = 8.0,
        resolution: float = 0.04,
        init_height: float = 0.0,
        init_std: float = 2.0,
    ):
        self.num_envs = num_envs
        self.device = device
        self.resolution = resolution
        self.n = int(round(map_size_m / resolution))
        self.origin_xy = torch.zeros(num_envs, 2, device=device)  # world XY of map center
        self.height = torch.full(
            (num_envs, self.n, self.n), init_height, device=device, dtype=torch.float32
        )
        self.var = torch.full(
            (num_envs, self.n, self.n), init_std ** 2, device=device, dtype=torch.float32
        )
        self.init_height = init_height
        self.init_var = init_std ** 2

    def reset(self, env_ids: torch.Tensor, base_xy: torch.Tensor, ground_z_b: float | None = None) -> None:
        if len(env_ids) == 0:
            return
        # base_xy may be full (N,2) or already gathered (len(env_ids),2)
        if base_xy.shape[0] == self.num_envs:
            self.origin_xy[env_ids] = base_xy[env_ids]
        else:
            self.origin_xy[env_ids] = base_xy
        h0 = self.init_height if ground_z_b is None else ground_z_b
        self.height[env_ids] = h0
        self.var[env_ids] = self.init_var

    def maybe_recenter(self, base_xy: torch.Tensor, margin_cells: int = 20) -> None:
        """Recenters map when robot nears boundary (paper Sec V-C)."""
        rel = (base_xy - self.origin_xy) / self.resolution
        half = 0.5 * self.n
        need = (rel.abs() > (half - margin_cells)).any(dim=-1)
        ids = need.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) == 0:
            return
        # Simple reset recenter (keeps only overlapping region approximately by full reset).
        self.origin_xy[ids] = base_xy[ids]
        self.height[ids] = self.init_height
        self.var[ids] = self.init_var

    def fuse_local(
        self,
        base_xy: torch.Tensor,
        base_yaw: torch.Tensor,
        local_mu: torch.Tensor,
        local_var: torch.Tensor,
        local_res: float,
        local_h: int,
        local_w: int,
        local_cx: float,
        local_cy: float = 0.0,
    ) -> None:
        """Fuse local (B,1,h,w) predictions into global map via Probabilistic WTA."""
        B = base_xy.shape[0]
        device = base_xy.device
        grid_x, grid_y = make_xy_grid(local_h, local_w, local_res, local_cx, local_cy, device)
        # local points in base frame → world
        cos = torch.cos(base_yaw)
        sin = torch.sin(base_yaw)
        # (H,W)
        x_b = grid_x
        y_b = grid_y
        # world: R * p + t
        x_w = cos[:, None, None] * x_b + (-sin[:, None, None]) * y_b + base_xy[:, 0:1, None]
        y_w = sin[:, None, None] * x_b + cos[:, None, None] * y_b + base_xy[:, 1:2, None]

        # map indices
        ix = torch.floor((x_w - self.origin_xy[:, 0:1, None]) / self.resolution + 0.5 * self.n).long()
        iy = torch.floor((y_w - self.origin_xy[:, 1:2, None]) / self.resolution + 0.5 * self.n).long()
        valid = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)

        mu = local_mu[:, 0]
        var_t = local_var[:, 0].clamp_min(1e-6)

        # Gather priors
        b_idx = torch.arange(B, device=device)[:, None, None].expand_as(ix)
        ix_c = ix.clamp(0, self.n - 1)
        iy_c = iy.clamp(0, self.n - 1)
        h_prior = self.height[b_idx, ix_c, iy_c]
        v_prior = self.var[b_idx, ix_c, iy_c].clamp_min(1e-6)

        # Eq 6
        var_hat = torch.maximum(var_t, 0.5 * v_prior)
        # valid update condition
        can = valid & ((var_hat < 1.5 * v_prior) | (var_hat < 0.04))
        # Eq 7
        inv_t = 1.0 / var_hat
        inv_p = 1.0 / v_prior
        p_win = inv_t / (inv_t + inv_p)
        # Eq 8 stochastic WTA
        xi = torch.rand_like(p_win)
        take = can & (xi < p_win)

        new_h = torch.where(take, mu, h_prior)
        new_v = torch.where(take, var_hat, v_prior)

        # Scatter back (only where valid)
        flat_b = b_idx[valid]
        flat_i = ix[valid]
        flat_j = iy[valid]
        self.height[flat_b, flat_i, flat_j] = new_h[valid]
        self.var[flat_b, flat_i, flat_j] = new_v[valid]

    def query_egocentric(
        self,
        base_xy: torch.Tensor,
        base_yaw: torch.Tensor,
        grid_h: int,
        grid_w: int,
        resolution: float,
        center_x: float,
        center_y: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query policy map: returns x,y,z,u each (B,1,H,W) in base frame.

        u is std (sqrt variance), matching student 4th channel usage.
        """
        device = base_xy.device
        B = base_xy.shape[0]
        grid_x, grid_y = make_xy_grid(grid_h, grid_w, resolution, center_x, center_y, device)
        cos = torch.cos(base_yaw)
        sin = torch.sin(base_yaw)
        x_w = cos[:, None, None] * grid_x + (-sin[:, None, None]) * grid_y + base_xy[:, 0:1, None]
        y_w = sin[:, None, None] * grid_x + cos[:, None, None] * grid_y + base_xy[:, 1:2, None]
        ix = torch.floor((x_w - self.origin_xy[:, 0:1, None]) / self.resolution + 0.5 * self.n).long()
        iy = torch.floor((y_w - self.origin_xy[:, 1:2, None]) / self.resolution + 0.5 * self.n).long()
        valid = (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        ix_c = ix.clamp(0, self.n - 1)
        iy_c = iy.clamp(0, self.n - 1)
        b_idx = torch.arange(B, device=device)[:, None, None].expand_as(ix)
        z = self.height[b_idx, ix_c, iy_c]
        v = self.var[b_idx, ix_c, iy_c]
        # Unobserved → high uncertainty sentinel height
        z = torch.where(valid, z, torch.full_like(z, self.init_height))
        v = torch.where(valid, v, torch.full_like(v, self.init_var))
        u = torch.sqrt(v.clamp_min(1e-6))
        x = grid_x.unsqueeze(0).expand(B, -1, -1)
        y = grid_y.unsqueeze(0).expand(B, -1, -1)
        return (
            x.unsqueeze(1),
            y.unsqueeze(1),
            z.unsqueeze(1),
            u.unsqueeze(1),
        )
