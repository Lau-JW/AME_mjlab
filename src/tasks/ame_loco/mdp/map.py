"""GT elevation map and terrain observation functions.

Paper: Section III-B, Section V
- Teacher: 3-channel (x, y, z) elevation grid, 8cm resolution
- Student: 4-channel (x, y, z, u) with uncertainty
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
import math

from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def sample_gt_elevation_map(
    env: "ManagerBasedRlEnv",
    map_height: int = 36,
    map_width: int = 14,
    resolution: float = 0.08,
    center_x: float = 0.6,
    center_y: float = 0.0,
    sensor_name: str = "elevation_map_scan",
) -> torch.Tensor:
    """Sample ground-truth elevation map using a raycast sensor.

    Returns a grid of (x, y, z) coordinates in the robot's base frame,
    where (x, y) are the grid positions and z is the terrain height.

    Shape: (B, 3, H, W)
      - Channel 0: x coordinates (m)
      - Channel 1: y coordinates (m)
      - Channel 2: z coordinates (terrain height, m)

    Paper: ANYmal-D uses 36x14 at 8cm, centered at (0.6, 0.0) in base frame.
    G1 uses the same but can be adjusted.
    """
    grid_h = map_height
    grid_w = map_width
    N = grid_h * grid_w

    try:
        sensor = env.scene[sensor_name]
    except (KeyError, AttributeError):
        return torch.zeros(
            env.num_envs, 3, grid_h, grid_w,
            device=env.device, dtype=torch.float32,
        )

    hit_pos = sensor.data.hit_pos_w  # (B, N, 3)
    base_pos = sensor.data.pos_w     # (B, 3)

    # quat_apply_inverse flattens quat and vec independently, so the quaternion
    # must be explicitly expanded to one quaternion per ray.
    rel_pos_w = hit_pos - base_pos.unsqueeze(1)
    quat_w = sensor.data.quat_w.unsqueeze(1).expand(-1, N, -1)
    rel_pos = quat_apply_inverse(quat_w, rel_pos_w)

    assert rel_pos.shape[1] == N, f"Expected {N} rays, got {rel_pos.shape[1]}"

    rel_pos = rel_pos.reshape(env.num_envs, grid_h, grid_w, 3)

    # Permute to (B, 3, H, W)
    elevation_map = rel_pos.permute(0, 3, 1, 2)

    # Handle missing hits (distances < 0): set z to a sentinel value
    distances = sensor.data.distances  # (B, N)
    miss_mask = (distances < 0).reshape(env.num_envs, 1, grid_h, grid_w)
    elevation_map = elevation_map.clone()
    elevation_map[:, 2:3, :, :][miss_mask] = -10.0  # sentinel for "no hit"

    return elevation_map


def sample_student_elevation_map(
    env: "ManagerBasedRlEnv",
    map_height: int = 18,
    map_width: int = 13,
    resolution: float = 0.08,
    center_x: float = 0.32,
    center_y: float = 0.0,
    sensor_name: str = "elevation_map_scan",
    miss_uncertainty: float = 2.0,
    hit_uncertainty: float = 0.05,
    corrupt_prob: float = 0.01,
    corrupt_uncertainty: float = 1.5,
) -> torch.Tensor:
    """Phase-1 student map: GT xyz + heuristic uncertainty (paper 4ch).

    Channel 3 approximates occlusion/noise before the full neural mapper lands:
      - missed rays → high uncertainty
      - hits → low uncertainty
      - random corruptions → random z + high uncertainty (paper ~1%)
    Shape: (B, 4, H, W)
    """
    xyz = sample_gt_elevation_map(
        env,
        map_height=map_height,
        map_width=map_width,
        resolution=resolution,
        center_x=center_x,
        center_y=center_y,
        sensor_name=sensor_name,
    )
    B, _, H, W = xyz.shape
    u = torch.full(
        (B, 1, H, W), hit_uncertainty, device=xyz.device, dtype=xyz.dtype
    )
    miss = xyz[:, 2:3] <= -9.0
    u = torch.where(miss, torch.full_like(u, miss_uncertainty), u)

    if corrupt_prob > 0.0 and env.num_envs > 0:
        corrupt = torch.rand(B, 1, H, W, device=xyz.device) < corrupt_prob
        if corrupt.any():
            xyz = xyz.clone()
            rand_z = torch.empty_like(xyz[:, 2:3]).uniform_(-1.0, 1.0)
            xyz[:, 2:3] = torch.where(corrupt, rand_z, xyz[:, 2:3])
            u = torch.where(corrupt, torch.full_like(u, corrupt_uncertainty), u)

    return torch.cat([xyz, u], dim=1)


@dataclass
class OffsetGridPatternCfg:
    size: tuple[float, float]
    resolution: float
    offset: tuple[float, float] = (0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)

    def generate_rays(self, mj_model, device: str):
        del mj_model
        size_x, size_y = self.size
        off_x, off_y = self.offset
        res = self.resolution
        x = torch.arange(
            off_x - size_x / 2, off_x + size_x / 2 + res * 0.5,
            res, device=device, dtype=torch.float32,
        )
        y = torch.arange(
            off_y - size_y / 2, off_y + size_y / 2 + res * 0.5,
            res, device=device, dtype=torch.float32,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        local_offsets = torch.zeros((grid_x.numel(), 3), device=device, dtype=torch.float32)
        local_offsets[:, 0] = grid_x.flatten()
        local_offsets[:, 1] = grid_y.flatten()
        direction = torch.tensor(self.direction, device=device, dtype=torch.float32)
        direction = direction / direction.norm()
        local_directions = direction.unsqueeze(0).expand(grid_x.numel(), 3).clone()
        return local_offsets, local_directions


def create_elevation_map_sensor_cfg(
    map_height: int = 36,
    map_width: int = 14,
    resolution: float = 0.08,
    center_x: float = 0.6,
    center_y: float = 0.0,
    max_distance: float = 5.0,
    frame_name: str = "torso_link",
    sensor_name: str = "elevation_map_scan",
    debug_vis: bool = False,
):
    """Create a dense raycast sensor config for GT elevation mapping.

    Configures a grid of rays at the specified resolution around the
    robot base frame for terrain height measurement.
    """
    from mjlab.sensor import RayCastSensorCfg, ObjRef

    grid_size_x = (map_width - 1) * resolution
    grid_size_y = (map_height - 1) * resolution

    return RayCastSensorCfg(
        name=sensor_name,
        frame=ObjRef(type="body", name=frame_name, entity="robot"),
        ray_alignment="yaw",
        pattern=OffsetGridPatternCfg(
            size=(grid_size_x, grid_size_y),
            resolution=resolution,
            offset=(center_x, center_y),
        ),
        max_distance=max_distance,
        debug_vis=debug_vis,
    )
