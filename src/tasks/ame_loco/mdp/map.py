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

from src.tasks.ame_loco.mdp.neural_mapping import (
    NeuralMapCfg,
    StudentMappingState,
    project_depth_to_local_grid,
)

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
    map_channels: int = 3,
    **kwargs,
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
        debug_vis=True,
    )


# ---------------------------------------------------------------------------
# Student neural mapping observation helpers
# ---------------------------------------------------------------------------


def ensure_student_mapping_state(env: "ManagerBasedRlEnv", cfg: NeuralMapCfg | None = None) -> StudentMappingState:
    """Create or return the per-environment neural mapping state."""
    if not hasattr(env, "student_mapping_state") or env.student_mapping_state is None:
        checkpoint = getattr(env, "student_mapping_checkpoint", None)
        env.student_mapping_state = StudentMappingState(
            num_envs=env.num_envs,
            device=env.device,
            cfg=cfg,
            checkpoint_path=checkpoint,
        )
    return env.student_mapping_state


def simulate_depth_cloud_from_gt(
    gt_map: torch.Tensor,
    center_x: float = 0.32,
    center_y: float = 0.0,
    fov_deg: float = 80.0,
    max_range: float = 5.0,
    drop_ratio: float = 0.05,
    outlier_ratio: float = 0.01,
    noise_std: float = 0.02,
) -> torch.Tensor:
    """Simulate a noisy/visible subset of the GT elevation map as a depth cloud.

    This is used to train the student in simulation when no real depth camera
    is available. The resulting point cloud is in the robot base frame.

    Args:
        gt_map: (B, 3, H, W) ground-truth elevation map with x, y, z channels.
        center_x, center_y: offset of the grid center in base frame.
        fov_deg: horizontal field of view of the simulated depth camera.
        max_range: maximum distance from robot to keep points.
        drop_ratio: random point dropout ratio.
        outlier_ratio: random outlier ratio.
        noise_std: additive Gaussian noise on z.

    Returns:
        points: (B, N, 3) base-frame point cloud (x, y, z). Variable N per env.
        mask: (B, N) valid mask.
    """
    B, _, H, W = gt_map.shape
    device = gt_map.device

    x = gt_map[:, 0].reshape(B, H * W)
    y = gt_map[:, 1].reshape(B, H * W)
    z = gt_map[:, 2].reshape(B, H * W)

    # Translate so that robot is at (0, 0)
    x_rel = x - center_x
    y_rel = y - center_y
    r = torch.sqrt(x_rel * x_rel + y_rel * y_rel + 1e-8)
    yaw = torch.atan2(y_rel, x_rel)

    half_fov = math.radians(fov_deg / 2.0)
    in_fov = (yaw.abs() < half_fov) & (r < max_range) & (z > -5.0)

    # Add noise to z
    z_noisy = z + torch.randn_like(z) * noise_std

    # Build point cloud padded to max visible points
    max_points = in_fov.sum(dim=1).max().item()
    points = torch.full((B, max_points, 3), -100.0, device=device, dtype=torch.float32)
    mask = torch.zeros(B, max_points, dtype=torch.bool, device=device)

    for b in range(B):
        valid = in_fov[b]
        n = valid.sum().item()
        if n == 0:
            continue
        pts = torch.stack([x[b][valid], y[b][valid], z_noisy[b][valid]], dim=-1)

        # Random dropout
        if drop_ratio > 0 and n > 0:
            keep = torch.rand(n, device=device) >= drop_ratio
            pts = pts[keep]
            n = pts.shape[0]

        # Add outliers
        if outlier_ratio > 0 and n > 0:
            n_out = max(1, int(n * outlier_ratio))
            outlier_idx = torch.randperm(n, device=device)[:n_out]
            pts[outlier_idx, 2] += torch.randn(n_out, device=device) * 0.3

        points[b, :n] = pts
        mask[b, :n] = True

    return points, mask


def sample_student_elevation_map(
    env: "ManagerBasedRlEnv",
    map_height: int = 18,
    map_width: int = 13,
    resolution: float = 0.08,
    center_x: float = 0.32,
    center_y: float = 0.0,
    sensor_name: str = "elevation_map_scan",
    cfg: NeuralMapCfg | None = None,
    map_channels: int = 4,
    proprio_single_dim: int = 93,
    proprio_history_len: int = 20,
    **kwargs,
) -> torch.Tensor:
    """Produce a student 4-channel elevation map (x, y, z, u) via neural mapping.

    In simulation this uses a synthetic depth cloud derived from the GT sensor.
    On real hardware the depth cloud would come from the depth camera.
    """
    # Get GT map as the source of synthetic depth
    gt_map = sample_gt_elevation_map(
        env,
        map_height=map_height,
        map_width=map_width,
        resolution=resolution,
        center_x=center_x,
        center_y=center_y,
        sensor_name=sensor_name,
    )

    state = ensure_student_mapping_state(env, cfg)

    # Simulated depth cloud from the GT map
    points, _ = simulate_depth_cloud_from_gt(
        gt_map,
        center_x=center_x,
        center_y=center_y,
    )

    # Get base world pose for map fusion
    sensor = env.scene[sensor_name]
    base_pos = sensor.data.pos_w
    base_quat = sensor.data.quat_w
    # Compute yaw from quaternion (x,y,z,w convention in MuJoCo)
    w, x, y, z = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    base_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return state.update_from_depth(points, base_pos, base_yaw)
