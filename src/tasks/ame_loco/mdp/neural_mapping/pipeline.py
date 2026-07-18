"""Online AME-2 neural mapping pipeline for parallel simulation (Sec V)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ObjRef, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import PinholeCameraPatternCfg

from .fusion import GlobalElevationMap
from .gated_unet import GatedElevationUNet
from .project import project_points_to_height_grid

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


# TRON1 / G1 defaults from AME-2 paper
LOCAL_H, LOCAL_W = 31, 31
LOCAL_RES = 0.04
LOCAL_CX = 0.6
POLICY_H, POLICY_W = 18, 13
POLICY_RES = 0.08
POLICY_CX = 0.32


def create_depth_cloud_sensor_cfg(
    width: int = 64,
    height: int = 48,
    fovy: float = 70.0,
    frame_name: str = "torso_link",
    sensor_name: str = "depth_cloud",
    max_distance: float = 5.0,
    debug_vis: bool = False,
) -> RayCastSensorCfg:
    """Forward-facing pinhole raycast ≈ depth camera point cloud (sim-friendly)."""
    return RayCastSensorCfg(
        name=sensor_name,
        frame=ObjRef(type="body", name=frame_name, entity="robot"),
        ray_alignment="base",  # full orientation, not yaw-only
        pattern=PinholeCameraPatternCfg(width=width, height=height, fovy=fovy),
        max_distance=max_distance,
        debug_vis=debug_vis,
    )


class NeuralMappingPipeline:
    """Depth cloud → local grid → U-Net → global WTA → egocentric 4ch query."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        mapper_ckpt: str | Path | None = None,
        depth_sensor_name: str = "depth_cloud",
        corrupt_prob: float = 0.01,
        missing_ratio: float = 0.15,
    ):
        self.num_envs = num_envs
        self.device = device
        self.depth_sensor_name = depth_sensor_name
        self.corrupt_prob = corrupt_prob
        self.missing_ratio = missing_ratio
        self.mapper = GatedElevationUNet(base_channels=16).to(device)
        self.mapper.eval()
        for p in self.mapper.parameters():
            p.requires_grad_(False)
        if mapper_ckpt is not None and Path(mapper_ckpt).exists():
            state = torch.load(mapper_ckpt, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.mapper.load_state_dict(state, strict=True)
            print(f"[AME-Map] Loaded mapper weights: {mapper_ckpt}")
        else:
            print(
                "[AME-Map] WARNING: no mapper checkpoint; using randomly initialized U-Net. "
                "Run scripts/train_mapper.py first for paper-aligned maps."
            )
        self.global_map = GlobalElevationMap(num_envs, device)
        self._initialized = False

    @torch.no_grad()
    def reset(self, env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        robot = env.scene["robot"]
        base_xy = robot.data.root_link_pos_w[env_ids, :2]
        self.global_map.reset(env_ids, base_xy)
        self._initialized = True

    @torch.no_grad()
    def update_and_query(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
        """One mapping step; returns (B, 4, POLICY_H, POLICY_W) xyz+u in base frame."""
        # Reset mapping when episodes restart.
        done = env.episode_length_buf == 0
        if done.any():
            self.reset(env, done.nonzero(as_tuple=False).squeeze(-1))

        robot = env.scene["robot"]
        base_pos = robot.data.root_link_pos_w
        base_xy = base_pos[:, :2]
        base_yaw = robot.data.heading_w
        self.global_map.maybe_recenter(base_xy)

        # Depth hits in world → base frame
        try:
            sensor = env.scene[self.depth_sensor_name]
            hit_w = sensor.data.hit_pos_w  # (B, N, 3)
            dist = sensor.data.distances
            valid = dist > 0
        except Exception:
            # Fallback: empty cloud
            B = env.num_envs
            return torch.zeros(B, 4, POLICY_H, POLICY_W, device=self.device)

        # Transform hits to yaw-aligned base frame (gravity-up height maps).
        yaw = base_yaw
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        rel = hit_w - base_pos.unsqueeze(1)
        x_b = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 1]
        y_b = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1]
        z_b = rel[..., 2]
        pts_b = torch.stack([x_b, y_b, z_b], dim=-1)

        # Depth DR: missing points + artifacts (paper Appendix)
        if self.missing_ratio > 0:
            drop = torch.rand_like(dist) < self.missing_ratio
            valid = valid & ~drop
        if self.corrupt_prob > 0:
            corrupt = torch.rand_like(dist) < self.corrupt_prob
            if corrupt.any():
                pts_b = pts_b.clone()
                pts_b[..., 2] = torch.where(
                    corrupt,
                    pts_b[..., 2] + torch.empty_like(pts_b[..., 2]).uniform_(-0.5, 0.5),
                    pts_b[..., 2],
                )

        local = project_points_to_height_grid(
            pts_b, valid, LOCAL_H, LOCAL_W, LOCAL_RES, LOCAL_CX, 0.0, empty_value=-1.0
        )
        mu, log_var, _ = self.mapper(local)
        var = torch.exp(log_var).clamp_min(1e-6)
        # Fuse world-Z = base_z + base-relative mu
        mu_world = mu + base_pos[:, 2].view(-1, 1, 1, 1)
        self.global_map.fuse_local(
            base_xy, base_yaw, mu_world, var,
            LOCAL_RES, LOCAL_H, LOCAL_W, LOCAL_CX, 0.0,
        )
        x, y, z_w, u = self.global_map.query_egocentric(
            base_xy, base_yaw, POLICY_H, POLICY_W, POLICY_RES, POLICY_CX, 0.0
        )
        z_b_out = z_w - base_pos[:, 2].view(-1, 1, 1, 1)
        return torch.cat([x, y, z_b_out, u], dim=1)


_PIPELINE_ATTR = "_ame_neural_mapping"


def get_or_create_pipeline(
    env: "ManagerBasedRlEnv",
    mapper_ckpt: str | None = None,
) -> NeuralMappingPipeline:
    pipe = getattr(env, _PIPELINE_ATTR, None)
    if pipe is None or pipe.num_envs != env.num_envs:
        ckpt = mapper_ckpt
        if ckpt is None:
            # default search
            for cand in (
                "logs/mappers/g1_elevation_unet.pt",
                "/data_nvme/getup/AME_mjlab/logs/mappers/g1_elevation_unet.pt",
            ):
                if Path(cand).exists():
                    ckpt = cand
                    break
        pipe = NeuralMappingPipeline(env.num_envs, env.device, mapper_ckpt=ckpt)
        setattr(env, _PIPELINE_ATTR, pipe)
        pipe.reset(env)
    return pipe


def sample_neural_elevation_map(
    env: "ManagerBasedRlEnv",
    mapper_ckpt: str | None = None,
    map_height: int = POLICY_H,
    map_width: int = POLICY_W,
    resolution: float = POLICY_RES,
    center_x: float = POLICY_CX,
    center_y: float = 0.0,
    sensor_name: str = "elevation_map_scan",
    **_: object,
) -> torch.Tensor:
    """Observation term: neural 4ch map, with optional GT mix curriculum.

    ``env._ame_map_gt_mix`` ∈ [0,1] (set by student runner) selects a fraction of
    envs that observe GT xyz+heuristic-u instead of the neural map. Teacher
    training never calls this term.
    """
    pipe = get_or_create_pipeline(env, mapper_ckpt=mapper_ckpt)
    neural = pipe.update_and_query(env)
    mix = float(getattr(env, "_ame_map_gt_mix", 0.0))
    if mix <= 0.0:
        return neural

    # Lazy import avoids circular deps with mdp.map.
    from src.tasks.ame_loco.mdp.map import sample_student_elevation_map

    gt4 = sample_student_elevation_map(
        env,
        map_height=map_height,
        map_width=map_width,
        resolution=resolution,
        center_x=center_x,
        center_y=center_y,
        sensor_name=sensor_name,
        corrupt_prob=0.0,
    )
    # Match channel layout / spatial size if GT grid differs.
    if gt4.shape[-2:] != neural.shape[-2:]:
        gt4 = torch.nn.functional.interpolate(
            gt4, size=neural.shape[-2:], mode="bilinear", align_corners=False
        )
    use_gt = torch.rand(neural.shape[0], device=neural.device) < mix
    mask = use_gt.view(-1, 1, 1, 1)
    return torch.where(mask, gt4, neural)
