"""Goal-reaching command for AME-2.

The command term stores a world-frame goal pose and exposes the current
base-relative goal as ``[x_b, y_b, yaw_error, remaining_episode_time]``.
Actor observations can clip/drop parts of this command while rewards and critic
observations keep the full target.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
import math

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class UniformGoalCommand(CommandTerm):
    cfg: "UniformGoalCommandCfg"

    def __init__(self, cfg: "UniformGoalCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.goal_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self.goal_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_yaw_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_command_b

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["goal_distance"] += torch.norm(self.goal_command_b[:, :2], dim=-1) / max_command_step
        self.metrics["goal_yaw_error"] += torch.abs(self.goal_command_b[:, 2]) / max_command_step

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        stand = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        self.is_standing_env[env_ids] = stand

        distance = r.uniform_(*self.cfg.ranges.distance)
        direction_b = r.uniform_(*self.cfg.ranges.direction)
        rel_x_b = distance * torch.cos(direction_b)
        rel_y_b = distance * torch.sin(direction_b)

        heading_w = self.robot.data.heading_w[env_ids]
        cos_yaw = torch.cos(heading_w)
        sin_yaw = torch.sin(heading_w)
        rel_x_w = rel_x_b * cos_yaw - rel_y_b * sin_yaw
        rel_y_w = rel_x_b * sin_yaw + rel_y_b * cos_yaw
        base_xy_w = self.robot.data.root_link_pos_w[env_ids, :2]
        self.goal_pos_w[env_ids, 0] = base_xy_w[:, 0] + rel_x_w
        self.goal_pos_w[env_ids, 1] = base_xy_w[:, 1] + rel_y_w

        rel_yaw = r.uniform_(*self.cfg.ranges.yaw)
        self.goal_yaw_w[env_ids] = wrap_to_pi(heading_w + rel_yaw)

        if torch.any(stand):
            stand_ids = env_ids[stand]
            self.goal_pos_w[stand_ids] = self.robot.data.root_link_pos_w[stand_ids, :2]
            self.goal_yaw_w[stand_ids] = self.robot.data.heading_w[stand_ids]

    def _update_command(self) -> None:
        base_xy_w = self.robot.data.root_link_pos_w[:, :2]
        delta_w = self.goal_pos_w - base_xy_w
        heading_w = self.robot.data.heading_w
        cos_yaw = torch.cos(heading_w)
        sin_yaw = torch.sin(heading_w)
        self.goal_command_b[:, 0] = delta_w[:, 0] * cos_yaw + delta_w[:, 1] * sin_yaw
        self.goal_command_b[:, 1] = -delta_w[:, 0] * sin_yaw + delta_w[:, 1] * cos_yaw
        self.goal_command_b[:, 2] = wrap_to_pi(self.goal_yaw_w - heading_w)
        dt = self._env.step_dt
        t_left = (self._env.max_episode_length - self._env.episode_length_buf.float()) * dt
        self.goal_command_b[:, 3] = torch.clamp(t_left, min=0.0)


@dataclass(kw_only=True)
class UniformGoalCommandCfg(CommandTermCfg):
    entity_name: str
    rel_standing_envs: float = 0.05

    @dataclass
    class Ranges:
        distance: tuple[float, float] = (1.0, 5.0)
        direction: tuple[float, float] = (-math.pi, math.pi)
        yaw: tuple[float, float] = (-math.pi, math.pi)

    ranges: Ranges = field(default_factory=Ranges)

    def build(self, env: "ManagerBasedRlEnv") -> UniformGoalCommand:
        return UniformGoalCommand(self, env)


def goal_command_actor(env: "ManagerBasedRlEnv", command_name: str = "goal",
                       max_distance: float = 2.0,
                       randomize_far_yaw: bool = False) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name).clone()
    d_xy = torch.norm(cmd[:, :2], dim=-1)
    scale = torch.clamp(max_distance / (d_xy + 1e-8), max=1.0)
    cmd[:, 0] *= scale
    cmd[:, 1] *= scale
    if randomize_far_yaw:
        far = d_xy > max_distance
        if torch.any(far):
            cmd[far, 2] = torch.empty_like(cmd[far, 2]).uniform_(-math.pi, math.pi)
    return cmd[:, :3]


def goal_command_critic(env: "ManagerBasedRlEnv", command_name: str = "goal") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def terrain_levels_goal(
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    del asset_cfg
    terrain = env.scene.terrain
    assert terrain is not None
    command = env.command_manager.get_command(command_name)
    d_xy = torch.norm(command[env_ids, :2], dim=1)
    move_up = d_xy < 0.5
    move_down = (d_xy > 4.0) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
