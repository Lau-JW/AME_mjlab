"""Goal-reaching command for AME-2.

The command term stores a world-frame goal pose and exposes the current
base-relative goal as ``[x_b, y_b, yaw_error, remaining_episode_time]``.
Actor observations can clip/drop parts of this command while rewards and critic
observations keep the full target.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
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
        self.has_active_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_pose_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.distance_sum = torch.zeros(self.num_envs, device=self.device)
        self.yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.metric_steps = torch.zeros(self.num_envs, device=self.device)
        self.final_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.final_goal_yaw_error = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_command_b

    def _update_metrics(self) -> None:
        distance, yaw_error = self.goal_error()
        self.distance_sum += distance
        self.yaw_error_sum += yaw_error
        self.metric_steps += 1.0
        self.final_goal_distance.copy_(distance)
        self.final_goal_yaw_error.copy_(yaw_error)
        self.goal_reached |= distance < self.cfg.position_threshold
        self.goal_pose_reached |= (
            (distance < self.cfg.position_threshold)
            & (yaw_error < self.cfg.heading_threshold)
        )

    def compute(self, dt: float) -> None:
        if not self.cfg.resample_on_reset_only:
            super().compute(dt)
            if self.cfg.resample_on_success:
                self._resample_reached_goals()
            return
        self._update_metrics()
        self.time_left -= dt
        self._update_command()
        if self.cfg.resample_on_success:
            self._resample_reached_goals()

    def _resample_reached_goals(self) -> None:
        """Resample a new goal as soon as the current one is reached."""
        distance, _ = self.goal_error()
        reached = (
            self.has_active_goal
            & ~self.is_standing_env
            & (distance < self.cfg.position_threshold)
        )
        env_ids = reached.nonzero().flatten()
        if len(env_ids) == 0:
            return
        self.goal_reached[env_ids] = False
        self.goal_pose_reached[env_ids] = False
        self._resample(env_ids)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        assert isinstance(env_ids, torch.Tensor)
        valid = self.has_active_goal[env_ids] & ~self.is_standing_env[env_ids]
        steps = torch.clamp(self.metric_steps[env_ids], min=1.0)

        def valid_mean(values: torch.Tensor) -> float:
            return torch.mean(values[valid]).item() if torch.any(valid) else 0.0

        extras = {
            "goal_distance": valid_mean(self.distance_sum[env_ids] / steps),
            "goal_yaw_error": valid_mean(self.yaw_error_sum[env_ids] / steps),
            "final_goal_distance": valid_mean(self.final_goal_distance[env_ids]),
            "final_goal_yaw_error": valid_mean(self.final_goal_yaw_error[env_ids]),
            "goal_success": valid_mean(self.goal_reached[env_ids].float()),
            "goal_pose_success": valid_mean(self.goal_pose_reached[env_ids].float()),
        }

        self.goal_reached[env_ids] = False
        self.goal_pose_reached[env_ids] = False
        self.distance_sum[env_ids] = 0.0
        self.yaw_error_sum[env_ids] = 0.0
        self.metric_steps[env_ids] = 0.0
        self.final_goal_distance[env_ids] = 0.0
        self.final_goal_yaw_error[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def goal_error(
        self, env_ids: torch.Tensor | slice | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        base_xy_w = self.robot.data.root_link_pos_w[env_ids, :2]
        distance = torch.norm(self.goal_pos_w[env_ids] - base_xy_w, dim=-1)
        yaw_error = torch.abs(
            wrap_to_pi(self.goal_yaw_w[env_ids] - self.robot.data.heading_w[env_ids])
        )
        return distance, yaw_error

    def record_terminal_outcome(
        self,
        env_ids: torch.Tensor,
        distance: torch.Tensor,
        yaw_error: torch.Tensor,
    ) -> None:
        self.final_goal_distance[env_ids] = distance
        self.final_goal_yaw_error[env_ids] = yaw_error
        self.goal_reached[env_ids] |= distance < self.cfg.position_threshold
        self.goal_pose_reached[env_ids] |= (
            (distance < self.cfg.position_threshold)
            & (yaw_error < self.cfg.heading_threshold)
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        stand = (
            torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
            <= self.cfg.rel_standing_envs
        )
        self.is_standing_env[env_ids] = stand

        distance = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.distance
        )
        direction_b = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.direction
        )
        rel_x_b = distance * torch.cos(direction_b)
        rel_y_b = distance * torch.sin(direction_b)

        root_qpos = self.robot.data.data.qpos[env_ids][
            :, self.robot.data.indexing.free_joint_q_adr
        ]
        base_xy_w = root_qpos[:, :2]
        root_quat_w = root_qpos[:, 3:7]
        heading_w = torch.atan2(
            2.0
            * (
                root_quat_w[:, 0] * root_quat_w[:, 3]
                + root_quat_w[:, 1] * root_quat_w[:, 2]
            ),
            1.0 - 2.0 * (root_quat_w[:, 2].square() + root_quat_w[:, 3].square()),
        )
        cos_yaw = torch.cos(heading_w)
        sin_yaw = torch.sin(heading_w)
        rel_x_w = rel_x_b * cos_yaw - rel_y_b * sin_yaw
        rel_y_w = rel_x_b * sin_yaw + rel_y_b * cos_yaw
        self.goal_pos_w[env_ids, 0] = base_xy_w[:, 0] + rel_x_w
        self.goal_pos_w[env_ids, 1] = base_xy_w[:, 1] + rel_y_w

        rel_yaw = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.yaw
        )
        self.goal_yaw_w[env_ids] = wrap_to_pi(heading_w + rel_yaw)
        self.has_active_goal[env_ids] = True

        if torch.any(stand):
            stand_ids = env_ids[stand]
            self.goal_pos_w[stand_ids] = base_xy_w[stand]
            self.goal_yaw_w[stand_ids] = heading_w[stand]

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
    resample_on_reset_only: bool = True
    resample_on_success: bool = False
    """If True, sample a new goal immediately after reaching the current one."""
    position_threshold: float = 0.5
    heading_threshold: float = 0.5

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


class TerrainLevelGoal:
    """Paper-style terrain curriculum with per-environment success-rate EMA."""

    def __init__(self, cfg: Any, env: "ManagerBasedRlEnv"):
        del cfg
        self.success_ema = torch.zeros(env.num_envs, device=env.device)
        self._skip_next_update = False

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor | slice,
        command_name: str,
        ema_alpha: float = 0.1,
        promotion_threshold: float = 0.5,
        demotion_distance: float = 4.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg
        terrain = env.scene.terrain
        assert terrain is not None
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]

        if self._skip_next_update:
            self._skip_next_update = False
            return torch.mean(terrain.terrain_levels.float())

        goal = env.command_manager.get_term(command_name)
        if not isinstance(goal, UniformGoalCommand):
            raise TypeError(f"Command '{command_name}' must be UniformGoalCommand")

        distance, yaw_error = goal.goal_error(env_ids)
        goal.record_terminal_outcome(env_ids, distance, yaw_error)

        valid = goal.has_active_goal[env_ids] & ~goal.is_standing_env[env_ids]
        success = goal.goal_reached[env_ids]
        valid_ids = env_ids[valid]
        if len(valid_ids) > 0:
            self.success_ema[valid_ids] = (
                (1.0 - ema_alpha) * self.success_ema[valid_ids]
                + ema_alpha * success[valid].float()
            )

        move_up = valid & success & (
            self.success_ema[env_ids] > promotion_threshold
        )
        move_down = valid & (distance > demotion_distance) & ~move_up
        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"success_ema": self.success_ema.detach().cpu()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        success_ema = state.get("success_ema")
        if success_ema is None:
            return
        if success_ema.shape != self.success_ema.shape:
            raise ValueError(
                "Curriculum success_ema shape mismatch: "
                f"checkpoint={tuple(success_ema.shape)}, "
                f"environment={tuple(self.success_ema.shape)}"
            )
        self.success_ema.copy_(success_ema.to(self.success_ema.device))

    def suspend_next_update(self) -> None:
        self._skip_next_update = True
