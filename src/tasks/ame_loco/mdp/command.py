"""Goal-reaching command for AME-2.

Paper: Section III-A, III-B
- Robot gets goal (x, y, yaw) relative to current position
- Actor sees clipped distance (max 2m), no remaining time (continuous deployment)
- Critic sees full command + remaining time
- Goal resampled every 5-10 seconds

The command is stored as a 3D vector in the command buffer:
  [goal_x, goal_y, goal_yaw]  (relative to robot base frame)
"""

from typing import TYPE_CHECKING
import torch
import math

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def sample_goal_command(
    env: "ManagerBasedRlEnv",
    command_name: str = "goal",
    resampling_time_range: tuple[float, float] = (5.0, 10.0),
    goal_dist_range: tuple[float, float] = (1.0, 5.0),
    heading_range: tuple[float, float] = (-math.pi, math.pi),
    standing_init_ratio: float = 0.05,
):
    """Sample a new goal command for a subset of environments.

    Args:
        env: The environment instance.
        command_name: Name of the command in the command manager.
        resampling_time_range: Range of resampling intervals in seconds.
        goal_dist_range: Range of goal distances in meters.
        heading_range: Range of goal heading offsets.
        standing_init_ratio: Fraction of envs that get zero command (standing).
    """
    num_envs = env.num_envs
    device = env.device

    # Determine which envs get new commands
    resample = env.episode_length_buf == 0  # reset on new episode
    if hasattr(env, "command_manager"):
        cmd_manager = env.command_manager
        if hasattr(cmd_manager, "resample_buf"):
            resample = resample | cmd_manager.resample_buf

    n_resample = resample.sum().item()
    if n_resample == 0:
        return

    # Command dimensions: (N, 3) = (goal_x, goal_y, goal_yaw)
    cmd = torch.zeros(num_envs, 3, device=device)

    # Standing envs get zero command
    n_stand = max(1, int(num_envs * standing_init_ratio))
    stand_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    stand_indices = torch.randperm(num_envs, device=device)[:n_stand]
    stand_mask[stand_indices] = True

    # Moving envs get random goal commands
    move_mask = resample & ~stand_mask
    n_move = move_mask.sum().item()

    if n_move > 0:
        # Sample goal distance
        goal_dist = torch.empty(n_move, device=device).uniform_(
            goal_dist_range[0], goal_dist_range[1]
        )
        # Sample goal direction (uniform angle)
        goal_angle = torch.empty(n_move, device=device).uniform_(-math.pi, math.pi)
        # Sample heading at goal
        goal_yaw = torch.empty(n_move, device=device).uniform_(
            heading_range[0], heading_range[1]
        )

        cmd[move_mask, 0] = goal_dist * torch.cos(goal_angle)
        cmd[move_mask, 1] = goal_dist * torch.sin(goal_angle)
        cmd[move_mask, 2] = goal_yaw

    # Write to command buffer
    # If env has command_manager, update it
    if hasattr(env, "command_manager"):
        cmd_manager = env.command_manager
        if hasattr(cmd_manager, "_cmd_buffer"):
            cmd_manager._cmd_buffer[resample] = cmd[resample]


def get_goal_command(
    env: "ManagerBasedRlEnv", command_name: str = "goal", clipped: bool = False
) -> torch.Tensor:
    """Get the current goal command.

    Args:
        env: The environment.
        command_name: Command name.
        clipped: If True, clip goal distance to 2m (actor mode).

    Returns:
        Tensor (B, 3) with (goal_x, goal_y, goal_yaw).
    """
    from mjlab.envs.mdp import generated_commands
    cmd = generated_commands(env, command_name)

    if clipped:
        # Clip distance to 2m (Section III-B: continuous deployment)
        d_xy = torch.norm(cmd[:, :2], dim=-1)
        scale = torch.clamp(2.0 / (d_xy + 1e-8), max=1.0)
        cmd[:, 0] = cmd[:, 0] * scale
        cmd[:, 1] = cmd[:, 1] * scale

    return cmd
