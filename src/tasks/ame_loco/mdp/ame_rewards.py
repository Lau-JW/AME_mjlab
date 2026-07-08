"""AME-2 reward functions.

Paper: Section IV-D1, Table I, Eq 1-5
Plus additional locomotion rewards.
All sensor reads use asset.data.XXX pattern from mjlab.
"""

from typing import TYPE_CHECKING
import torch
import math
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_robot = SceneEntityCfg("robot")


# ──────────────────────────────────────────────
# Task Rewards
# ──────────────────────────────────────────────


def track_position(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd = env.command_manager.get_command("goal")
    d_xy = torch.norm(cmd[:, :2], dim=-1)
    reward = 1.0 / (1.0 + 0.25 * d_xy ** 2)
    reward = reward * time_mask(env, 4.0)
    return reward


def track_heading(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd = env.command_manager.get_command("goal")
    d_xy = torch.norm(cmd[:, :2], dim=-1)
    d_yaw = torch.abs(cmd[:, 2])
    reward = 1.0 / (1.0 + d_yaw ** 2)
    reward = reward * time_mask(env, 2.0)
    reward = reward * (d_xy < 0.5).float()
    return reward


def move_to_goal(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd = env.command_manager.get_command("goal")
    d_xy = torch.norm(cmd[:, :2], dim=-1)
    near_goal = (d_xy < 0.5).float()
    try:
        asset = env.scene["robot"]
        base_vel = asset.data.body_link_lin_vel_w[:, 0, :3]  # (B, 3)
        vel_xy = base_vel[:, :2]
        vel_norm = torch.norm(vel_xy, dim=-1)
        goal_dir = cmd[:, :2] / (d_xy.unsqueeze(-1) + 1e-8)
        cos_theta = torch.sum(vel_xy * goal_dir, dim=-1) / (vel_norm + 1e-8)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        moving = (cos_theta > 0.5) & (vel_norm >= 0.3) & (vel_norm <= 2.0)
        return torch.max(near_goal, moving.float())
    except Exception:
        return near_goal


def stand_at_goal(env: "ManagerBasedRlEnv") -> torch.Tensor:
    cmd = env.command_manager.get_command("goal")
    d_xy = torch.norm(cmd[:, :2], dim=-1)
    d_yaw = torch.abs(cmd[:, 2])
    near = ((d_xy < 0.5) & (d_yaw < 0.5)).float()
    try:
        contact_sensor = env.scene["feet_ground_contact"]
        foot_contact = contact_sensor.data.found
        n_feet = foot_contact.shape[-1]
        d_foot = (n_feet - foot_contact.sum(dim=-1).float()) / n_feet
    except Exception:
        d_foot = torch.zeros(env.num_envs, device=env.device)
    try:
        from mjlab.envs.mdp import projected_gravity
        grav = projected_gravity(env)
        d_g = 1.0 - (-grav[:, 2])  # g_z should be ~ -1 when upright
    except Exception:
        d_g = torch.zeros(env.num_envs, device=env.device)
    penalty = (d_foot + d_g + d_xy) / 3.0
    return near * torch.exp(-penalty)


def track_lin_vel_xy_yaw_frame_exp(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Velocity tracking in yaw frame, exponential kernel."""
    cmd = env.command_manager.get_command("goal")
    target_x, target_y = cmd[:, 0], cmd[:, 1]
    try:
        asset = env.scene["robot"]
        base_vel = asset.data.body_link_lin_vel_w[:, 0, :3]
        base_quat = asset.data.body_link_quat_w[:, 0]  # (B, 4)
        # Yaw from quat
        yaw = torch.atan2(
            2 * (base_quat[:, 0] * base_quat[:, 3] + base_quat[:, 1] * base_quat[:, 2]),
            1 - 2 * (base_quat[:, 2] ** 2 + base_quat[:, 3] ** 2),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vel_x = base_vel[:, 0] * cos_yaw + base_vel[:, 1] * sin_yaw
        vel_y = -base_vel[:, 0] * sin_yaw + base_vel[:, 1] * cos_yaw
        error = (vel_x - target_x) ** 2 + (vel_y - target_y) ** 2
        return torch.exp(-error * 10.0)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def track_ang_vel_z_exp(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Angular velocity tracking, exponential kernel."""
    cmd = env.command_manager.get_command("goal")
    target_yaw_rate = cmd[:, 2]
    try:
        asset = env.scene["robot"]
        ang_vel = asset.data.body_link_ang_vel_w[:, 0, :3]
        error = (ang_vel[:, 2] - target_yaw_rate) ** 2
        return torch.exp(-error * 10.0)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Survival
# ──────────────────────────────────────────────


def is_alive(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Regularization & Penalties
# ──────────────────────────────────────────────


def lin_vel_z_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize vertical body velocity squared."""
    try:
        asset = env.scene["robot"]
        vel = asset.data.body_link_lin_vel_w[:, 0, :3]
        return vel[:, 2] ** 2
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def ang_vel_xy_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize roll/pitch angular velocity squared."""
    try:
        asset = env.scene["robot"]
        ang_vel = asset.data.body_link_ang_vel_w[:, 0, :3]
        return torch.sum(ang_vel[:, :2] ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_vel_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint velocities squared."""
    try:
        asset = env.scene["robot"]
        return torch.sum(asset.data.joint_vel ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_acc_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint accelerations squared."""
    try:
        asset = env.scene["robot"]
        return torch.sum(asset.data.joint_acc ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def action_rate_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize action change rate."""
    try:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        return torch.sum((action - prev_action) ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def energy(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Energy penalty: sum(|vel * torque|)."""
    try:
        asset = env.scene["robot"]
        vel = asset.data.joint_vel
        torque = asset.data.actuator_force
        return torch.sum(torch.abs(vel * torque), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_arms(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize arm joint deviation from default pose."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names)
               if any(s in n.lower() for s in ["shoulder", "elbow", "wrist"])]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.mean(torch.abs(jpos[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_waist(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize waist joint deviation from default pose."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names) if "waist" in n.lower()]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.mean(torch.abs(jpos[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_legs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize leg joint deviation from default pose."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names)
               if any(s in n.lower() for s in ["hip", "knee", "ankle"])]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        return torch.mean(torch.abs(jpos[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def flat_orientation_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize deviation from upright torso orientation."""
    try:
        from mjlab.envs.mdp import projected_gravity
        grav = projected_gravity(env)
        return torch.sum(grav[:, :2] ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def base_height_l2(env: "ManagerBasedRlEnv", target_height: float = 0.78) -> torch.Tensor:
    """Penalize deviation from target base height."""
    try:
        asset = env.scene["robot"]
        height = asset.data.body_link_pos_w[:, 0, 2]
        return (height - target_height) ** 2
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def feet_gait(env: "ManagerBasedRlEnv",
              period: float = 0.8,
              offset: tuple[float, float] = (0.0, 0.5),
              threshold: float = 0.5) -> torch.Tensor:
    """Reward foot contact following desired gait pattern."""
    try:
        sensor = env.scene["feet_ground_contact"]
        foot_contact = sensor.data.found.float()
        phase = _gait_phase(env, period)
        left_target = _gait_target(phase, offset[0], threshold)
        right_target = _gait_target(phase, offset[1], threshold)
        reward = (1 - torch.abs(foot_contact[:, 0] - left_target)) \
               + (1 - torch.abs(foot_contact[:, 1] - right_target))
        return reward / 2.0
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def feet_slide(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize foot sliding (foot velocity when in contact)."""
    try:
        sensor = env.scene["feet_ground_contact"]
        contact = sensor.data.found.float()
        asset = env.scene["robot"]
        foot_vel = asset.data.body_link_lin_vel_w[:, -2:, :2]  # last 2 bodies = feet
        slide = torch.sum(foot_vel ** 2, dim=-1)  # (B, 2)
        return torch.mean(slide * contact, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def foot_clearance_reward(env: "ManagerBasedRlEnv",
                          target_height: float = 0.1) -> torch.Tensor:
    """Reward feet clearance during swing phase."""
    try:
        asset = env.scene["robot"]
        foot_height = asset.data.body_link_pos_w[:, -2:, 2]
        sensor = env.scene["feet_ground_contact"]
        contact = sensor.data.found.float()
        swing = 1.0 - contact
        error = torch.abs(foot_height - target_height)
        reward = torch.exp(-error * 10.0) * swing
        return torch.mean(reward, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def undesired_contacts(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize any non-foot contact."""
    try:
        sensor = env.scene["body_contact"]
        return sensor.data.found.any(dim=-1).float()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def time_mask(env: "ManagerBasedRlEnv", T: float) -> torch.Tensor:
    if env.max_episode_length <= 0:
        return torch.ones(env.num_envs, device=env.device)
    dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
    t_left = (env.max_episode_length - env.episode_length_buf.float()) * dt
    return (1.0 / T) * (t_left < T).float()


def _gait_phase(env, period: float) -> torch.Tensor:
    dt = env.cfg.sim.mujoco.timestep * env.cfg.decimation
    progress = env.episode_length_buf.float() * dt
    return (progress % period) / period


def _gait_target(phase: torch.Tensor, offset: float, threshold: float) -> torch.Tensor:
    p = (phase - offset) % 1.0
    return (p < (1.0 - threshold)).float()
