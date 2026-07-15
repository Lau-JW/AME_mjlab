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
        base_vel = asset.data.root_link_lin_vel_w
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
        foot_contact = contact_sensor.data.found > 0
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
    try:
        asset = env.scene["robot"]
        dq = torch.mean(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=-1)
    except Exception:
        dq = torch.zeros(env.num_envs, device=env.device)
    penalty = (d_foot + d_g + dq + d_xy) / 4.0
    return near * torch.exp(-penalty)


def track_lin_vel_xy_yaw_frame_exp(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Velocity tracking in yaw frame, exponential kernel."""
    cmd = env.command_manager.get_command("goal")
    target_x, target_y = cmd[:, 0], cmd[:, 1]
    try:
        asset = env.scene["robot"]
        base_vel = asset.data.root_link_lin_vel_w
        base_quat = asset.data.root_link_quat_w
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
        ang_vel = asset.data.root_link_ang_vel_w
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
        vel = asset.data.root_link_lin_vel_w
        return vel[:, 2] ** 2
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def ang_vel_xy_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize roll/pitch angular velocity squared."""
    try:
        asset = env.scene["robot"]
        ang_vel = asset.data.root_link_ang_vel_w
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
    """Penalize arm joint deviation from default pose (relative, Isaac-style)."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names)
               if any(s in n.lower() for s in ["shoulder", "elbow", "wrist"])]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        default = asset.data.default_joint_pos
        return torch.mean(torch.abs(jpos[:, idx] - default[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_waist(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize waist joint deviation from default pose (relative, Isaac-style)."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names) if "waist" in n.lower()]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        default = asset.data.default_joint_pos
        return torch.mean(torch.abs(jpos[:, idx] - default[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_legs(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize leg joint deviation from default pose (hip/knee/ankle)."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [i for i, n in enumerate(names)
               if any(s in n.lower() for s in ["hip", "knee", "ankle"])]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        default = asset.data.default_joint_pos
        return torch.mean(torch.abs(jpos[:, idx] - default[:, idx]), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_deviation_hip(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize only hip yaw/roll deviation (do not constrain pitch/knee/ankle)."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        idx = [
            i for i, n in enumerate(names)
            if ("hip_yaw" in n.lower() or "hip_roll" in n.lower())
        ]
        if not idx:
            return torch.zeros(env.num_envs, device=env.device)
        default = asset.data.default_joint_pos
        return torch.mean(torch.abs(jpos[:, idx] - default[:, idx]), dim=-1)
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
        height = asset.data.root_link_pos_w[:, 2]
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
        foot_contact = (sensor.data.found > 0).float()
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
        contact = (sensor.data.found > 0).float()
        asset = env.scene["robot"]
        foot_vel = asset.data.body_link_lin_vel_w[:, _foot_body_ids(asset), :2]
        slide = torch.sum(foot_vel ** 2, dim=-1)  # (B, 2)
        return torch.mean(slide * contact, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def foot_clearance_reward(env: "ManagerBasedRlEnv",
                          target_height: float = 0.1) -> torch.Tensor:
    """Reward feet clearance during swing phase."""
    try:
        asset = env.scene["robot"]
        foot_height = asset.data.body_link_pos_w[:, _foot_body_ids(asset), 2]
        sensor = env.scene["feet_ground_contact"]
        contact = (sensor.data.found > 0).float()
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
        found = sensor.data.found > 0
        names = _contact_primary_names(sensor)
        if names is not None and len(names) == found.shape[1]:
            mask = torch.tensor(
                ["ankle_roll_link" not in n for n in names],
                device=env.device,
                dtype=torch.bool,
            )
            found = found[:, mask]
        return found.any(dim=-1).float()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Torque & limits (from AME_Locomotion)
# ──────────────────────────────────────────────


def dof_torques_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint torques squared."""
    try:
        asset = env.scene["robot"]
        return torch.sum(asset.data.actuator_force ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def dof_torques_limits(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint torques exceeding 80% of effort limit."""
    try:
        asset = env.scene["robot"]
        torque = torch.abs(asset.data.actuator_force)
        limits = _actuator_effort_limits(asset, env.device).unsqueeze(0)
        excess = torque - 0.8 * limits
        return torch.sum(torch.clamp(excess, min=0), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Feet rewards (from AME_Locomotion)
# ──────────────────────────────────────────────


def feet_air_time(
    env: "ManagerBasedRlEnv",
    threshold: float = 0.6,
    command_name: str = "goal",
    min_goal_distance: float = 0.5,
) -> torch.Tensor:
    """Reward single-stance stepping while far from the goal.

    Inspired by Isaac Lab ``feet_air_time_positive_biped``: encourage one foot in
    air / one in contact, gated so near-goal standing is not rewarded for stepping.
    """
    try:
        sensor = env.scene["feet_ground_contact"]
        air_time = sensor.data.current_air_time
        contact_time = getattr(sensor.data, "current_contact_time", None)
        if air_time is None:
            return torch.zeros(env.num_envs, device=env.device)

        if contact_time is not None:
            in_contact = contact_time > 0.0
            in_mode_time = torch.where(in_contact, contact_time, air_time)
            single_stance = torch.sum(in_contact.int(), dim=1) == 1
            reward = torch.min(
                torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0),
                dim=1,
            )[0]
            reward = torch.clamp(reward, max=threshold)
        else:
            reward = torch.sum(torch.clamp(air_time - threshold, min=0.0, max=0.5), dim=-1)

        cmd = env.command_manager.get_command(command_name)
        d_xy = torch.norm(cmd[:, :2], dim=-1)
        return reward * (d_xy > min_goal_distance).float()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def feet_air_time_variance(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize variance in foot air/contact time (asymmetric gait)."""
    try:
        sensor = env.scene["feet_ground_contact"]
        air_time = sensor.data.current_air_time
        if air_time is not None:
            at = torch.clamp(air_time, min=0.0, max=0.5)
            return torch.var(at, dim=-1)
    except Exception:
        pass
    return torch.zeros(env.num_envs, device=env.device)


def feet_stumble(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (horizontal force > 4× vertical)."""
    try:
        sensor = env.scene["feet_ground_contact"]
        forces = sensor.data.force  # (B, n_feet, 3)
        forces_z = torch.abs(forces[:, :, 2])
        forces_xy = torch.norm(forces[:, :, :2], dim=-1)
        return (forces_xy > 4 * forces_z).any(dim=-1).float()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def feet_too_near(env: "ManagerBasedRlEnv", threshold: float = 0.2) -> torch.Tensor:
    """Penalize feet being too close together."""
    try:
        asset = env.scene["robot"]
        foot_pos = asset.data.body_link_pos_w[:, _foot_body_ids(asset), :]
        distance = torch.norm(foot_pos[:, 0] - foot_pos[:, 1], dim=-1)
        return torch.clamp(threshold - distance, min=0)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def feet_height_body(env: "ManagerBasedRlEnv", target_height: float = 0.1,
                     tanh_mult: float = 2.0) -> torch.Tensor:
    """Reward swinging foot clearance."""
    try:
        asset = env.scene["robot"]
        foot_height = asset.data.body_link_pos_w[:, _foot_body_ids(asset), 2]
        sensor = env.scene["feet_ground_contact"]
        contact = (sensor.data.found > 0).float()
        foot_vel = asset.data.body_link_lin_vel_w[:, _foot_body_ids(asset), :2]
        speed = torch.norm(foot_vel, dim=-1)
        # Reward when foot is in swing (not contact) and moving
        swing = 1.0 - contact
        height_error = (foot_height - target_height) ** 2
        velocity_scale = torch.tanh(tanh_mult * speed)
        reward = height_error * velocity_scale * swing
        return torch.exp(-torch.sum(reward, dim=-1))
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def stand_still(env: "ManagerBasedRlEnv", command_name: str = "goal",
                threshold: float = 0.1) -> torch.Tensor:
    """Penalize joint deviation from default when no command."""
    try:
        cmd = env.command_manager.get_command(command_name)
        cmd_norm = torch.norm(cmd[:, :2], dim=-1)
        asset = env.scene["robot"]
        dev = torch.sum(torch.abs(asset.data.joint_pos), dim=-1)  # default = 0
        return dev * (cmd_norm < threshold).float()
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Link-level penalties (Table I)
# ──────────────────────────────────────────────


def link_contact_forces(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize link contact forces exceeding robot weight (Table I).
    sum(max(F_con - G, 0)^2) * -0.00001
    """
    try:
        sensor = env.scene["body_contact"]
        forces = sensor.data.force  # (B, N, 3)
        force_mag = torch.norm(forces, dim=-1)  # (B, N)
        robot_weight = 35.0 * 9.81  # G1 ~35kg
        excess = torch.clamp(force_mag - robot_weight, min=0)
        return torch.sum(excess ** 2, dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def link_acceleration(env: "ManagerBasedRlEngine") -> torch.Tensor:
    """Penalize link accelerations (Table I). sum_l ||v_l_dot|| * -0.001"""
    try:
        asset = env.scene["robot"]
        # Approximate link acceleration from velocity difference
        # Use body_link_lin_vel_w for all bodies
        lin_vel = asset.data.body_link_lin_vel_w  # (B, N_bodies, 3)
        if hasattr(asset.data, "body_link_lin_vel_w_prev"):
            prev_vel = asset.data.body_link_lin_vel_w_prev
        else:
            prev_vel = lin_vel
        accel = torch.abs(lin_vel - prev_vel)
        return torch.sum(accel.reshape(accel.shape[0], -1), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_velocity_limits(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint velocity exceeding 90% of limits (Table I).
    sum(max(0, |q_dot| - 0.9*q_dot_max))
    """
    try:
        asset = env.scene["robot"]
        vel = torch.abs(asset.data.joint_vel)
        limits = _joint_velocity_limits(asset, env.device).unsqueeze(0)
        excess = vel - 0.9 * limits
        return torch.sum(torch.clamp(excess, min=0), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


def joint_torque_limits(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize joint torque exceeding 80% of limits (Table I).
    sum(max(0, |tau| - 0.8*tau_max))
    """
    try:
        asset = env.scene["robot"]
        torque = torch.abs(asset.data.actuator_force)
        limits = _actuator_effort_limits(asset, env.device).unsqueeze(0)
        excess = torque - 0.8 * limits
        return torch.sum(torch.clamp(excess, min=0), dim=-1)
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


# ──────────────────────────────────────────────
# Joint coordination (from AME_Locomotion)
# ──────────────────────────────────────────────


def joint_coordination(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Reward cross-body coordination: L hip ∥ R shoulder, R hip ∥ L shoulder."""
    try:
        asset = env.scene["robot"]
        jpos = asset.data.joint_pos
        names = asset.joint_names
        def idx(s): return [i for i, n in enumerate(names) if s in n.lower()]
        l_hip = idx("left_hip_pitch")
        r_hip = idx("right_hip_pitch")
        l_shoulder = idx("left_shoulder_pitch")
        r_shoulder = idx("right_shoulder_pitch")
        if not (l_hip and r_hip and l_shoulder and r_shoulder):
            return torch.zeros(env.num_envs, device=env.device)
        pair1 = (jpos[:, l_hip[0]] - jpos[:, r_shoulder[0]]) ** 2
        pair2 = (jpos[:, r_hip[0]] - jpos[:, l_shoulder[0]]) ** 2
        return (pair1 + pair2) / 2
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


def _foot_body_ids(asset) -> list[int]:
    ids, _ = asset.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)
    return ids


def _contact_primary_names(sensor) -> list[str] | None:
    slots = getattr(sensor, "_slots", None)
    if slots is None:
        return None
    return [slot.primary_name for slot in slots if slot.field_name == "found"]


def _actuator_effort_limits(asset, device) -> torch.Tensor:
    limits = []
    for name in asset.actuator_names:
        lname = name.lower()
        if "hip_roll" in lname or "knee" in lname:
            limits.append(139.0)
        elif "hip_pitch" in lname or "hip_yaw" in lname or "waist_yaw" in lname:
            limits.append(88.0)
        elif "wrist_pitch" in lname or "wrist_yaw" in lname:
            limits.append(5.0)
        else:
            limits.append(25.0)
    return torch.tensor(limits, device=device, dtype=torch.float32)


def _joint_velocity_limits(asset, device) -> torch.Tensor:
    limits = []
    for name in asset.joint_names:
        lname = name.lower()
        if "hip_roll" in lname or "knee" in lname:
            limits.append(20.0)
        elif "hip_pitch" in lname or "hip_yaw" in lname or "waist_yaw" in lname:
            limits.append(32.0)
        elif "wrist_pitch" in lname or "wrist_yaw" in lname:
            limits.append(22.0)
        else:
            limits.append(37.0)
    return torch.tensor(limits, device=device, dtype=torch.float32)
