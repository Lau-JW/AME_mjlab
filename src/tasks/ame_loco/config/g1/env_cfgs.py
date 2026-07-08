"""G1 AME-2 environment configuration.

Paper: AME-2 (arXiv:2601.08485)
- Observations: Section III-B, Fig 3 (Right)
- Rewards: Section IV-D1, Table I
- Terminations: Section IV-D2
- Terrains & Curriculum: Section IV-D3, Appendix A
- Domain Randomization: Section IV-D4, Appendix B
"""

import math
from dataclasses import replace

import src.tasks.ame_loco.mdp.ame_rewards as rwd
import src.tasks.ame_loco.mdp.ame_terminations as term
from src.tasks.ame_loco.mdp.map import (
    create_elevation_map_sensor_cfg,
    sample_gt_elevation_map,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, terrain_levels_vel
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg


def g1_ame_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create G1 AME-2 environment config (teacher)."""

    ##
    # Sensors — terrain scan + elevation map + foot contacts
    ##
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        debug_vis=True,
    )
    # GT elevation map sensor (dense grid, teacher)
    # 18×7 grid at 8cm, centered at (0.6, 0) — reduced from 36×14 for G1/memory
    elev_map_sensor = create_elevation_map_sensor_cfg(
        map_height=18, map_width=7, resolution=0.08,
        center_x=0.6, center_y=0.0,
        frame_name="torso_link",
        sensor_name="elevation_map_scan",
    )
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Body contact sensor for critic (Sec IV-B: contact state of each link)
    # Monitors all key robot bodies against terrain
    body_names_str = "|".join([
        "pelvis", "torso_link",
        "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ])
    body_contact_cfg = ContactSensorCfg(
        name="body_contact",
        primary=ContactMatch(
            mode="subtree", pattern=body_names_str, entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    ##
    # Observations (Fig 3 Right)
    ##
    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "goal"},
        ),
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        # Elevation map — GT elevation map (3ch: xyz, 18x7 grid)
        "elevation_map": ObservationTermCfg(
            func=sample_gt_elevation_map,
            params={
                "map_height": 18, "map_width": 7,
                "resolution": 0.08,
                "center_x": 0.6, "center_y": 0.0,
                "sensor_name": "elevation_map_scan",
            },
        ),
    }
    def _body_contact(env):
        """Contact state of each link (Sec IV-B). Returns (B, N) binary flags."""
        try:
            sensor = env.scene["body_contact"]
            return sensor.data.found.float()  # (B, N_contacts)
        except Exception:
            return torch.zeros(env.num_envs, 14, device=env.device)

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "body_contact": ObservationTermCfg(func=_body_contact),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    ##
    # Actions
    ##
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    ##
    # Commands — Goal reaching (Sec III-A)
    ##
    commands: dict[str, CommandTermCfg] = {
        "goal": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 10.0),
            rel_standing_envs=0.05,
            heading_command=True,
            heading_control_stiffness=0.5,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 3.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-math.pi / 4, math.pi / 4),
                heading=(-math.pi, math.pi),
            ),
        )
    }

    ##
    # Events — Domain Randomization (Sec IV-D4, Appendix B)
    ##
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )
    events = {
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5),
                    "z": (0.0, 0.0), "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.0, 0.0),
                "velocity_range": (-0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=envs_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "ranges": (0.3, 1.0),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
            },
        ),
    }

    ##
    # Rewards — Table I
    ##
    rewards = {
        # Task
        "track_position": RewardTermCfg(func=rwd.track_position, weight=100.0),
        "track_heading": RewardTermCfg(func=rwd.track_heading, weight=50.0),
        "move_to_goal": RewardTermCfg(func=rwd.move_to_goal, weight=5.0),
        "stand_at_goal": RewardTermCfg(func=rwd.stand_at_goal, weight=5.0),
        "track_lin_vel_xy_yaw_frame_exp": RewardTermCfg(
            func=rwd.track_lin_vel_xy_yaw_frame_exp, weight=1.0,
        ),
        "track_ang_vel_z_exp": RewardTermCfg(
            func=rwd.track_ang_vel_z_exp, weight=0.5,
        ),
        # Survival
        "is_alive": RewardTermCfg(func=rwd.is_alive, weight=0.15),
        # Early termination (Table I: -10/d_tau, fires only on bad_orientation/base_collision)
        "early_termination": RewardTermCfg(
            func=envs_mdp.is_terminated, weight=-10.0 / 0.02,
        ),
        # Penalties
        "lin_vel_z_l2": RewardTermCfg(func=rwd.lin_vel_z_l2, weight=-2.0),
        "ang_vel_xy_l2": RewardTermCfg(func=rwd.ang_vel_xy_l2, weight=-0.05),
        "joint_vel_l2": RewardTermCfg(func=rwd.joint_vel_l2, weight=-0.001),
        "joint_acc_l2": RewardTermCfg(func=rwd.joint_acc_l2, weight=-2.5e-7),
        "action_rate_l2": RewardTermCfg(func=rwd.action_rate_l2, weight=-0.05),
        "joint_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits, weight=-5.0,
        ),
        "energy": RewardTermCfg(func=rwd.energy, weight=-2e-5),
        "joint_deviation_arms": RewardTermCfg(
            func=rwd.joint_deviation_arms, weight=-0.1,
        ),
        "joint_deviation_waist": RewardTermCfg(
            func=rwd.joint_deviation_waist, weight=-1.0,
        ),
        "joint_deviation_legs": RewardTermCfg(
            func=rwd.joint_deviation_legs, weight=-1.0,
        ),
        "flat_orientation_l2": RewardTermCfg(
            func=rwd.flat_orientation_l2, weight=-5.0,
        ),
        "base_height_l2": RewardTermCfg(
            func=rwd.base_height_l2, weight=-10.0,
            params={"target_height": 0.78},
        ),
        "feet_gait": RewardTermCfg(
            func=rwd.feet_gait, weight=0.5,
            params={"period": 0.8, "offset": (0.0, 0.5), "threshold": 0.5},
        ),
        "feet_slide": RewardTermCfg(func=rwd.feet_slide, weight=-0.2),
        "foot_clearance_reward": RewardTermCfg(
            func=rwd.foot_clearance_reward, weight=1.0,
            params={"target_height": 0.1},
        ),
        "undesired_contacts": RewardTermCfg(
            func=rwd.undesired_contacts, weight=-1.0,
        ),
        "dof_torques_l2": RewardTermCfg(func=rwd.dof_torques_l2, weight=-1.5e-7),
        "dof_torques_limits": RewardTermCfg(
            func=rwd.dof_torques_limits, weight=-0.01,
        ),
        "feet_air_time": RewardTermCfg(
            func=rwd.feet_air_time, weight=0.25,
            params={"threshold": 0.6},
        ),
        "feet_air_time_variance": RewardTermCfg(
            func=rwd.feet_air_time_variance, weight=-0.1,
        ),
        "feet_stumble": RewardTermCfg(func=rwd.feet_stumble, weight=-1.0),
        "feet_too_near": RewardTermCfg(
            func=rwd.feet_too_near, weight=-1.0,
            params={"threshold": 0.2},
        ),
        "joint_coordination": RewardTermCfg(
            func=rwd.joint_coordination, weight=-0.1,
        ),
        "stand_still": RewardTermCfg(
            func=rwd.stand_still, weight=-0.1,
            params={"command_name": "goal", "threshold": 0.1},
        ),
        "feet_height_body": RewardTermCfg(
            func=rwd.feet_height_body, weight=1.0,
            params={"target_height": 0.1, "tanh_mult": 2.0},
        ),
    }

    ##
    # Terminations — Sec IV-D2
    ##
    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=envs_mdp.bad_orientation,
            params={"limit_angle": 0.8},
        ),
        "base_collision": TerminationTermCfg(
            func=term.base_collision,
            params={"force_threshold_factor": 1.0},
        ),
    }

    ##
    # Curriculum (Sec IV-D3, Appendix A)
    ##
    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=terrain_levels_vel,
            params={"command_name": "goal"},
        ),
    }

    ##
    # Metrics
    ##
    metrics = {
        "mean_action_acc": MetricsTermCfg(func=envs_mdp.mean_action_acc),
    }

    # Play overrides
    if play:
        observations["actor"].enable_corruption = False
        events.pop("push_robot", None)
        curriculum = {}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": get_g1_robot_cfg()},
            sensors=(terrain_scan, elev_map_sensor, feet_ground_cfg, body_contact_cfg),
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(ROUGH_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            num_envs=1 if play else 2048,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=1500,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=20.0,
    )
