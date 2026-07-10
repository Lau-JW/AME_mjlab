"""AME-2 termination conditions.

Paper: Section IV-D2
- Bad orientation: |g_x| > 0.985, |g_y| > 0.7 (fall), g_z > 0.0 (flipped)
- Base collision: base contact force > robot weight
- High thigh acceleration: (future)
- Stagnation: (future)
"""

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_orientation_ame(
    env: "ManagerBasedRlEnv",
    limit_gx: float = 0.985,
    limit_gy: float = 0.7,
    limit_gz: float = 0.0,
) -> torch.Tensor:
    """Check bad orientation (Sec IV-D2).

    Uses projected gravity vector in base frame (from IMU).
    - |g_x| > 0.985: large roll (near 90 deg)
    - |g_y| > 0.7: large pitch
    - g_z > 0.0: robot flipped (gravity pointing up in base frame)
    """
    try:
        grav = _get_projected_gravity(env)
        if grav is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        g_x = torch.abs(grav[:, 0])
        g_y = torch.abs(grav[:, 1])
        g_z = grav[:, 2]
        bad = (g_x > limit_gx) | (g_y > limit_gy) | (g_z > limit_gz)
        return bad
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def base_collision(
    env: "ManagerBasedRlEnv",
    force_threshold_factor: float = 1.0,
    sensor_name: str = "base_contact",
) -> torch.Tensor:
    """Terminate if pelvis/torso contact force exceeds robot weight * factor."""
    try:
        sensor = env.scene[sensor_name]
        force = sensor.data.force
        if force is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        base_force = torch.norm(force, dim=-1).amax(dim=1)
        robot_mass = 35.0
        weight = robot_mass * 9.81
        return base_force > weight * force_threshold_factor
    except (AttributeError, KeyError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _get_projected_gravity(env):
    """Get projected gravity vector in base frame."""
    try:
        # Try mjlab's builtin projected_gravity observation function
        from mjlab.envs.mdp import projected_gravity
        return projected_gravity(env)
    except Exception:
        pass
    return None
