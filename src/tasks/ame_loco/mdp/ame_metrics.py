"""AME-2 custom metrics for per-terrain-type tracking.

Each metric returns terrain_level for envs of that type, and 0 for others.
The runner takes mean across all completed episodes.
The logged value = true_mean_level × proportion_of_that_type.
"""

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _make_terrain_level_fn(type_idx: int):
    """Factory: create a metric function for a specific terrain type."""
    def fn(env):
        try:
            t = env.scene.terrain
            mask = t.terrain_types == type_idx
            # Return terrain_level for this type, -1 for others (ignored in logging)
            result = torch.where(mask, t.terrain_levels.float(),
                                 torch.full((env.num_envs,), -1.0, device=env.device))
            return result
        except Exception:
            return -torch.ones(env.num_envs, device=env.device)
    return fn


# Generate per-type functions (7 terrain types from mjlab ROUGH_TERRAINS_CFG)
terrain_level_flat                = _make_terrain_level_fn(0)
terrain_level_pyramid_stairs      = _make_terrain_level_fn(1)
terrain_level_pyramid_stairs_inv  = _make_terrain_level_fn(2)
terrain_level_hf_pyramid_slope    = _make_terrain_level_fn(3)
terrain_level_hf_pyramid_slope_inv = _make_terrain_level_fn(4)
terrain_level_random_rough        = _make_terrain_level_fn(5)
terrain_level_wave_terrain        = _make_terrain_level_fn(6)
