import unittest
from types import SimpleNamespace

import torch

from src.tasks.ame_loco.mdp.command import (
    TerrainLevelGoal,
    UniformGoalCommand,
    UniformGoalCommandCfg,
)


class _Scene(dict):
    terrain = None


class _Terrain:
    def __init__(self, levels: list[int]):
        self.terrain_levels = torch.tensor(levels, dtype=torch.long)

    def update_env_origins(
        self,
        env_ids: torch.Tensor,
        move_up: torch.Tensor,
        move_down: torch.Tensor,
    ) -> None:
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels.clamp_(min=0, max=9)


def _make_goal(num_envs: int) -> tuple[SimpleNamespace, UniformGoalCommand]:
    qpos = torch.zeros(num_envs, 7)
    qpos[:, 3] = 1.0
    data = SimpleNamespace(
        data=SimpleNamespace(qpos=qpos),
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
        root_link_pos_w=torch.zeros(num_envs, 3),
        heading_w=torch.zeros(num_envs),
    )
    scene = _Scene(robot=SimpleNamespace(data=data))
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        step_dt=0.02,
        max_episode_length=1000,
        episode_length_buf=torch.zeros(num_envs),
    )
    cfg = UniformGoalCommandCfg(
        entity_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
    )
    return env, UniformGoalCommand(cfg, env)


class GoalCommandTest(unittest.TestCase):
    def test_sampled_goal_distance_uses_configured_range(self) -> None:
        torch.manual_seed(7)
        env, goal = _make_goal(256)
        env_ids = torch.arange(env.num_envs)

        goal._resample_command(env_ids)

        base_xy = goal.robot.data.data.qpos[:, :2]
        distance = torch.norm(goal.goal_pos_w - base_xy, dim=-1)
        self.assertGreaterEqual(distance.min().item(), 1.0)
        self.assertLessEqual(distance.max().item(), 5.0)

    def test_curriculum_uses_success_ema_and_skips_standing_goals(self) -> None:
        env, goal = _make_goal(3)
        env.scene.terrain = _Terrain([2, 2, 2])
        env.command_manager = SimpleNamespace(get_term=lambda _: goal)
        goal.goal_pos_w[:] = torch.tensor(
            [[0.1, 0.0], [5.0, 0.0], [0.1, 0.0]]
        )
        goal.has_active_goal[:] = True
        goal.is_standing_env[2] = True

        curriculum = TerrainLevelGoal(None, env)
        env_ids = torch.arange(env.num_envs)
        for _ in range(7):
            goal.goal_reached[:] = False
            curriculum(
                env,
                env_ids,
                command_name="goal",
                ema_alpha=0.1,
                promotion_threshold=0.5,
                demotion_distance=4.0,
            )

        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [3, 0, 2])
        self.assertAlmostEqual(curriculum.success_ema[0].item(), 0.5217031)
        self.assertEqual(curriculum.success_ema[1].item(), 0.0)
        self.assertEqual(curriculum.success_ema[2].item(), 0.0)

        restored = TerrainLevelGoal(None, env)
        restored.load_state_dict(curriculum.state_dict())
        self.assertTrue(torch.allclose(restored.success_ema, curriculum.success_ema))


if __name__ == "__main__":
    unittest.main()
