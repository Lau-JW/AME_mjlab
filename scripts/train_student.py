"""Train AME-2 student policy with neural mapping pipeline.

Paper: Section IV-C, Section IV-E1
- Student: 40000 iterations
- Uses learned mapping pipeline (depth → elevation map)
- PPO + action distillation + representation loss
- Surrogate loss disabled first 5000 iterations
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401, register tasks

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    task_id = "Unitree-G1-AME-Student"

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

    from mjlab.envs import ManagerBasedRlEnv
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    log_dir = Path("logs") / "rsl_rl" / rl_cfg.experiment_name / "student"
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = runner_cls(env, rl_cfg, str(log_dir), device="cuda:0")
    print(f"[AME] Training student: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    from pathlib import Path
    main()
