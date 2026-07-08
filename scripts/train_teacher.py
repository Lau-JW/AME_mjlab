"""Train AME-2 teacher policy (Sec IV-E1, Table VI).

Teacher: 80000 iterations, ground-truth elevation maps, PPO + MoE critic.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401 — register tasks

from dataclasses import asdict
from datetime import datetime

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml


def run_train(
    task_id: str,
    log_dir: Path,
    device: str = "cuda:0",
    num_envs: int | None = None,
    max_iterations: int | None = None,
):
    os.environ.setdefault("MUJOCO_GL", "egl")

    configure_torch_backends()

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)

    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    if max_iterations is not None:
        rl_cfg.max_iterations = max_iterations

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), str(log_dir), device)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

    print(f"[AME] Teacher training started: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    print(f"[AME] Device: {device}")
    print(f"[AME] Num envs: {env_cfg.scene.num_envs}")
    print(f"[AME] Max iterations: {rl_cfg.max_iterations}")
    runner.learn(
        num_learning_iterations=rl_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train AME-2 teacher policy.")
    parser.add_argument("--task-id", default="Unitree-G1-AME-Teacher")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-root", type=Path, default=Path("logs") / "rsl_rl")
    args = parser.parse_args()

    log_dir = (
        args.log_root
        / "g1_ame_teacher"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    run_train(
        task_id=args.task_id,
        log_dir=log_dir,
        device=args.device,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
