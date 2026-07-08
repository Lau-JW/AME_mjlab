"""Train AME-2 teacher policy (Sec IV-E1, Table VI).

Teacher: 80000 iterations, ground-truth elevation maps, PPO + MoE critic.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401 — register tasks

import logging
from dataclasses import asdict
from datetime import datetime

import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.os import dump_yaml


def run_train(task_id: str, log_dir: Path):
    os.environ["MUJOCO_GL"] = "egl"
    device = "cuda:0"

    configure_torch_backends()

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), str(log_dir), device)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

    print(f"[AME] Teacher training started: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    print(f"[AME] Max iterations: {rl_cfg.max_iterations}")
    runner.learn(
        num_learning_iterations=rl_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    env.close()


def main():
    task_id = "Unitree-G1-AME-Teacher"
    log_dir = (
        Path("logs")
        / "rsl_rl"
        / "g1_ame_teacher"
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    run_train(task_id, log_dir)


if __name__ == "__main__":
    main()
