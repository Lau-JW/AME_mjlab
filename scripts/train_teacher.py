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
    resume: Path | None = None,
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
    if resume is not None:
        runner.load(str(resume), map_location=device)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

    start_iteration = runner.current_learning_iteration
    remaining_iterations = max(0, rl_cfg.max_iterations - start_iteration)

    print(f"[AME] Teacher training started: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    print(f"[AME] Device: {device}")
    print(f"[AME] Num envs: {env_cfg.scene.num_envs}")
    print(f"[AME] Target iteration: {rl_cfg.max_iterations}")
    print(f"[AME] Start iteration: {start_iteration}")
    print(f"[AME] Remaining iterations: {remaining_iterations}")
    if resume is not None:
        print(f"[AME] Resumed from: {resume}")
    if remaining_iterations == 0:
        print("[AME] Target iteration already reached; nothing to train.")
        env.close()
        return

    runner.learn(
        num_learning_iterations=remaining_iterations,
        init_at_random_ep_len=resume is None,
    )
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train AME-2 teacher policy.")
    parser.add_argument("--task-id", default="Unitree-G1-AME-Teacher")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-root", type=Path, default=Path("logs") / "rsl_rl")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path. Terrain curriculum and EMA are restored when available.",
    )
    args = parser.parse_args()
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Checkpoint does not exist: {args.resume}")

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
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
