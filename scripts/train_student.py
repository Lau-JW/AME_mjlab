"""Train AME-2 student policy (Phase-1).

Paper Sec IV-C / IV-E1:
  - Online student rollouts with 4ch map (GT xyz + heuristic uncertainty)
  - LSIO proprio history (20 steps, no base lin-vel)
  - PPO + action distillation + map-embedding representation loss
  - PPO surrogate disabled for the first 5000 iterations

Full neural mapping pipeline is still TODO; this stage unlocks the control-side
student training loop against a frozen teacher checkpoint.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def _disable_cuda_graphs_for_compat() -> None:
    from mjlab.sim.sim import Simulation

    Simulation._should_use_cuda_graph = lambda self: False


def run_train(
    task_id: str,
    log_dir: Path,
    teacher_checkpoint: Path,
    device: str = "cuda:0",
    num_envs: int | None = None,
    max_iterations: int | None = None,
    resume: Path | None = None,
    enable_cuda_graph: bool = False,
    learning_rate: float | None = None,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    if not enable_cuda_graph:
        _disable_cuda_graphs_for_compat()

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    if max_iterations is not None:
        rl_cfg.max_iterations = max_iterations
    if learning_rate is not None:
        rl_cfg.algorithm.learning_rate = learning_rate

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(
        env,
        asdict(rl_cfg),
        str(log_dir),
        device,
        teacher_checkpoint=str(teacher_checkpoint),
    )
    if resume is not None:
        runner.load(str(resume), map_location=device)
        lr = float(rl_cfg.algorithm.learning_rate)
        runner.alg.learning_rate = lr
        for param_group in runner.alg.optimizer.param_groups:
            param_group["lr"] = lr
        print(f"[AME] Reset optimizer LR after resume -> {lr:g}")

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

    start_iteration = runner.current_learning_iteration
    remaining = max(0, rl_cfg.max_iterations - start_iteration)
    print(f"[AME] Student training started: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    print(f"[AME] Teacher: {teacher_checkpoint}")
    print(f"[AME] Device: {device}")
    print(f"[AME] Num envs: {env_cfg.scene.num_envs}")
    print(f"[AME] Target iteration: {rl_cfg.max_iterations}")
    print(f"[AME] Start iteration: {start_iteration}")
    print(f"[AME] Remaining: {remaining}")
    print(f"[AME] Surrogate disabled for first "
          f"{getattr(runner, 'cfg', {}).get('surrogate_disable_iters', 5000)} iters")
    if remaining == 0:
        print("[AME] Target already reached; nothing to train.")
        env.close()
        return

    runner.learn(
        num_learning_iterations=remaining,
        init_at_random_ep_len=resume is None,
    )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AME-2 student policy (Phase-1).")
    parser.add_argument("--task-id", default="Unitree-G1-AME-Student")
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    args = parser.parse_args()

    if not args.teacher_checkpoint.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_checkpoint}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = args.log_root / "g1_ame_student" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    run_train(
        task_id=args.task_id,
        log_dir=log_dir,
        teacher_checkpoint=args.teacher_checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        max_iterations=args.max_iterations,
        resume=args.resume,
        enable_cuda_graph=args.enable_cuda_graph,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
