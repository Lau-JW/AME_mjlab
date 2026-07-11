"""Train AME-2 student policy with neural mapping pipeline.

Paper: Section IV-C, Section IV-E1
- Student: 40000 iterations
- Uses learned mapping pipeline (depth -> elevation map)
- PPO + action distillation + representation loss
- Surrogate loss disabled first 5000 iterations
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401, register tasks

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv


def load_teacher_actor(teacher_checkpoint: str, device: str):
    """Load a trained teacher AME-2 actor from a checkpoint.

    The teacher uses the same AMEOnPolicyRunner architecture, so we instantiate
    a teacher environment and runner, load the checkpoint, and return the actor.
    """
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    teacher_env_cfg = load_env_cfg("Unitree-G1-AME-Teacher")
    teacher_rl_cfg = load_rl_cfg("Unitree-G1-AME-Teacher")
    runner_cls = load_runner_cls("Unitree-G1-AME-Teacher")

    teacher_env = ManagerBasedRlEnv(cfg=teacher_env_cfg, device=device)
    teacher_env = RslRlVecEnvWrapper(teacher_env, clip_actions=teacher_rl_cfg.clip_actions)

    # Use a temporary log directory since we only need to load the model.
    tmp_log_dir = Path("/tmp/ame_teacher_load")
    tmp_log_dir.mkdir(parents=True, exist_ok=True)
    teacher_runner = runner_cls(teacher_env, teacher_rl_cfg, str(tmp_log_dir), device=device)

    print(f"[AME] Loading teacher checkpoint: {teacher_checkpoint}")
    teacher_runner.load(teacher_checkpoint)
    teacher_actor = teacher_runner.alg.policy.actor

    # Clean up the temporary teacher runner to save memory.
    del teacher_runner
    del teacher_env

    return teacher_actor


def main():
    parser = argparse.ArgumentParser(description="Train AME-2 student policy.")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher checkpoint .pt file")
    parser.add_argument("--mapping-checkpoint", type=str, default=None, help="Optional pretrained neural mapping model .pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--log-root", type=str, default="logs/rsl_rl")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    task_id = "Unitree-G1-AME-Student"

    env_cfg = load_env_cfg(task_id)
    rl_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    if args.mapping_checkpoint is not None:
        env.student_mapping_checkpoint = args.mapping_checkpoint
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    log_dir = Path(args.log_root) / rl_cfg.experiment_name / "student"
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = runner_cls(env, rl_cfg, str(log_dir), device=args.device)

    # Load teacher model and attach it to the StudentPPO algorithm.
    teacher_actor = load_teacher_actor(args.teacher, args.device)
    runner.alg.set_teacher(teacher_actor)

    # Optionally resume student training.
    if args.resume is not None:
        print(f"[AME] Resuming student training from: {args.resume}")
        runner.load(args.resume)
        runner.alg.current_iteration = runner.current_learning_iteration

    print(f"[AME] Training student: {task_id}")
    print(f"[AME] Log dir: {log_dir}")
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
