"""Play/evaluate a trained AME-2 teacher policy."""

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.tasks  # noqa: F401 - register tasks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _select_viewer(requested: str) -> str:
    if requested != "auto":
        return requested
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _disable_cuda_graphs_for_compat() -> None:
    # Some mjlab/warp combinations disagree on where Warp exposes
    # driver_version. Play/video does not need CUDA graph capture, so disable it
    # to keep checkpoint playback robust across local environments.
    from mjlab.sim.sim import Simulation

    Simulation._should_use_cuda_graph = lambda self: False


def _run_headless(env: RslRlVecEnvWrapper, policy, steps: int) -> None:
    reward_sum = torch.zeros(env.num_envs, device=env.device)
    done_count = torch.zeros(env.num_envs, device=env.device)
    for _ in range(steps):
        with torch.no_grad():
            obs = env.get_observations()
            actions = policy(obs)
            _, rewards, dones, _ = env.step(actions)
        reward_sum += rewards
        done_count += dones.float()

    print("[AME] Headless rollout complete")
    print(f"[AME] Steps: {steps}")
    print(f"[AME] Mean reward sum: {reward_sum.mean().item():.3f}")
    print(f"[AME] Mean resets: {done_count.mean().item():.3f}")


def run_play(
    task_id: str,
    checkpoint: Path,
    device: str,
    num_envs: int | None,
    viewer: str,
    steps: int,
    no_terminations: bool,
    video: bool,
    video_length: int,
    video_height: int | None,
    video_width: int | None,
    enable_cuda_graph: bool,
) -> None:
    configure_torch_backends()
    if not enable_cuda_graph:
        _disable_cuda_graphs_for_compat()

    env_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    if no_terminations:
        env_cfg.terminations = {}
    if video_height is not None:
        env_cfg.viewer.height = video_height
    if video_width is not None:
        env_cfg.viewer.width = video_width

    render_mode = "rgb_array" if video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if video:
        video_folder = checkpoint.parent / "videos" / "play"
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[AME] Recording video to: {video_folder}")
    env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(rl_cfg), None, device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True, "critic": False},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    resolved_viewer = _select_viewer(viewer)
    print(f"[AME] Loaded checkpoint: {checkpoint}")
    print(f"[AME] Device: {device}")
    print(f"[AME] Num envs: {env.num_envs}")
    print(f"[AME] Viewer: {resolved_viewer}")
    if video and resolved_viewer != "headless":
        print("[AME] Video recording is active while viewer runs.")

    try:
        if resolved_viewer == "headless":
            _run_headless(env, policy, steps)
        elif resolved_viewer == "native":
            NativeMujocoViewer(env, policy).run()
        elif resolved_viewer == "viser":
            ViserPlayViewer(env, policy).run()
        else:
            raise ValueError(f"Unsupported viewer: {resolved_viewer}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play AME-2 teacher policy.")
    parser.add_argument("--task-id", default="Unitree-G1-AME-Teacher")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser", "headless"),
        default="auto",
        help="auto uses native when DISPLAY exists, otherwise Viser web viewer.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=200)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="Opt into CUDA graph capture during play. Disabled by default for compatibility.",
    )
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable terminations while viewing/debugging.",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    run_play(
        task_id=args.task_id,
        checkpoint=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        viewer=args.viewer,
        steps=args.steps,
        no_terminations=args.no_terminations,
        video=args.video,
        video_length=args.video_length,
        video_height=args.video_height,
        video_width=args.video_width,
        enable_cuda_graph=args.enable_cuda_graph,
    )


if __name__ == "__main__":
    main()
