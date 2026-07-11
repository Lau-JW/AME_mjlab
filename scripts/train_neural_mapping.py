"""Train the AME-2 neural local mapping model (Section V-B).

This is a standalone pre-training script that learns to predict local elevation
maps and uncertainties from noisy/partial depth clouds. The trained model can be
loaded during student policy training to provide 4-channel ego maps.

Paper:
- Loss: beta-NLL (beta=0.5) with per-sample total-variation reweighting.
- Data: GT elevation maps are degraded with noise, dropout, outliers, and
  simulated occlusions to mimic real depth-sensor observations.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim

import src.tasks  # noqa: F401, register tasks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.ame_loco.mdp.map import sample_gt_elevation_map, simulate_depth_cloud_from_gt
from src.tasks.ame_loco.mdp.neural_mapping import (
    LocalMapPredictor,
    NeuralMapCfg,
    project_depth_to_local_grid,
    tv_reweight_loss,
)


def generate_local_grid_batch(env, cfg: NeuralMapCfg, batch_size: int, device: str):
    """Sample GT local maps, simulate depth clouds, and return (input, target)."""
    gt_map = sample_gt_elevation_map(
        env,
        map_height=cfg.ego_grid_size[0],
        map_width=cfg.ego_grid_size[1],
        resolution=cfg.ego_resolution,
        center_x=cfg.ego_center[0],
        center_y=cfg.ego_center[1],
    )

    B = env.num_envs
    # Repeat/envs may already be batched; take a subset if needed.
    if B < batch_size:
        raise ValueError(f"num_envs ({B}) must be >= batch_size ({batch_size})")

    gt_map = gt_map[:batch_size]

    points, _ = simulate_depth_cloud_from_gt(
        gt_map,
        center_x=cfg.ego_center[0],
        center_y=cfg.ego_center[1],
        fov_deg=80.0,
        drop_ratio=0.1,
        outlier_ratio=0.02,
        noise_std=0.03,
    )

    local_grid = project_depth_to_local_grid(
        points,
        grid_size=cfg.local_grid_size,
        resolution=cfg.local_resolution,
        center=cfg.local_center,
    )

    # Target is the z channel of the GT ego map, with same local grid center/resolution.
    # For this simplified pre-training we train the predictor directly on the local grid
    # target: project the GT elevation map to the local frame and take the z channel.
    target_z = gt_map[:batch_size, 2:3, :, :]
    # Interpolate target to local grid size if different (simplified).
    if target_z.shape[-2:] != local_grid.shape[-2:]:
        target_z = torch.nn.functional.interpolate(
            target_z, size=local_grid.shape[-2:], mode="bilinear", align_corners=False
        )

    return local_grid, target_z


def main():
    parser = argparse.ArgumentParser(description="Train AME-2 neural mapping model.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--log-root", type=str, default="logs/neural_mapping")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    env_cfg = load_env_cfg("Unitree-G1-AME-Teacher")
    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()

    cfg = NeuralMapCfg(
        local_grid_size=(31, 31),
        local_resolution=0.04,
        local_center=(0.6, 0.0),
        ego_grid_size=(18, 13),
        ego_resolution=0.08,
        ego_center=(0.32, 0.0),
        device=args.device,
    )

    model = LocalMapPredictor().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    start_batch = 0

    if args.resume is not None:
        print(f"[AME] Resuming neural mapping from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        key = "predictor" if "predictor" in ckpt else "model"
        model.load_state_dict(ckpt[key])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_batch = ckpt.get("batch", 0)

    log_dir = Path(args.log_root)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AME] Training neural mapping on {args.device} with batch_size={args.batch_size}")
    for batch_idx in range(start_batch, args.num_batches):
        env.step(torch.zeros(args.num_envs, env.num_actions, device=args.device))
        input_grid, target_z = generate_local_grid_batch(env, cfg, args.batch_size, args.device)

        pred = model(input_grid)
        loss = tv_reweight_loss(pred["elevation"], pred["log_var"], target_z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0:
            print(f"[AME] Batch {batch_idx + 1}/{args.num_batches}: loss={loss.item():.6f}")

        if (batch_idx + 1) % args.save_interval == 0:
            save_path = log_dir / f"mapping_model_{batch_idx + 1}.pt"
            torch.save(
                {
                    "predictor": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "batch": batch_idx + 1,
                    "cfg": cfg,
                },
                save_path,
            )
            print(f"[AME] Saved mapping model to {save_path}")

    final_path = log_dir / "mapping_model_final.pt"
    torch.save(
        {
            "predictor": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "batch": args.num_batches,
            "cfg": cfg,
        },
        final_path,
    )
    print(f"[AME] Final mapping model saved to {final_path}")


if __name__ == "__main__":
    main()
