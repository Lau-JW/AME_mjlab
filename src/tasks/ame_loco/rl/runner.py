"""AME-2 runner with MoE critic (Sec IV-B).

- Actor: standard ActorCritic MLP (replaced by AME-2 encoder at policy level)
- Critic: MoE (Mixture-of-Experts) from [68]
- Asymmetric: critic gets extra inputs (base_lin_vel + body_contact)
"""

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from src.tasks.ame_loco.rl.ame_encoder import MoECritic


class AMEOnPolicyRunner(MjlabOnPolicyRunner):
    """On-policy runner with MoE critic."""

    env: RslRlVecEnvWrapper

    def __init__(self, env, cfg, log_dir, device="cuda:0", **kwargs):
        super().__init__(env, cfg, log_dir, device, **kwargs)

    def _setup_algorithm(self):
        """Set up algorithm and replace critic with MoE if configured."""
        super()._setup_algorithm()

        use_moe = self.cfg.get("use_moe_critic", False)
        if not use_moe:
            return

        # Get critic observation dimension
        if hasattr(self.env, "num_critic_obs"):
            critic_dim = self.env.num_critic_obs
        else:
            critic_dim = self.env.num_obs

        num_experts = self.cfg.get("moe_num_experts", 8)
        expert_hidden = self.cfg.get("moe_expert_hidden", 256)
        print(f"[AME] Replacing critic with MoE: {num_experts} experts, "
              f"input_dim={critic_dim}, expert_hidden={expert_hidden}")

        # Build MoE critic
        moe = MoECritic(
            input_dim=critic_dim,
            num_experts=num_experts,
            expert_hidden=expert_hidden,
        ).to(self.device)

        # Replace the MLP critic in the ActorCritic module
        if hasattr(self.alg, "policy") and hasattr(self.alg.policy, "critic"):
            self.alg.policy.critic = moe
            print(f"[AME] MoE critic installed: {sum(p.numel() for p in moe.parameters())} params")

    def save(self, path: str, infos=None) -> None:
        """Save checkpoint, skipping logger.save_model when logger isn't available."""
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        # MjlabOnPolicyRunner.save() calls self.logger.save_model, but our
        # rsl_rl fork doesn't set up self.logger. Skip if not available.
        if hasattr(self, "logger") and self.cfg.get("upload_model", False):
            self.logger.save_model(path, self.current_learning_iteration)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Override: call parent log, then add per-terrain-type curriculum levels."""
        # Parent log handles all standard logging
        super().log(locs, width, pad)

        # Add per-terrain-type curriculum levels
        try:
            terrain = self.env.unwrapped.scene.terrain
            types = terrain.terrain_types
            levels = terrain.terrain_levels.float()
            n_types = int(types.max().item()) + 1
            type_names = ["flat", "stairs", "stairs_inv", "slope",
                          "slope_inv", "rough", "wave"]

            for t in range(n_types):
                mask = types == t
                cnt = mask.sum().item()
                if cnt > 0:
                    mean_lv = levels[mask].mean().item()
                    name = type_names[t] if t < len(type_names) else f"type{t}"
                    self.writer.add_scalar(f"Curriculum/{name}_level", mean_lv, locs["it"])
                else:
                    name = type_names[t] if t < len(type_names) else f"type{t}"
                    self.writer.add_scalar(f"Curriculum/{name}_level", 0.0, locs["it"])
        except Exception:
            pass

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        return super().learn(num_learning_iterations, init_at_random_ep_len)
