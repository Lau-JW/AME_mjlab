"""AME-2 runner with AME-2 encoder actor + optional MoE critic.

- Actor: ProprioEncoder + AME2Encoder (CNN+MHA) + MLP decoder (paper Fig 3)
- Critic: standard MLP (can be replaced with MoE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from rsl_rl.runners.on_policy_runner import _unpack_obs
from src.tasks.ame_loco.rl.ame_encoder import (
    ProprioEncoder, AME2Encoder, SimpleMapEncoder, MoECritic,
)


class AME2Actor(nn.Module):
    """AME-2 actor: proprio → encoder + map → AME2 → concat → MLP decoder.

    Wraps the full pipeline so it can replace ActorCritic.actor.
    Input: observation tensor (B, 1608) — proprio(96) + elevation_map(1512)
    Output: action mean (B, 29)
    """

    def __init__(self, num_actions=29, proprio_dim=96,
                 map_channels=3, map_height=18, map_width=7,
                 proprio_hidden=128, encoder_proprio_dim=64,
                 local_feat_dim=64, global_feat_dim=64,
                 decoder_hidden=(512, 256, 128)):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.map_channels = map_channels
        self.map_height = map_height
        self.map_width = map_width

        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim, hidden_dim=proprio_hidden,
            output_dim=encoder_proprio_dim,
        )
        self.map_encoder = AME2Encoder(
            map_channels=map_channels, map_height=map_height, map_width=map_width,
            local_feat_dim=local_feat_dim, global_feat_dim=global_feat_dim,
            proprio_dim=encoder_proprio_dim,
        )
        decoder_in = encoder_proprio_dim + global_feat_dim + local_feat_dim
        layers = []
        prev = decoder_in
        for h in decoder_hidden:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.decoder = nn.Sequential(*layers)

    def forward(self, obs):
        proprio = obs[:, :self.proprio_dim]
        map_flat = obs[:, self.proprio_dim:]
        elev_map = map_flat.view(-1, self.map_channels, self.map_height, self.map_width)
        prop_embed = self.proprio_encoder(proprio)
        map_embed = self.map_encoder(elev_map, prop_embed)
        combined = torch.cat([prop_embed, map_embed], dim=-1)
        return self.decoder(combined)


class AMEOnPolicyRunner(MjlabOnPolicyRunner):
    """On-policy runner with AME-2 encoder actor + optional MoE critic."""

    env: RslRlVecEnvWrapper

    def __init__(self, env, cfg, log_dir, device="cuda:0", **kwargs):
        super().__init__(env, cfg, log_dir, device, **kwargs)
        # Immediately replace actor after parent init completes
        if hasattr(self, "alg") and hasattr(self.alg, "policy"):
            self._install_ame2_actor()
        else:
            print("[AME] WARNING: alg/policy not available after init, skipping AME-2")

    def _install_ame2_actor(self):
        """Replace actor with AME-2 encoder, add SimpleMapEncoder for critic."""
        policy = self.alg.policy

        # Get actual observation dimensions from env
        obs, extras = _unpack_obs(self.env.get_observations())
        num_actor_obs = obs.shape[1]
        num_critic_obs = extras["observations"].get("critic", obs).shape[1]

        # Infer proprio vs map split: map is always 3 channels × grid
        # Grid dims from sensor: need to match the actual sensor output
        # For now: use total actor obs minus flat map size
        map_channels = 3
        # Compute map grid dims from the elevation map observation
        try:
            elev_term = self.env.env.cfg.observations["actor"].terms.get("elevation_map", None)
            if elev_term:
                map_h = elev_term.params.get("map_height", 18)
                map_w = elev_term.params.get("map_width", 13)
            else:
                map_h, map_w = 18, 13
        except Exception:
            map_h, map_w = 18, 13

        map_flat_dim = map_channels * map_h * map_w  # 3*18*7 = 378
        actor_proprio_dim = num_actor_obs - map_flat_dim
        critic_proprio_dim = num_critic_obs - map_flat_dim

        print(f"[AME] Obs dims: actor={num_actor_obs}, critic={num_critic_obs}, "
              f"proprio={actor_proprio_dim}/{critic_proprio_dim}, map={map_h}x{map_w}x{map_channels}")

        # ── Actor ──
        num_actions = self.env.num_actions
        ame_actor = AME2Actor(num_actions=num_actions,
                              proprio_dim=actor_proprio_dim,
                              map_channels=map_channels,
                              map_height=map_h, map_width=map_w).to(self.device)
        policy.actor = ame_actor
        params = sum(p.numel() for p in ame_actor.parameters())
        print(f"[AME] AME-2 encoder actor installed: {params} params")
        print(f"[AME] AME-2 architecture:\n{ame_actor}")

        # ── Critic: wrap with SimpleMapEncoder (CNN downsample, no MHA) ──
        # We insert a CNN encoder before the MLP to reduce map dimensionality.
        orig_critic = policy.critic
        use_moe = self.cfg.get("use_moe_critic", False)

        class CriticWithMapEncoder(nn.Module):
            def __init__(self, proprio_dim, map_c, map_h, map_w, orig_mlp, use_moe_critic=False):
                super().__init__()
                self.proprio_dim = proprio_dim
                self.map_c = map_c
                self.map_h = map_h
                self.map_w = map_w
                self.map_encoder = SimpleMapEncoder(
                    map_channels=map_c, map_height=map_h, map_width=map_w,
                    output_dim=64,
                )
                new_in = proprio_dim + 64
                if use_moe_critic:
                    self.mlp = MoECritic(input_dim=new_in, num_experts=8, expert_hidden=256)
                else:
                    new_first = nn.Linear(new_in, orig_mlp[0].out_features)
                    remaining = orig_mlp[1:]
                    self.mlp = nn.Sequential(new_first, *remaining)

            def forward(self, obs):
                proprio = obs[:, :self.proprio_dim]
                map_flat = obs[:, self.proprio_dim:]
                elev_map = map_flat.view(-1, self.map_c, self.map_h, self.map_w)
                map_feat = self.map_encoder(elev_map)
                combined = torch.cat([proprio, map_feat], dim=-1)
                return self.mlp(combined)

        policy.critic = CriticWithMapEncoder(
            critic_proprio_dim, map_channels, map_h, map_w, orig_critic, use_moe
        ).to(self.device)
        c_params = sum(p.numel() for p in policy.critic.parameters())
        critic_name = "MoE critic with SimpleMapEncoder" if use_moe else "Critic with SimpleMapEncoder"
        print(f"[AME] {critic_name} installed: {c_params} params")

        # Parent runner created the PPO optimizer before we replaced modules.
        # Rebuild it so AME actor/critic parameters are actually trained.
        self.alg.optimizer = optim.Adam(policy.parameters(), lr=self.alg.learning_rate)

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

    def _inject_terrain_curriculum(self, locs: dict):
        """Inject per-terrain curriculum levels into ep_infos so parent log prints them."""
        try:
            env = self.env.env if hasattr(self.env, 'env') else self.env
            terrain = env.scene.terrain
            col_types = terrain.terrain_types
            levels = terrain.terrain_levels.float()
            num_cols = terrain.terrain_origins.shape[1]

            # Build column→name mapping from terrain generator proportions (once)
            if not hasattr(self, '_terrain_col_names'):
                from src.tasks.ame_loco.mdp.terrain import SIMPLE_TERRAINS_CFG
                sub_cfgs = list(SIMPLE_TERRAINS_CFG.sub_terrains.values())
                sub_keys = list(SIMPLE_TERRAINS_CFG.sub_terrains.keys())
                props = np.array([s.proportion for s in sub_cfgs])
                props /= props.sum()
                cumsum = np.cumsum(props)
                self._terrain_col_names = {}
                for c in range(num_cols):
                    idx = np.min(np.where(c / num_cols + 0.001 < cumsum)[0])
                    self._terrain_col_names[c] = sub_keys[idx]

            # Group envs by sub-terrain name
            import collections
            groups = collections.defaultdict(list)
            for i in range(env.num_envs):
                name = self._terrain_col_names[col_types[i].item()]
                groups[name].append(levels[i].item())

            # Inject into ep_infos: add entries right after Curriculum/terrain_levels
            for ep_info in locs.get("ep_infos", []):
                # Build new dict with terrain entries inserted after Curriculum/
                new_ep = {}
                for k, v in ep_info.items():
                    new_ep[k] = v
                    if k == "Curriculum/terrain_levels":
                        for name in sorted(groups.keys()):
                            vals = groups[name]
                            if vals:
                                display = name.replace("_", " ").title()
                                new_ep[f"Curriculum/{display}"] = sum(vals) / len(vals)
                # Replace ep_info contents
                ep_info.clear()
                ep_info.update(new_ep)
        except Exception as e:
            print(f"[AME] terrain injection error: {type(e).__name__}: {e}")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Override: inject terrain data, then call parent log."""
        self._inject_terrain_curriculum(locs)
        super().log(locs, width, pad)

    def learn(self, num_learning_iterations, init_at_random_ep_len=True):
        return super().learn(num_learning_iterations, init_at_random_ep_len)
