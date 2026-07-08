"""AME-2 Actor-Critic with attention-based map encoder.

Paper: AME-2 (arXiv:2601.08485)
- Actor: AME-2 Encoder (CNN + MHA) + MLP decoder
- Critic: standard MLP (or MoE via runner override)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from src.tasks.ame_loco.rl.ame_encoder import (
    AME2Encoder, ProprioEncoder, MoECritic,
)


class AMEActorCritic(nn.Module):
    """Actor-Critic with AME-2 attention-based map encoder.

    Observation layout (1608 dim):
      [0..3)    base_ang_vel
      [3..6)    projected_gravity
      [6..9)    command
      [9..38)   joint_pos (29)
      [38..67)  joint_vel (29)
      [67..96)  actions (29)
      [96..1608) elevation_map (3x36x14 = 1512)

    Actor: proprio(96) → ProprioEncoder → proprio_embed
           elevation_map(3,36,14) → AME2Encoder → map_embed
           [proprio_embed ∥ map_embed] → MLP decoder → actions

    Critic: standard MLP over full observation
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # AME-2 specific
        proprio_dim: int = 96,
        map_channels: int = 3,
        map_height: int = 36,
        map_width: int = 14,
        proprio_hidden: int = 128,
        encoder_proprio_dim: int = 64,
        local_feat_dim: int = 64,
        global_feat_dim: int = 64,
        **kwargs,
    ):
        if kwargs:
            print(f"[AME] ActorCritic extra kwargs (ignored): {list(kwargs.keys())}")

        super().__init__()

        self.num_actions = num_actions
        self.proprio_dim = proprio_dim
        self.map_channels = map_channels
        self.map_height = map_height
        self.map_width = map_width
        activation_fn = resolve_nn_activation(activation)

        # ── Proprioception Encoder ──
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim,
            hidden_dim=proprio_hidden,
            output_dim=encoder_proprio_dim,
        )

        # ── AME-2 Map Encoder ──
        self.map_encoder = AME2Encoder(
            map_channels=map_channels,
            map_height=map_height,
            map_width=map_width,
            local_feat_dim=local_feat_dim,
            global_feat_dim=global_feat_dim,
            proprio_dim=encoder_proprio_dim,
        )

        # ── MLP Decoder ──
        decoder_input_dim = encoder_proprio_dim + global_feat_dim + local_feat_dim
        decoder_layers = []
        prev_dim = decoder_input_dim
        for h in actor_hidden_dims:
            decoder_layers.extend([nn.Linear(prev_dim, h), activation_fn])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, num_actions))
        self.decoder = nn.Sequential(*decoder_layers)

        # ── Critic (standard MLP) ──
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims)):
            if i == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        # ── Action noise ──
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown std type: {noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def _split_obs(self, obs: torch.Tensor):
        """Split observation into proprioception and elevation map."""
        proprio = obs[:, :self.proprio_dim]
        map_flat = obs[:, self.proprio_dim:]
        # Reshape flat map to (B, C, H, W)
        elevation_map = map_flat.view(
            -1, self.map_channels, self.map_height, self.map_width
        )
        return proprio, elevation_map

    # ── Actor forward ──

    def forward_actor(self, observations):
        proprio, elevation_map = self._split_obs(observations)
        proprio_embed = self.proprio_encoder(proprio)
        map_embed = self.map_encoder(elevation_map, proprio_embed)
        combined = torch.cat([proprio_embed, map_embed], dim=-1)
        return self.decoder(combined)

    def update_distribution(self, observations):
        mean = self.forward_actor(observations)
        if self.noise_std_type == "scalar":
            std = torch.clamp_min(self.std, 1.0e-6).expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.clamp_min(torch.exp(self.log_std), 1.0e-6).expand_as(mean)
        else:
            raise ValueError(f"Unknown std type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self.forward_actor(observations)

    # ── Critic ──

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def reset(self, dones=None):
        pass
