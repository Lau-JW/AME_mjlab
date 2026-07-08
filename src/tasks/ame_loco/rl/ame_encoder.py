"""AME-2: Attention-based Neural Map Encoder.

Paper: AME-2 (arXiv:2601.08485)
Section IV-A, Figure 3 (Left Bottom)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AME2Encoder(nn.Module):
    """AME-2 attention-based map encoder.

    Architecture (paper Fig 3 left-bottom):
      1. CNN extracts local features from elevation map
      2. MLP + MaxPool produces global features
      3. Global features + proprioception embedding → query vector (via MLP)
      4. Multi-Head Attention: query attends to local features
      5. Output: global features ∥ weighted local features

    Args:
        map_channels: Input channels of elevation map (3: teacher xyz, 4: student xyzu)
        map_height: Map grid height (L)
        map_width: Map grid width (W)
        local_feat_dim: Dimension of pointwise local features
        global_feat_dim: Dimension of global context features
        proprio_dim: Dimension of proprioception embedding
        num_heads: Number of attention heads
        cnn_channels: CNN hidden channels
    """

    def __init__(
        self,
        map_channels: int = 3,
        map_height: int = 36,
        map_width: int = 14,
        local_feat_dim: int = 64,
        global_feat_dim: int = 64,
        proprio_dim: int = 64,
        num_heads: int = 4,
        cnn_channels: int = 32,
    ):
        super().__init__()

        # --- CNN for local features ---
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
        )

        # Positional embedding (MLP per point)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )

        # Fuse CNN features + positional embedding → pointwise local features
        self.local_fusion = nn.Sequential(
            nn.Linear(cnn_channels + 32, local_feat_dim),
            nn.ELU(),
            nn.Linear(local_feat_dim, local_feat_dim),
            nn.ELU(),
        )

        # Global feature extractor (MLP + MaxPool over points)
        self.global_mlp = nn.Sequential(
            nn.Linear(local_feat_dim, global_feat_dim),
            nn.ELU(),
            nn.Linear(global_feat_dim, global_feat_dim),
            nn.ELU(),
        )

        # Query generation: [global_feat ∥ proprio] → query
        self.query_mlp = nn.Sequential(
            nn.Linear(global_feat_dim + proprio_dim, global_feat_dim),
            nn.ELU(),
            nn.Linear(global_feat_dim, global_feat_dim),
        )

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=local_feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(local_feat_dim, local_feat_dim)

    def forward(
        self, elevation_map: torch.Tensor, proprio_embed: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            elevation_map: (B, C, H, W) — C=3 (teacher xyz) or 4 (student xyzu)
            proprio_embed: (B, D) — proprioception embedding

        Returns:
            map_embedding: (B, local_feat_dim + global_feat_dim)
        """
        B = elevation_map.shape[0]

        # 1. CNN local features: (B, C, H, W) → (B, cnn_ch, H, W)
        cnn_feat = self.cnn(elevation_map)

        # 2. Create grid of (x, y) coordinates normalized to [-1, 1]
        H, W = elevation_map.shape[-2:]
        xs = torch.linspace(-1, 1, W, device=elevation_map.device)
        ys = torch.linspace(-1, 1, H, device=elevation_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
        pos_embed = self.pos_mlp(pos_grid)  # (H, W, 32)

        # 3. Fuse CNN + positional → pointwise local features
        # cnn_feat: (B, C, H, W) → (B, H, W, C) + pos_embed: (H, W, 32)
        cnn_feat = cnn_feat.permute(0, 2, 3, 1)  # (B, H, W, cnn_ch)
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 32)
        fused = torch.cat([cnn_feat, pos_embed], dim=-1)  # (B, H, W, cnn_ch+32)
        pointwise_features = self.local_fusion(fused)  # (B, H, W, local_feat_dim)

        # Flatten spatial dims: (B, H*W, local_feat_dim)
        N = H * W
        local_features = pointwise_features.view(B, N, -1)

        # 4. Global features via MLP + MaxPool
        global_feat = self.global_mlp(local_features)  # (B, N, global_feat_dim)
        global_feat_pooled = global_feat.max(dim=1)[0]  # (B, global_feat_dim)

        # 5. Generate query from global + proprio
        query_input = torch.cat([global_feat_pooled, proprio_embed], dim=-1)  # (B, global_feat_dim + D)
        query = self.query_mlp(query_input).unsqueeze(1)  # (B, 1, global_feat_dim)

        # 6. Multi-Head Attention: query → weighted local features
        # Project query to match local_feat_dim if needed
        if global_feat_dim != local_features.shape[-1]:
            query_proj = nn.Linear(global_feat_dim, local_features.shape[-1], device=query.device)(query)
        else:
            query_proj = query

        attn_out, attn_weights = self.mha(
            query=query_proj,
            key=local_features,
            value=local_features,
        )  # attn_out: (B, 1, local_feat_dim)

        weighted_local = attn_out.squeeze(1)  # (B, local_feat_dim)

        # 7. Concatenate global + weighted local → map embedding
        map_embedding = torch.cat([global_feat_pooled, weighted_local], dim=-1)

        return map_embedding


class ProprioEncoder(nn.Module):
    """Proprioception encoder (teacher: plain MLP).

    Paper Sec IV-A, Fig 3 (Right).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class ProprioEncoderStudent(nn.Module):
    """Proprioception encoder (student: LSIO + MLP).

    The student stacks 20 steps of past proprioception (excluding base lin vel
    and commands) and processes them with LSIO (a GRU-based temporal encoder).
    Commands are concatenated after the temporal embedding.
    """

    def __init__(
        self,
        obs_dim: int,
        cmd_dim: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 64,
        history_len: int = 20,
    ):
        super().__init__()
        self.history_len = history_len
        self.cmd_dim = cmd_dim

        # LSIO — GRU-based temporal encoder
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim + cmd_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs_history: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs_history: (B, T, obs_dim) — T=20 steps of proprioception
            commands: (B, cmd_dim) — velocity/goal commands

        Returns:
            proprio_embed: (B, output_dim)
        """
        _, hidden = self.gru(obs_history)  # hidden: (1, B, hidden_dim)
        temporal_embed = hidden.squeeze(0)  # (B, hidden_dim)
        combined = torch.cat([temporal_embed, commands], dim=-1)
        return self.output_mlp(combined)


class AMEPolicy(nn.Module):
    """Full AME-2 policy network (teacher version).

    Architecture (paper Fig 3 Left-Top):
      proprioception → ProprioEncoder → proprio_embed
      elevation_map  → AME2Encoder    → map_embed
      [proprio_embed ∥ map_embed] → MLP decoder → actions
    """

    def __init__(
        self,
        proprio_dim: int,
        map_channels: int = 3,
        map_height: int = 36,
        map_width: int = 14,
        action_dim: int = 29,
        proprio_hidden: int = 128,
        encoder_proprio_dim: int = 64,
        local_feat_dim: int = 64,
        global_feat_dim: int = 64,
        decoder_hidden: tuple = (512, 256, 128),
    ):
        super().__init__()

        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim,
            hidden_dim=proprio_hidden,
            output_dim=encoder_proprio_dim,
        )

        self.map_encoder = AME2Encoder(
            map_channels=map_channels,
            map_height=map_height,
            map_width=map_width,
            local_feat_dim=local_feat_dim,
            global_feat_dim=global_feat_dim,
            proprio_dim=encoder_proprio_dim,
        )

        decoder_input_dim = encoder_proprio_dim + global_feat_dim + local_feat_dim

        decoder_layers = []
        prev_dim = decoder_input_dim
        for h in decoder_hidden:
            decoder_layers.extend([nn.Linear(prev_dim, h), nn.ELU()])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, action_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self, proprio: torch.Tensor, elevation_map: torch.Tensor
    ) -> torch.Tensor:
        proprio_embed = self.proprio_encoder(proprio)
        map_embed = self.map_encoder(elevation_map, proprio_embed)
        combined = torch.cat([proprio_embed, map_embed], dim=-1)
        return self.decoder(combined)


class MoECritic(nn.Module):
    """Mixture-of-Experts critic network.

    Paper Sec IV-B, ref [68] (GMT). Uses multiple expert MLPs and a gating
    network to combine their outputs.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_hidden: int = 256,
        output_dim: int = 1,
    ):
        super().__init__()
        self.num_experts = num_experts

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, num_experts),
        )

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ELU(),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ELU(),
                nn.Linear(expert_hidden, output_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = F.softmax(self.gate(x), dim=-1)  # (B, num_experts)
        expert_outputs = torch.stack(
            [e(x) for e in self.experts], dim=-1
        )  # (B, 1, num_experts)
        output = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=-1)
        return output
