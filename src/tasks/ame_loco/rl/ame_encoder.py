"""AME-2: Attention-based Neural Map Encoder + Simple Map Encoder for critic.

Paper: AME-2 (arXiv:2601.08485)
Section IV-A, Figure 3 (Left Bottom)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AME2Encoder(nn.Module):
    """AME-2 attention-based map encoder with CNN stride=2 downsampling.

    Architecture (paper Fig 3 left-bottom):
      1. CNN (stride=2) downsamples and extracts local features
      2. MLP + MaxPool produces global features
      3. Global features + proprioception embedding → query vector (via MLP)
      4. Multi-Head Attention (16 heads): query attends to local features
      5. Output: global features ∥ weighted local features
    """

    def __init__(
        self,
        map_channels: int = 3,
        map_height: int = 18,
        map_width: int = 7,
        local_feat_dim: int = 64,
        global_feat_dim: int = 64,
        proprio_dim: int = 64,
        num_heads: int = 16,
        cnn_channels: int = 32,
    ):
        super().__init__()

        # CNN with stride=2 downsampling (like AME_Locomotion)
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, cnn_channels, kernel_size=3, padding=1, stride=2),
            nn.ELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
        )

        # Compute downsampled grid size
        ds_h = map_height // 2 + 1  # 18//2+1 = 10, actual: ceil(18/2) = 9
        ds_w = map_width // 2 + 1   # 7//2+1 = 4
        # Real output size after Conv2d stride=2, padding=1:
        # H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        # = floor((18 + 2 - 3 - 1)/2 + 1) = floor(16/2 + 1) = 9
        self.ds_h = (map_height + 1) // 2  # = 9
        self.ds_w = (map_width + 1) // 2   # = 4
        self.num_points = self.ds_h * self.ds_w

        # Positional embedding (MLP per point)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 32), nn.ELU(),
            nn.Linear(32, 32), nn.ELU(),
        )

        # Fuse CNN features + positional embedding → pointwise local features
        self.local_fusion = nn.Sequential(
            nn.Linear(cnn_channels + 32, local_feat_dim), nn.ELU(),
            nn.Linear(local_feat_dim, local_feat_dim), nn.ELU(),
        )

        # Global feature extractor (MLP + MaxPool over points)
        self.global_mlp = nn.Sequential(
            nn.Linear(local_feat_dim, global_feat_dim), nn.ELU(),
            nn.Linear(global_feat_dim, global_feat_dim), nn.ELU(),
        )

        # Query generation: [global_feat ∥ proprio] → query
        self.query_mlp = nn.Sequential(
            nn.Linear(global_feat_dim + proprio_dim, global_feat_dim), nn.ELU(),
            nn.Linear(global_feat_dim, global_feat_dim),
        )

        # Multi-Head Attention (16 heads like AME_Locomotion)
        self.mha = nn.MultiheadAttention(
            embed_dim=local_feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, elevation_map: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        B = elevation_map.shape[0]

        # 1. CNN local features with downsampling
        cnn_feat = self.cnn(elevation_map)  # (B, C, ds_h, ds_w)

        # 2. Positional embedding on downsampled grid
        xs = torch.linspace(-1, 1, self.ds_w, device=elevation_map.device)
        ys = torch.linspace(-1, 1, self.ds_h, device=elevation_map.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pos_grid = torch.stack([grid_x, grid_y], dim=-1)
        pos_embed = self.pos_mlp(pos_grid).unsqueeze(0).expand(B, -1, -1, -1)

        # 3. Fuse CNN + positional → pointwise local features
        cnn_feat = cnn_feat.permute(0, 2, 3, 1)  # (B, ds_h, ds_w, C)
        fused = torch.cat([cnn_feat, pos_embed], dim=-1)
        pointwise_features = self.local_fusion(fused)
        local_features = pointwise_features.view(B, self.num_points, -1)

        # 4. Global features via MLP + MaxPool
        global_feat = self.global_mlp(local_features)
        global_feat_pooled = global_feat.max(dim=1)[0]

        # 5. Generate query from global + proprio
        query_input = torch.cat([global_feat_pooled, proprio_embed], dim=-1)
        query = self.query_mlp(query_input).unsqueeze(1)

        # 6. Multi-Head Attention
        attn_out, _ = self.mha(query=query, key=local_features, value=local_features)
        weighted_local = attn_out.squeeze(1)

        # 7. Concatenate global + weighted local → map embedding
        return torch.cat([global_feat_pooled, weighted_local], dim=-1)


class SimpleMapEncoder(nn.Module):
    """Lightweight map encoder for critic: CNN downsample only, no MHA.

    Paper Sec IV-B: critic does NOT use attention (too costly),
    but still needs processed terrain features.
    Just CNN downsample → flatten → MLP project.
    """

    def __init__(self, map_channels=3, map_height=18, map_width=7,
                 output_dim=64, cnn_channels=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(map_channels, cnn_channels, kernel_size=5, padding=2, stride=2),
            nn.ELU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ELU(),
        )
        ds_h = (map_height + 1) // 2
        ds_w = (map_width + 1) // 2
        cnn_out = ds_h * ds_w * cnn_channels
        self.project = nn.Linear(cnn_out, output_dim)

    def forward(self, elevation_map):
        feat = self.cnn(elevation_map)
        feat = feat.view(feat.shape[0], -1)
        return self.project(feat)


class ProprioEncoder(nn.Module):
    """Proprioception encoder (teacher: plain MLP)."""
    def __init__(self, input_dim=96, hidden_dim=128, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs):
        return self.mlp(obs)


class LSIOProprioEncoder(nn.Module):
    """Student proprioception encoder with Long-Short I/O (LSIO) history.

    Paper Section III-A / Fig. 3 Right:
    Stack proprioceptive observations (excluding base linear velocity and
    commands) from the past K steps, encode them with a temporal network, and
    concatenate the current command before a final MLP.

    The input layout is term-major history produced by mjlab's observation
    manager: [term0_t0..tK-1, term1_t0..tK-1, ..., command, elevation_map].
    """

    def __init__(
        self,
        term_dims: list[int],
        history_len: int,
        command_dim: int,
        output_dim: int = 64,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.term_dims = list(term_dims)
        self.history_len = history_len
        self.command_dim = command_dim
        self.proprio_total_dim = sum(term_dims)
        self.history_total_dim = self.proprio_total_dim * history_len

        self.lstm = nn.LSTM(
            self.proprio_total_dim,
            lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden + command_dim, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Args: obs (B, history_total_dim + command_dim + map...).

        Returns: proprio_embed (B, output_dim).
        """
        B = obs.shape[0]
        hist = obs[:, :self.history_total_dim]
        cmd = obs[:, self.history_total_dim:self.history_total_dim + self.command_dim]

        # Reshape term-major history into (B, K, D) time-major.
        term_tensors = []
        start = 0
        for d in self.term_dims:
            end = start + d * self.history_len
            term_hist = hist[:, start:end].view(B, self.history_len, d)
            term_tensors.append(term_hist)
            start = end
        time_major = torch.cat(term_tensors, dim=-1)  # (B, K, sum(term_dims))

        lstm_out, _ = self.lstm(time_major)
        feat = lstm_out[:, -1]  # (B, lstm_hidden)
        return self.mlp(torch.cat([feat, cmd], dim=-1))


class MoECritic(nn.Module):
    """Mixture-of-Experts critic network."""
    def __init__(self, input_dim, num_experts=8, expert_hidden=256, output_dim=1):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ELU(),
            nn.Linear(128, num_experts),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden), nn.ELU(),
                nn.Linear(expert_hidden, expert_hidden), nn.ELU(),
                nn.Linear(expert_hidden, output_dim),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)
        return (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=-1)
