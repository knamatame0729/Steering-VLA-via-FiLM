"""Fusion module to combine image, text, and state tokens."""

import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, d_model=128, bottleneck_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token, gamma=None, beta=None):
        x = torch.cat([img_token, txt_token, state_token], dim=-1)  # (B, 3 * d_model)

        z = self.encoder(x)                                             # (B, d_model)

        # print(f"x.shape: {x.shape}")
        # print(f"x (first 10 dims): {x[:, :10].detach().cpu().numpy()}")
        if gamma is not None and beta is not None:
            # Apply FiLM modulation
            z = z * gamma + beta  # (B, d_model)

        x = self.decoder(z)  # (B, d_model)
        # x = self.ln(x)      # (B, d_model)

        return x  # fused context