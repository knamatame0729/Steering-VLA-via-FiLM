"""Diffusion policy head for action generation."""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    T: int = 16  # number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    action_dim: int = 4
    cond_dim: int = 128 # conditional input dim


def make_beta_schedule(cfg: DiffusionConfig):
    """
    Compute a schedule for linear beta values from beta_start to beta_end.

    param cfg: A DiffusionConfig object containing the diffusion hyperparameters.
    return: A tuple of tensors containing the beta values, alpha values and alpha bar values.
    """
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)  # As t increases, beta_t becomes larger, so later steps have more noise
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)                     # As t increases, alpha_bar[t] decreases and x_t becomes noisier
    return betas, alphas, alpha_bar

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: (B,) integer timesteps in [0, T-1]
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                device=device
            )
        )
        # (B, half_dim)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class ActionDenoiseModel(nn.Module):
    """
    epsilon_theta(x_t, t, cond)
    x_t:   (B, action_dim)
    t:     (B,) integer timestep
    cond:  (B, cond_dim) fused VLA token
    """
    def __init__(self, cfg: DiffusionConfig, time_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # Input dimension is action_dim + time_emb_dim + cond_dim (4 + 32 + 128)
        in_dim = cfg.action_dim + time_emb_dim + cfg.cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        """
        x_t: (B, action_dim) noisy actions at timestep t
        t:   (B,)
        cond: (B, cond_dim) fused VLA token
        returns: (B, action_dim) predicted noise
        """
        t_emb = self.time_emb(t)  # (B, time_emb_dim)

        # Concatenate x_t, t_emb, and cond
        x = torch.cat([x_t, t_emb, cond], dim=-1)  # (B, in_dim)

        # MLP to predict noise
        eps_pred = self.net(x)  # (B, action_dim)
        
        return eps_pred

class DiffusionPolicyHead(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.denoise_model = ActionDenoiseModel(cfg)
        betas, alphas, alpha_bar = make_beta_schedule(cfg)
        # register as buffers so they move with the module’s device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        x0: (B, action_dim) ground-truth actions
        t:  (B,) Timestep for each example in the batch
        noise: (B, action_dim) sampled noise
        returns: x_t (B, action_dim)
        """
        # gather alpha_bar_t for each t in batch
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)  # (B, 1)

        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def loss(self, actions, cond):
        """
        actions: (B, action_dim)  ground-truth actions
        cond:    (B, cond_dim)    fused VLA token
        """
        B = actions.size(0)
        device = actions.device

        # Sample t uniformly for each example in the batch
        t = torch.randint(0, self.cfg.T, (B,), device=device) # uniform sampling t

        # Sample noise
        noise = torch.randn_like(actions)           # (B, action_dim)

        # Forward diffusion
        x_t = self.q_sample(actions, t, noise)      # noisy actions at timestep t

        # Predict the noise
        eps_pred = self.denoise_model(x_t, t, cond)

        # Main Loss
        main_loss = F.mse_loss(eps_pred, noise)
        
        # Loss per timestep
        loss_per_t = F.mse_loss(eps_pred, noise, reduction='none').mean(dim=-1)  # (B,)

        # Collect loss per timestep for logging
        loss_dict = {}
        for t_val in range(self.cfg.T):
            mask = (t == t_val)
            if mask.any():
                loss_dict[f"loss_t{t_val}"] = loss_per_t[mask].mean().item()

        # Compute MSE loss between the true noise and predicted noise
        return main_loss, loss_dict

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        """
        cond: (B, cond_dim) Fused VLA token
        returns: (B, action_dim) sampled actions x_0
        """
        self.eval()
        if n_samples is None:
            B = cond.size(0)
        else:
            B = n_samples
            cond = cond.expand(B, -1)

        # Start from pure noise (action_dim)
        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)
        
        # Reverse Loop
        for t_step in reversed(range(self.cfg.T)):  # 15, 14, ..., 0

            # Create timestep tensor
            t = torch.full((B,), t_step, device=cond.device, dtype=torch.long)

            # Predict noise
            eps_pred = self.denoise_model(x_t, t, cond)

            # Get parameters for current timestep
            beta_t = self.betas[t_step]
            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alpha_bar[t_step]

            # now reverse diffusion  (aka DDPM)
            # mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred) # original
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t) # simplified
            if t_step > 0:
                noise = torch.randn_like(x_t)
                # x_t = mean + torch.sqrt(beta_t) * noise # original
                x_t = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(beta_t) * noise
            else:
                x_t = x0_pred # x0_pred or mean
        return x_t
