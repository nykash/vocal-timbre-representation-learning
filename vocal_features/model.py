from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VocalTimbreVAE(nn.Module):
    """MLP VAE for fixed-length log-mel spectrogram chunks."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: tuple[int, ...]):
        super().__init__()
        encoder: list[nn.Module] = []
        width = input_dim
        for hidden_width in hidden_dims:
            encoder.extend(
                [nn.Linear(width, hidden_width), nn.LayerNorm(hidden_width), nn.GELU()]
            )
            width = hidden_width
        self.encoder = nn.Sequential(*encoder)
        self.to_mean = nn.Linear(width, latent_dim)
        self.to_log_variance = nn.Linear(width, latent_dim)

        decoder: list[nn.Module] = []
        width = latent_dim
        for hidden_width in reversed(hidden_dims):
            decoder.extend(
                [nn.Linear(width, hidden_width), nn.LayerNorm(hidden_width), nn.GELU()]
            )
            width = hidden_width
        decoder.append(nn.Linear(width, input_dim))
        self.decoder = nn.Sequential(*decoder)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(features)
        return self.to_mean(hidden), self.to_log_variance(hidden).clamp(-20.0, 2.0)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(features)
        noise = torch.randn_like(mean)
        latent = mean + noise * torch.exp(0.5 * log_variance)
        return self.decoder(latent), mean, log_variance


def vae_loss(
    features: torch.Tensor,
    reconstruction: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction_loss = F.mse_loss(reconstruction, features)
    kl_loss = -0.5 * torch.mean(
        1 + log_variance - mean.square() - log_variance.exp()
    )
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss
