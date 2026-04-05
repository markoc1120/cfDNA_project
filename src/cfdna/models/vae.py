from collections import namedtuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

VAEOutput = namedtuple('VAEOutput', ['reconstruction', 'mu', 'logvar'])
VAELossComponents = namedtuple('VAELossComponents', ['total', 'recon', 'kl'])


# (same as RebinnedCNNUnit: Conv+BN+LeakyReLU+Conv+BN+LeakyReLU+Pool)
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_ks: tuple[int, int] = (2, 4)):
        super().__init__()
        DefaultConv2d = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers = nn.Sequential(
            DefaultConv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            DefaultConv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=pool_ks),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (2, 4),
        output_padding: tuple[int, int] = (0, 0),
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=output_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class VAEModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        base_channels: int = 32,
        input_height: int = 50,
        input_width: int = 200,
    ):
        super().__init__()
        C = base_channels
        self._input_height = input_height
        self._input_width = input_width

        # encoder
        self.encoder = nn.Sequential(
            EncoderBlock(1, C, pool_ks=(2, 4)),
            EncoderBlock(C, C * 2, pool_ks=(2, 4)),
            EncoderBlock(C * 2, C * 4, pool_ks=(1, 5)),
        )

        # discover encoded spatial shape
        with torch.no_grad():
            probe = torch.zeros(1, 1, input_height, input_width)
            encoded = self.encoder(probe)
            self._encoded_shape = encoded.shape[1:]
        flat_dim = self._encoded_shape.numel()

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # decode
        self.fc_decode = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            DecoderBlock(C * 4, C * 2, stride=(1, 5)),
            DecoderBlock(C * 2, C, stride=(2, 4)),
            nn.ConvTranspose2d(
                C,
                1,
                kernel_size=3,
                stride=(2, 4),
                padding=1,
                output_padding=(1, 3),
            ),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.flatten(self.encoder(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        # during inference, use the mean directly (deterministic)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, *self._encoded_shape)
        h = self.decoder(h)
        # interpolate to exact input dims if ConvTranspose doesn't land exactly
        target_size = (self._input_height, self._input_width)
        if h.shape[2:] != target_size:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)
        return h

    def forward(self, x: Tensor) -> VAEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return VAEOutput(reconstruction=reconstruction, mu=mu, logvar=logvar)


def vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    return_components: bool = False,
) -> Tensor | VAELossComponents:
    recon_per_sample = F.mse_loss(recon, target, reduction='none').flatten(1).mean(dim=1)
    recon_loss = recon_per_sample.mean()

    # KL(q(z|x) || p(z)) where p(z) = N(0, I)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_per_sample.mean()

    total_loss = recon_loss + beta * kl_loss

    if return_components:
        return VAELossComponents(total=total_loss, recon=recon_loss, kl=kl_loss)
    return total_loss
