"""Import-safe model definitions for the Streamlit DDPM app.

This file must not execute any dataset loading, training, or plotting on import.
It only contains the model architecture.
"""

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = self.mlp(embeddings)
        return embeddings


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        h = self.conv1(x)
        time_emb = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_emb
        h = self.conv2(h)
        return h + residual


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
    ):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]

        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        prev_channels = base_channels
        for ch in channels:
            self.encoder_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(prev_channels, ch, time_emb_dim),
                        ResidualBlock(ch, ch, time_emb_dim),
                    ]
                )
            )
            if ch != channels[-1]:
                self.downsample_blocks.append(Downsample(ch))
            else:
                self.downsample_blocks.append(nn.Identity())
            prev_channels = ch

        self.bottleneck = nn.ModuleList(
            [
                ResidualBlock(channels[-1], channels[-1], time_emb_dim),
                ResidualBlock(channels[-1], channels[-1], time_emb_dim),
            ]
        )

        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        reversed_channels = list(reversed(channels))
        for i, ch in enumerate(reversed_channels):
            if i != 0:
                self.upsample_blocks.append(Upsample(prev_channels))
            else:
                self.upsample_blocks.append(nn.Identity())

            self.decoder_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(prev_channels + ch, ch, time_emb_dim),
                        ResidualBlock(ch, ch, time_emb_dim),
                    ]
                )
            )
            prev_channels = ch

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        x = self.init_conv(x)
        skip_connections: list[torch.Tensor] = []

        for blocks, downsample in zip(self.encoder_blocks, self.downsample_blocks):
            for block in blocks:
                x = block(x, time_emb)
            skip_connections.append(x)
            x = downsample(x)

        for block in self.bottleneck:
            x = block(x, time_emb)

        skip_connections = list(reversed(skip_connections))

        for upsample, blocks, skip in zip(self.upsample_blocks, self.decoder_blocks, skip_connections):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x, time_emb)

        x = self.final_conv(x)
        return x


__all__ = ["TimeEmbedding", "ResidualBlock", "Downsample", "Upsample", "UNet"]
