"""Model architecture used by the Streamlit app.

Important: This file must be safe to import in Streamlit Cloud.
The notebook export originally included dataset loading, training loops,
and plotting at module import time, which crashes deployment.

This module intentionally contains ONLY the PyTorch model definitions.
"""

import math

import torch
import torch.nn as nn

# ── Your time embedding (kept as-is, it's correct) ──────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        # Add the MLP that was missing
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
        )

    def forward(self, t):
        half_dim = self.emb_dim // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) *
            -(math.log(10000) / (half_dim - 1))
        ).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.mlp(emb)   # ← MLP added here


# ── Residual Block (unchanged) ───────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # ← Pick largest divisor of in_ch that is ≤ 8
        groups_in  = min(8, in_ch)  if in_ch  % min(8, in_ch)  == 0 else 1
        groups_out = min(8, out_ch) if out_ch % min(8, out_ch) == 0 else 1

        self.norm1    = nn.GroupNorm(groups_in,  in_ch)   # ← was hardcoded 8
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2    = nn.GroupNorm(groups_out, out_ch)  # ← was hardcoded 8
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act      = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


# ── Fixed Downsample (strided conv, not MaxPool) ─────────────────────
class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ── Fixed Upsample (nearest + conv, avoids checkerboard) ────────────
class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


# ── Merged U-Net ─────────────────────────────────────────────────────
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_emb_dim = 256
        self.time_mlp = TimeEmbedding(self.time_emb_dim)

        # Encoder
        self.enc1  = ResidualBlock(3,   64,  self.time_emb_dim)
        self.down1 = Downsample(64)
        self.enc2  = ResidualBlock(64,  128, self.time_emb_dim)
        self.down2 = Downsample(128)
        self.enc3  = ResidualBlock(128, 256, self.time_emb_dim)
        self.down3 = Downsample(256)

        # Bottleneck
        self.bot   = ResidualBlock(256, 256, self.time_emb_dim)

        # Decoder — note all 3 skip connections used
        self.up3   = Upsample(256)
        self.dec3  = ResidualBlock(256 + 256, 128, self.time_emb_dim)  # skip enc3
        self.up2   = Upsample(128)
        self.dec2  = ResidualBlock(128 + 128, 64,  self.time_emb_dim)  # skip enc2
        self.up1   = Upsample(64)
        self.dec1  = ResidualBlock(64  + 64,  64,  self.time_emb_dim)  # skip enc1 ← fixed

        # Output
        self.out_norm = nn.GroupNorm(8, 64)
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)                       # (B,  64, 128, 128)
        e2 = self.enc2(self.down1(e1), t_emb)           # (B, 128,  64,  64)
        e3 = self.enc3(self.down2(e2), t_emb)           # (B, 256,  32,  32)

        # Bottleneck
        b  = self.bot(self.down3(e3), t_emb)            # (B, 256,  16,  16)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)

        return self.out_conv(nn.functional.silu(self.out_norm(d1)))


__all__ = [
    "TimeEmbedding",
    "ResidualBlock",
    "Downsample",
    "Upsample",
    "UNet",
]
axes[0].imshow(target_np)
axes[0].set_title("Target (Original)", fontweight='bold')
axes[0].axis("off")

# Noised starting point
noised_np = tensor_to_numpy(x_noised[0].cpu())
axes[1].imshow(noised_np)
axes[1].set_title(f"Noised Input (t={T_start})")
axes[1].axis("off")

# Reconstructed
axes[2].imshow(reconstructed_np)
axes[2].set_title(
    f"Reconstructed\nPSNR: {psnr_score:.2f} dB | SSIM: {ssim_score:.4f}"
)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("reconstruction_comparison.png", dpi=150)
plt.show()


# ── Step 6: Intermediate Reconstruction Steps ─────────────────────────
fig, axes = plt.subplots(1, len(intermediates), figsize=(18, 4))
fig.suptitle("Denoising Steps During Reconstruction", fontsize=13, fontweight='bold')

for ax, (step, img) in zip(axes, reversed(intermediates)):
    ax.imshow(tensor_to_numpy(img[0]))
    ax.set_title(f"t={step}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("reconstruction_steps.png", dpi=150)
plt.show()


# ── Step 7: Score Interpretation ─────────────────────────────────────
print("\nScore Interpretation:")
print(f"  PSNR > 20 dB  → Acceptable reconstruction")
print(f"  PSNR > 30 dB  → Good reconstruction")
print(f"  SSIM > 0.7    → Structurally similar")
print(f"  SSIM > 0.9    → Excellent similarity")
print(f"\n  Your PSNR: {psnr_score:.2f} dB")
print(f"  Your SSIM: {ssim_score:.4f}")