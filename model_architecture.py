import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # should print: cuda

import os
images=[]
data_path = "/kaggle/input/datasets/sairam3/wikiart/wikiart"
for root, dirs, files in os.walk(data_path):
    for f in files:
        if f.endswith((".jpg",".png",".jpeg")):
            images.append(os.path.join(root,f))
            
   
print("images",len(images))


images=images[:5000]

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torch


transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                      
])

class WikiArtDataset(Dataset):
    def __init__(self,images,transform=None):
        self.images=images
        self.transform=transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        img=Image.open(self.images[index]).convert("RGB")
        if img.size[0] > 2000 or img.size[1] > 2000:
                return self.__getitem__((index + 1) % len(self.images))
        if self.transform:
            img=self.transform(img)
        return img
        
        

dataset=WikiArtDataset(images,transform)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,          # ← THIS IS THE FIX
    pin_memory=True,
)


T=300
beta_start=1e-4
beta_end=0.02
betas=torch.linspace(beta_start,beta_end,T)
alphas=1.0-betas
alpha_hat=torch.cumprod(alphas,dim=0)
# Do this ONCE before training starts
sqrt_alpha_hat = torch.sqrt(alpha_hat)
sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

def add_noise_fast(x, t, sqrt_alpha_hat, sqrt_one_minus_alpha_hat):
    noise = torch.randn_like(x)
    s_a = sqrt_alpha_hat[t].to(x.device)[:, None, None, None]
    s_1a = sqrt_one_minus_alpha_hat[t].to(x.device)[:, None, None, None]
    return s_a * x + s_1a * noise, noise



import matplotlib.pyplot as plt
batch = next(iter(loader))          # ✅ get batch

img = batch[0].unsqueeze(0)  # single image

steps = [0, 50, 100, 200, 299]

plt.figure(figsize=(15,3))

for i, step in enumerate(steps):
    t = torch.tensor([step])
    noisy_img, _ = add_noise_fast(
    img, 
    t, 
    sqrt_alpha_hat, 
    sqrt_one_minus_alpha_hat
)

    noisy_img = noisy_img.squeeze().permute(1,2,0)
    noisy_img = (noisy_img * 0.5 + 0.5).clamp(0,1)

    plt.subplot(1,5,i+1)
    plt.imshow(noisy_img)
    plt.title(f"t={step}")
    plt.axis("off")

plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import torch
import torch.nn as nn
import math

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


# ── Sanity check ─────────────────────────────────────────────────────
model = UNet().to(device)
x = torch.randn(4, 3, 128, 128).to(device)
t = torch.randint(0, 300, (4,)).to(device)
out = model(x, t)
print(out.shape)   # torch.Size([4, 3, 128, 128])
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

from PIL import Image
Image.MAX_IMAGE_PIXELS = None                  # fix DecompressionBomb

from torch.amp import GradScaler, autocast
import torch.optim as optim
import matplotlib.pyplot as plt

# ── Device & Model ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNet().to(device)


#if torch.cuda.device_count() > 1:
    #print(f"Using {torch.cuda.device_count()} GPUs")
    #model = nn.DataParallel(model)

# ── Optimizer & Scaler ───────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scaler    = GradScaler('cuda')                 # ← fixed
loss_fn   = nn.MSELoss()

# ── Noise Schedule ───────────────────────────────────────────────────
T                        = 200
betas                    = torch.linspace(1e-4, 0.02, T).to(device)
alphas                   = 1.0 - betas
alpha_hat                = torch.cumprod(alphas, dim=0)
sqrt_alpha_hat           = torch.sqrt(alpha_hat)
sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

def add_noise(x, t):
    noise = torch.randn_like(x)
    sa    = sqrt_alpha_hat[t][:, None, None, None]
    s1a   = sqrt_one_minus_alpha_hat[t][:, None, None, None]
    return sa * x + s1a * noise, noise

# ── Training Loop ────────────────────────────────────────────────────
EPOCHS       = 20
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    
    for i,batch in enumerate(loader):
        x0 = batch.to(device)
        B  = x0.shape[0]
        t  = torch.randint(0, T, (B,), device=device).long()

        x_noisy, noise = add_noise(x0, t)
        optimizer.zero_grad()

        with autocast('cuda'):                 # ← fixed
            noise_pred = model(x_noisy, t)
            loss       = loss_fn(noise_pred, noise)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()


    avg_loss = epoch_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")

# ── Loss Plot ────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(loss_history, linewidth=2, color='royalblue')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# ── Save Checkpoint ──────────────────────────────────────────────────
torch.save(model.state_dict(), "ddpm_model.pth")
print("Model saved → ddpm_model.pth")

import torch
import matplotlib.pyplot as plt

# ── Precompute values needed for reverse process ─────────────────────
# (these use the same schedule from training)

betas                    = torch.linspace(1e-4, 0.02, T).to(device)
alphas                   = 1.0 - betas
alpha_hat                = torch.cumprod(alphas, dim=0)
sqrt_alpha_hat           = torch.sqrt(alpha_hat)
sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
sqrt_recip_alpha         = torch.sqrt(1.0 / alphas)

# posterior variance β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
alpha_hat_prev           = torch.cat([torch.tensor([1.0]).to(device), alpha_hat[:-1]])
posterior_variance       = betas * (1.0 - alpha_hat_prev) / (1.0 - alpha_hat)


# ── Single Reverse Step ──────────────────────────────────────────────
@torch.no_grad()
def reverse_step(model, x, t_index):
    """
    Given noisy image x at timestep t_index,
    return the slightly less noisy image at t_index - 1.
    """
    t_batch = torch.full((x.shape[0],), t_index, device=device, dtype=torch.long)

    # U-Net predicts the noise
    noise_pred = model(x, t_batch)

    # DDPM reverse formula
    # x_{t-1} = (1/√α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
    coef1 = sqrt_recip_alpha[t_index]
    coef2 = betas[t_index] / sqrt_one_minus_alpha_hat[t_index]

    mean  = coef1 * (x - coef2 * noise_pred)

    if t_index > 0:
        noise = torch.randn_like(x)
        std   = torch.sqrt(posterior_variance[t_index])
        x_prev = mean + std * noise           # add stochasticity
    else:
        x_prev = mean                         # final step — no noise added

    return x_prev


# ── Full Sampling Loop ───────────────────────────────────────────────
@torch.no_grad()
def sample(model, n_images=5, save_intermediates=True):
    """
    Generate n_images from pure Gaussian noise.
    Returns final images + intermediate steps for visualization.
    """
    model.eval()

    # Start from pure noise  x_T ~ N(0, I)
    x = torch.randn(n_images, 3, 128, 128).to(device)

    intermediates = []                        # store snapshots for visualization
    capture_steps = set([T-1, 200, 150, 100, 50, 0])  # which steps to save

    for t_index in reversed(range(T)):       # T-1 → T-2 → ... → 0
        x = reverse_step(model, x, t_index)

        if save_intermediates and t_index in capture_steps:
            intermediates.append((t_index, x.clone().cpu()))

    return x.cpu(), intermediates


# ── Helper: tensor → plottable image ────────────────────────────────
def to_img(tensor):
    """Denormalize from [-1,1] to [0,1] and clamp."""
    img = (tensor * 0.5 + 0.5).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()      # (H, W, C)


# ── Generate & Visualize ─────────────────────────────────────────────
generated, intermediates = sample(model, n_images=5)


# ── Plot 1: 5 Generated Images ───────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Generated Images from Pure Noise", fontsize=14, fontweight='bold')

for i, ax in enumerate(axes):
    ax.imshow(to_img(generated[i]))
    ax.set_title(f"Image {i+1}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("generated_images.png", dpi=150)
plt.show()


# ── Plot 2: Reverse Diffusion Steps (for image 0) ────────────────────
fig, axes = plt.subplots(1, len(intermediates), figsize=(18, 4))
fig.suptitle("Reverse Diffusion Process (Image 1)", fontsize=14, fontweight='bold')

for ax, (step, imgs) in zip(axes, reversed(intermediates)):
    ax.imshow(to_img(imgs[0]))              # show first image at this step
    ax.set_title(f"t={step}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("reverse_steps.png", dpi=150)
plt.show()

print("Sampling complete.")
print(f"Generated shape: {generated.shape}")

from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import numpy as np
import torch
import matplotlib.pyplot as plt


# ── Helper: tensor → numpy [0,1] ─────────────────────────────────────
def tensor_to_numpy(t):
    """(C,H,W) tensor in [-1,1]  →  (H,W,C) numpy in [0,1]"""
    return ((t * 0.5 + 0.5).clamp(0, 1)
              .permute(1, 2, 0)
              .cpu()
              .numpy())


# ── Step 1: Pick a target image from your dataset ────────────────────
model.eval()

# Grab one real image from the dataloader
target_batch = next(iter(loader))          # (B, 3, 128, 128)
target_clean = target_batch[0:1].to(device)   # use first image, keep batch dim

target_np = tensor_to_numpy(target_clean[0])  # save numpy version for scoring


# ── Step 2: Noise the target to a mid-level timestep ─────────────────
# Instead of starting from t=T (pure noise), we start from t=200
# This gives the model a "hint" of the target structure — classic reconstruction

T_start = 200
t_tensor = torch.full((1,), T_start, device=device, dtype=torch.long)
x_noised, _ = add_noise(target_clean, t_tensor)   # partially noised target


# ── Step 3: Reverse from T_start → 0 ────────────────────────────────
@torch.no_grad()
def reconstruct(model, x_start, t_start):
    """
    Run reverse diffusion from t_start down to 0.
    Returns final reconstruction + intermediates.
    """
    x = x_start.clone()
    intermediates = []
    capture_steps = set([t_start, 150, 100, 50, 0])

    for t_index in reversed(range(t_start)):
        x = reverse_step(model, x, t_index)

        if t_index in capture_steps:
            intermediates.append((t_index, x.clone().cpu()))

    return x.cpu(), intermediates


reconstructed, intermediates = reconstruct(model, x_noised, T_start)
reconstructed_np = tensor_to_numpy(reconstructed[0])


# ── Step 4: PSNR & SSIM Scores ───────────────────────────────────────
psnr_score = psnr_fn(
    target_np,
    reconstructed_np,
    data_range=1.0
)

ssim_score = ssim_fn(
    target_np,
    reconstructed_np,
    data_range=1.0,
    channel_axis=2          # color images — specify channel axis
)

print("=" * 40)
print(f"  PSNR Score : {psnr_score:.2f} dB")
print(f"  SSIM Score : {ssim_score:.4f}   (1.0 = perfect)")
print("=" * 40)


# ── Step 5: Side-by-Side Comparison ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Image Reconstruction", fontsize=14, fontweight='bold')

# Target
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