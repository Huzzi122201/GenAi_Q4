import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from model_architecture import UNet  # Must be available
except ImportError:
    st.error("model_architecture.py not found or UNet class is missing.")
    st.stop()

st.title("DDPM Image Generator")
st.write("Starts from random noise, generates an image, and displays intermediate denoising steps.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 128
TIMESTEPS = 500
BETA_START = 0.0001
BETA_END = 0.02


def create_noise_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    return betas, alphas, alphas_cumprod, alphas_cumprod_prev


betas, alphas, alphas_cumprod, alphas_cumprod_prev = create_noise_schedule(
    timesteps=TIMESTEPS,
    beta_start=BETA_START,
    beta_end=BETA_END,
    device=device,
)


@st.cache_resource
def load_model():
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_emb_dim=128,
        channel_mults=(1, 2, 4),
    ).to(device)

    model_path_candidates = [
        "best_model.pt",
        "best_model (3).pt",
    ]

    model_path = None
    for candidate in model_path_candidates:
        try:
            with open(candidate, "rb"):
                model_path = candidate
                break
        except OSError:
            continue

    if model_path is None:
        st.error("Model weights file not found. Expected best_model.pt (or best_model (3).pt).")
        return None

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

model = load_model()

def inverse_transform(tensor):
    """Convert a tensor in [-1, 1] back to a PIL Image."""
    tensor = tensor.clone().detach().cpu().squeeze(0)
    tensor = (tensor + 1.0) / 2.0  # [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    return transforms.ToPILImage()(tensor)

if st.button("Generate Image"):
    if model is None:
        st.error("Model is not loaded.")
    else:
        st.write("Starting generation...")
        
        # Placeholders for intermediate steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.write("Current Step:")
            image_placeholder = st.empty()
        with col2:
            st.write("Final Output will appear below.")
            final_image_placeholder = st.empty()
        
        # 1) Start from random noise
        x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device)
        
        # Show initial noise
        image_placeholder.image(inverse_transform(x), caption=f"t={TIMESTEPS-1} (Initial Noise)", width=256)
        
        capture_steps = {TIMESTEPS - 1, 400, 300, 200, 100, 50, 0}
        captured = []

        # 2) Reverse diffusion / denoising
        with torch.no_grad():
            for t_val in reversed(range(TIMESTEPS)):
                t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)

                pred_noise = model(x, t_batch)

                alpha_t = alphas[t_val]
                alpha_hat_t = alphas_cumprod[t_val]
                alpha_hat_tm1 = alphas_cumprod_prev[t_val]

                # DDPM reverse step (as in notebook)
                x0_pred = (x - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)

                mean = (
                    (torch.sqrt(alpha_hat_tm1) * (1 - alpha_t) / (1 - alpha_hat_t)) * x0_pred
                    + (torch.sqrt(alpha_t) * (1 - alpha_hat_tm1) / (1 - alpha_hat_t)) * x
                )
                var = (1 - alpha_hat_tm1) / (1 - alpha_hat_t) * (1 - alpha_t)
                x = mean + torch.sqrt(var) * noise

                progress = (TIMESTEPS - t_val) / TIMESTEPS
                progress_bar.progress(progress)
                status_text.text(f"Denoising step: {TIMESTEPS - t_val} / {TIMESTEPS}")

                if t_val in capture_steps:
                    img = inverse_transform(x)
                    captured.append((t_val, img))
                    image_placeholder.image(img, caption=f"t={t_val}", width=256)
                    
        final_image_placeholder.image(inverse_transform(x), caption="Final Generated Image", width=300)
        st.success("Generation complete!")

        if captured:
            st.subheader("Intermediate denoising steps")
            captured_sorted = sorted(captured, key=lambda p: p[0], reverse=True)
            cols = st.columns(min(4, len(captured_sorted)))
            for idx, (t_val, img) in enumerate(captured_sorted):
                cols[idx % len(cols)].image(img, caption=f"t={t_val}", use_container_width=True)
