import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io

try:
    from model_architecture import UNet  # Must be available
except ImportError:
    st.error("model_architecture.py not found or UNet class is missing.")
    st.stop()

st.title("WikiArt Generation using DDPM 🎨")
st.write("Starts from random noise, generates an image, and displays intermediate denoising steps.")

# Constants and DDPM parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 300
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1. - torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)

@st.cache_resource
def load_model():
    model = UNet().to(device)
    try:
        model.load_state_dict(torch.load('ddpm_model.pth', map_location=device))
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

if st.button("Generate Art"):
    if model is None:
        st.error("Model is not loaded.")
    else:
        st.write("Starting generation process...")
        
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
        
        # 1. Start from random noise
        x = torch.randn((1, 3, 128, 128)).to(device)
        
        # Show initial noise
        image_placeholder.image(inverse_transform(x), caption=f"Step {T} (Initial Noise)", width=256)
        
        # 2. Denoising process
        with torch.no_grad():
            for i in reversed(range(T)):
                t = torch.tensor([i], dtype=torch.long).to(device)
                
                # Predict noise
                predicted_noise = model(x, t)
                
                # Equation for x_{t-1}
                alpha = alphas[t]
                alpha_cumprod = alphas_cumprod[t]
                beta = betas[t]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = sqrt_recip_alphas[t] * (x - beta / sqrt_one_minus_alphas_cumprod[t] * predicted_noise) + torch.sqrt(posterior_variance[t]) * noise
                
                # Update progress
                progress = (T - i) / T
                progress_bar.progress(progress)
                status_text.text(f"Denoising step: {T - i} / {T}")
                
                # 3. Display intermediate denoising steps
                if i % 30 == 0 or i == 0:  # Show every 30 steps and the final step
                    img = inverse_transform(x)
                    image_placeholder.image(img, caption=f"Step {i} remaining", width=256)
                    
        final_image_placeholder.image(inverse_transform(x), caption="Final Generated Art", width=300)
        st.success("Generation complete!")
        st.balloons()
