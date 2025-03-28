import torch
import matplotlib.pyplot as plt
import os
from train import UNet  # Import your trained model
import numpy as np

# ✅ Define Paths
MODEL_CHECKPOINT = "/home/amithk/Desktop/Dhun_ki_sun/FinalModel/Diffusion/sd/checkpoints/best_model.pt"
OUTPUT_DIR = "generated_spectrograms/"

# ✅ Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT))
model.eval()

# ✅ Generate Random Input
random_input = torch.randn((1, 1, 128, 300), device=device)  # Shape: (batch, channels, mel_bins, time_frames)

# ✅ Run inference
with torch.no_grad():
    generated_spectrogram = model(random_input, torch.tensor([0.0]).to(device))  # Add time tensor if needed

# ✅ Convert to NumPy and Save
generated_spectrogram = generated_spectrogram.cpu().squeeze().numpy()

# ✅ Save Spectrogram as Image
plt.imshow(generated_spectrogram, aspect='auto', cmap='magma')
plt.colorbar()
plt.title("Generated Mel Spectrogram")
spectrogram_path = os.path.join(OUTPUT_DIR, "generated_spectrogram.png")
plt.savefig(spectrogram_path)
plt.show()

print(f"✅ Spectrogram saved at: {spectrogram_path}")

