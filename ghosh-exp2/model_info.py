from VAEtrain import *
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def print_model_info(model, name):
    print(f"\n{name} Architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_encoder = VAEEncoder().to(device)
vae_decoder = VAEDecoder().to(device)
clip_encoder = CLIPEncoder().to(device)

print_model_info(vae_encoder, "VAE Encoder")
print_model_info(vae_decoder, "VAE Decoder")
print_model_info(clip_encoder, "CLIP Encoder")

# Print input shape
print("\nInput shape (batch_size=4):")
print(f"Waveform shape: (4, 1, 110250)")

# Calculate theoretical memory usage
def calculate_memory_usage():
    batch_size = 4
    input_size = 110250
    
    # Memory for input
    input_mem = batch_size * input_size * 4  # 4 bytes for float32
    
    # Memory for gradients (roughly same as parameters)
    grad_mem = sum(p.numel() * 4 for model in [vae_encoder, vae_decoder, clip_encoder] for p in model.parameters())
    
    # Memory for optimizer state (Adam uses 2 additional buffers)
    optimizer_mem = grad_mem * 2
    
    total_mem = (input_mem + grad_mem + optimizer_mem) / (1024 * 1024)  # Convert to MB
    print(f"\nEstimated minimum memory usage: {total_mem:.2f} MB")

calculate_memory_usage()
