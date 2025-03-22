import torch

# Read the original file
with open('VAEtrain.py', 'r') as f:
    content = f.read()

# Additional imports
imports_to_add = """
import gc
import torch.nn as nn
from torch.cuda import amp
"""

# Modify batch size
content = content.replace('batch_size=8', 'batch_size=4')

# Add memory optimization before training loop
memory_opt = """
# Enable gradient checkpointing for memory efficiency
def enable_checkpointing(model):
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True

enable_checkpointing(vae_encoder)
enable_checkpointing(vae_decoder)
enable_checkpointing(clip_encoder)

torch.cuda.empty_cache()
gc.collect()
"""

# Add memory clearing in training loop
training_modifications = """
for epoch in range(num_epochs):
    for batch_idx, (audio_features, waveform) in enumerate(dataloader):
        audio_features = audio_features.to(device)
        waveform = waveform.to(device)
        
        optimizer.zero_grad(True)  # Set grads to None instead of zero
        
        # Forward pass with mixed precision
        with amp.autocast():
            mu, log_var = vae_encoder(waveform)
            z = vae_encoder.reparameterize(mu, log_var)
            reconstructed = vae_decoder(z)
            
            # Calculate losses
            recon_loss = reconstruction_loss(reconstructed, waveform)
            kl_loss = kl_divergence_loss(mu, log_var)
            loss = recon_loss + beta * kl_loss
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Clear memory
        del mu, log_var, z, reconstructed
        torch.cuda.empty_cache()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}')
            gc.collect()
            torch.cuda.empty_cache()
"""

# Add imports at the top
content = imports_to_add + content

# Add memory optimization before training loop
content = content.replace('# Training loop', memory_opt + '\n# Training loop')

# Replace training loop
content = content.replace('for epoch in range(num_epochs):', training_modifications, 1)

# Write the modified content back
with open('VAEtrain.py', 'w') as f:
    f.write(content)
