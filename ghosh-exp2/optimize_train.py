import torch
from torch.cuda import amp

# Read the original file
with open('VAEtrain.py', 'r') as f:
    content = f.read()

# Add imports
imports_to_add = """
from torch.cuda import amp
"""

# Add scaler before the training loop
scaler_init = """
# Initialize gradient scaler for mixed precision training
scaler = amp.GradScaler()
"""

# Modify the training loop to use mixed precision
training_modifications = """
for epoch in range(num_epochs):
    for batch_idx, (audio_features, waveform) in enumerate(dataloader):
        audio_features = audio_features.to(device)
        waveform = waveform.to(device)
        
        optimizer.zero_grad()
        
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
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}')
"""

# Find and replace the training loop
import re
content = content.replace('for epoch in range(num_epochs):', training_modifications, 1)

# Add imports at the top
content = imports_to_add + content

# Add scaler initialization before the training loop
content = content.replace('# Training loop', scaler_init + '\n# Training loop')

# Write the modified content back
with open('VAEtrain.py', 'w') as f:
    f.write(content)
