import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import gc
import torch.amp as amp  # Updated import

from config import Config
from New_loader import ThaatRagaDataset
from Encoder import VAEEncoder
from decoder import VAEDecoder

def train_vae():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    
    # Initialize models
    vae_encoder = VAEEncoder().to(device)
    vae_decoder = VAEDecoder().to(device)
    
    # Set models to training mode
    vae_encoder.train()
    vae_decoder.train()
    
    # Initialize dataset and dataloader
    dataset = ThaatRagaDataset("/home/amithk/Desktop/Dhun_ki_sun/Gmodel/Thaat_and_Raga/Thaat_and_Raga")
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )
    
    # Initialize optimizer
    params = list(vae_encoder.parameters()) + list(vae_decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE)
    
    # Initialize gradient scaler for mixed precision training
    scaler = amp.GradScaler('cuda')
    
    # Training loop
    for epoch in range(10):  # Number of epochs
        total_loss = 0
        for batch_idx, (waveform, _) in enumerate(dataloader):
            # Move data to device and ensure correct dtype
            waveform = waveform.to(device=device, dtype=torch.float32)
            waveform.requires_grad_(True)  # Enable gradients
            
            # Clear gradients
            optimizer.zero_grad(True)
            
            # Forward pass with mixed precision
            with amp.autocast('cuda'):
                # Encode
                mu, log_var = vae_encoder(waveform)
                z = vae_encoder.reparameterize(mu, log_var)
                
                # Decode
                reconstructed = vae_decoder(z)
                
                # Calculate losses
                recon_loss = nn.MSELoss()(reconstructed, waveform)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + Config.BETA * kl_loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update total loss
            total_loss += loss.item()
            
            # Clear memory
            del mu, log_var, z, reconstructed
            torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f'Epoch [{epoch+1}], Batch [{batch_idx+1}], '
                      f'Avg Loss: {avg_loss:.4f}, Batch Loss: {loss.item():.4f}, '
                      f'Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}')
                gc.collect()
                torch.cuda.empty_cache()
            
            # Save checkpoint
            if batch_idx % 500 == 0:
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': vae_encoder.state_dict(),
                    'decoder_state_dict': vae_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'scaler_state_dict': scaler.state_dict(),
                }, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pt')
    
if __name__ == "__main__":
    train_vae()
