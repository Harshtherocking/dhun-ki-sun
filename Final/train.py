import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.multiprocessing import freeze_support
from torch.amp import autocast
from torch.amp import GradScaler
from data_loader import get_dataloader
import os
from tqdm import tqdm
import gc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

class DiffusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def pad_to_even(self, x):
        _, _, h, w = x.shape
        pad_h = 0 if h % 2 == 0 else 1
        pad_w = 0 if w % 2 == 0 else 1
        return F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
    def noise_images(self, x, t):
        x = self.pad_to_even(x)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - self.alphas_cumprod[t])
        
        noise = torch.randn_like(x)
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise,
            noise
        )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._conv_block(1, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        self.dec4 = self._conv_block(768, 256)
        self.dec3 = self._conv_block(384, 128)
        self.dec2 = self._conv_block(192, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        self.maxpool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        
        x1 = self.enc1(x)
        x2 = self.enc2(self.maxpool(x1))
        x3 = self.enc3(self.maxpool(x2))
        x4 = self.enc4(self.maxpool(x3))
        
        x4 = x4 + t.unsqueeze(-1).unsqueeze(-1)
        
        x = F.interpolate(x4, size=x3.shape[2:], mode='nearest')
        x = self.dec4(torch.cat([x, x3], dim=1))
        
        x = F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x = self.dec3(torch.cat([x, x2], dim=1))
        
        x = F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x = self.dec2(torch.cat([x, x1], dim=1))
        
        x = self.dec1(torch.cat([x, x1], dim=1))
        
        return x

def train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            
            batch = batch.to(device)
            t = torch.randint(0, diffusion.noise_steps, (batch.shape[0],)).to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                x_t, noise = diffusion.noise_images(batch, t)
                predicted_noise = model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Clear memory every 100 batches
            if batch_idx % 50 == 0:
                cleanup()
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            cleanup()
            continue
    
    return total_loss / len(dataloader)

def main():
    cleanup()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Initializing hyperparameters...")
    num_epochs = 50
    batch_size = 7  # Modified batch size
    learning_rate = 1e-4
    
    logger.info("Creating model and diffusion...")
    model = UNet().to(device)
    diffusion = DiffusionModel(device=device)
    
    logger.info("Setting up optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()
    
    logger.info("Loading data...")
    dataloader = get_dataloader(batch_size=batch_size, num_workers=16)
    logger.info(f"Dataset size: {len(dataloader.dataset)} samples")
    logger.info(f"Number of batches: {len(dataloader)}")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    best_loss = float('inf')
    try:
        for epoch in range(num_epochs):
            logger.info(f"\nStarting epoch {epoch}...")
            avg_loss = train_one_epoch(model, diffusion, dataloader, optimizer, device, epoch, scaler)
            scheduler.step()
            
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            
            cleanup()
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        cleanup()

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn')
        freeze_support()
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
