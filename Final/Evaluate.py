import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
from train import UNet, DiffusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sample_plot_image(diffusion, model, n_samples=1, device='cuda'):
    with torch.no_grad():
        model.eval()
        
        # Starting from random noise
        x = torch.randn((n_samples, 1, 128, 256)).to(device)
        
        logger.info("Sampling...")
        for i in tqdm(reversed(range(diffusion.noise_steps)), desc='Sampling'):
            t = torch.full((n_samples,), i, device=device, dtype=torch.long)
            
            with torch.no_grad():
                predicted_noise = model(x, t)
                alpha = diffusion.alphas[t][:, None, None, None]
                alpha_hat = diffusion.alphas_cumprod[t][:, None, None, None]
                beta = diffusion.betas[t][:, None, None, None]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        # Normalize to [0, 1] range
        x = (x - x.min()) / (x.max() - x.min())
        
        return x

def save_samples(samples, save_dir='generated_samples'):
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, sample in enumerate(samples):
        # Convert to numpy and squeeze channel dimension
        spec = sample.squeeze().cpu().numpy()
        
        # Plot and save
        plt.figure(figsize=(10, 5))
        plt.imshow(spec, aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Generated Spectrogram {idx+1}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{idx+1}.png'))
        plt.close()
        
        # Save the raw tensor
        torch.save(sample, os.path.join(save_dir, f'sample_{idx+1}.pt'))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = UNet().to(device)
    diffusion = DiffusionModel(device=device)
    
    logger.info("Loading model checkpoint...")
    model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    
    # Generate samples
    n_samples = 5
    logger.info(f"Generating {n_samples} samples...")
    samples = sample_plot_image(diffusion, model, n_samples=n_samples, device=device)
    
    # Save samples
    logger.info("Saving generated samples...")
    save_samples(samples)
    logger.info("Done! Check the 'generated_samples' directory for the results.")

if __name__ == '__main__':
    main()
