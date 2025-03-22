import torch
from torch import nn
from config import Config
import torch.utils.checkpoint as checkpoint

class VAEEncoder(nn.Module):
    def __init__(self, input_dim=Config.SEGMENT_LENGTH, latent_dim=Config.LATENT_DIM):
        super().__init__()
        self.gradient_checkpointing = Config.GRADIENT_CHECKPOINT
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
        )
        
        self.mu_head = nn.Linear(Config.HIDDEN_DIM, latent_dim)
        self.logvar_head = nn.Linear(Config.HIDDEN_DIM, latent_dim)

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            encoded = checkpoint.checkpoint(self.encoder, x)
        else:
            encoded = self.encoder(x)
            
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
