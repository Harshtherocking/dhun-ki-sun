import torch
import torch.nn as nn
import numpy as np
from config import Config

class Diffusion(nn.Module):
    def __init__(self, latent_dim=Config.LATENT_DIM, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x, t):
        noise = torch.randn_like(x)
        return self.network(x + noise)
