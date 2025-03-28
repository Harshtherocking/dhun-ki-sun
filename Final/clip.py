import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class SpectrogramEncoder(nn.Module):
    def __init__(self, in_channels: int, n_embd: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, n_embd, kernel_size=3, stride=2, padding=1)
        
        self.norm = nn.LayerNorm(n_embd)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.mean(dim=[-2, -1])  # Global average pooling
        return self.norm(x)

class SpectrogramCLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=False)
        x += residue

        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        x = self.linear_2(x)
        x += residue
        
        return x

class SpectrogramCLIP(nn.Module):
    def __init__(self, in_channels=1, n_embd=768, n_layers=12, n_head=12):
        super().__init__()
        
        self.encoder = SpectrogramEncoder(in_channels, n_embd)
        self.layers = nn.ModuleList([SpectrogramCLIPLayer(n_head, n_embd) for _ in range(n_layers)])
        self.layernorm = nn.LayerNorm(n_embd)
    
    def forward(self, spectrograms: torch.FloatTensor) -> torch.FloatTensor:
        x = self.encoder(spectrograms)
        for layer in self.layers:
            x = layer(x)
        return self.layernorm(x)
