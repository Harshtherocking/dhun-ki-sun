import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from config import Config

class MemoryEfficientAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            channels, 
            Config.ATTENTION_HEADS, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(channels)
        self.slice_size = Config.ATTENTION_SLICE_SIZE
        
    def forward(self, x):
        x = x.view(x.shape[0], -1, x.shape[-1])
        norm_x = self.norm(x)
        
        if Config.ENABLE_MEMORY_EFFICIENT_ATTENTION and self.training:
            # Process attention in chunks to save memory
            chunks = torch.split(norm_x, self.slice_size, dim=1)
            attn_chunks = []
            
            for chunk in chunks:
                chunk_out, _ = self.attention(chunk, chunk, chunk)
                attn_chunks.append(chunk_out)
                
            attn_out = torch.cat(attn_chunks, dim=1)
        else:
            attn_out, _ = self.attention(norm_x, norm_x, norm_x)
            
        return (x + attn_out).view(*x.shape)

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=Config.LATENT_DIM, output_dim=Config.SEGMENT_LENGTH):
        super().__init__()
        self.gradient_checkpointing = Config.GRADIENT_CHECKPOINT
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.GELU(),
            MemoryEfficientAttention(Config.HIDDEN_DIM),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM * 2),
            nn.LayerNorm(Config.HIDDEN_DIM * 2),
            nn.GELU(),
            nn.Linear(Config.HIDDEN_DIM * 2, output_dim),
        )

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return checkpoint.checkpoint(self.decoder, x)
        return self.decoder(x)
