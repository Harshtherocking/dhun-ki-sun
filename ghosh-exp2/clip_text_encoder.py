import torch
from torch import nn
from config import Config

class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=512, max_len=77):  # Reduced embed_dim
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))
        
        # Simplified transformer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,  # Reduced from default
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)  # Reduced layers
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(max_len * embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, Config.SEGMENT_LENGTH)
        )

    def forward(self, tokens):
        # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        x = self.token_embedding(tokens)
        x = x + self.position_embedding.unsqueeze(0)
        
        # Transform sequence
        x = self.transformer(x)  # batch_first=True, so no transpose needed
        
        # Flatten and project
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.projection(x)
        return x
