import torch
from torch import nn
from attention import SelfAttention

class CLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=768, max_len=77):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))
        self.transformer = nn.Transformer(embed_dim, num_encoder_layers=6)

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding
        x = self.transformer(x)
        return x

