import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """
    Adds learnable positional embeddings to the patch embeddings.
    One extra token is added for the classification token.
    """
    def __init__(self, num_patches, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # +1 for class token

    def forward(self, x):
        # x shape: (batch_size, num_patches + 1, embed_dim)
        return x + self.pos_embed
