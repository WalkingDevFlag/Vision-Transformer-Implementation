import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer.
    Consists of pre-layer norm, multi-head self-attention, residual connection,
    followed by a feed-forward MLP block (with GELU activation) and another residual connection.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        # Apply self-attention block
        x_res = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x_res + attn_output
        
        # Apply MLP block
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_res + x
        return x
