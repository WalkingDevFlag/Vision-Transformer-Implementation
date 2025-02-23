# model/vit.py
import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds each patch via a convolutional projection.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Using a Conv2d with kernel and stride equal to patch_size to extract patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                   # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)                   # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)              # (B, num_patches, embed_dim)
        return x

class MLP(nn.Module):
    """
    Simple MLP used inside the Transformer block.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention module.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)                          # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]            # Each: (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v                              # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)      # (B, N, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block with pre-layernorm, multi-head attention, and MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, drop_path=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, dim, dropout=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample (when applied to the main branch of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (stochastic depth).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0, drop_path_rate=0.0):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Patch embedding layer: splits image and projects patches.
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable class token (similar to BERT’s [CLS] token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable position embeddings for patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay rule for each Transformer block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head applied to the output corresponding to the class token
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)                # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)    # (B, 1 + num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Use the output of the class token for classification
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits
