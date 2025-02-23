import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .positional_embedding import PositionalEmbedding
from .transformer_encoder import TransformerEncoderLayer

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    Splits the image into patches, adds a learnable class token and positional embeddings,
    and processes the sequence with a stack of Transformer encoder layers.
    The output from the class token is passed through a linear classifier.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super(VisionTransformer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding module
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding module (includes extra token for cls)
        self.pos_embed = PositionalEmbedding(self.num_patches, embed_dim)
        
        # Stacking Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize class token and positional embeddings with a truncated normal distribution.
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)
        # Initialize all Linear and Conv2d layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        batch_size = x.size(0)
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Prepend the class token to the patch embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, 1 + num_patches, embed_dim)
        
        # Add positional embeddings
        x = self.pos_embed(x)
        
        # Pass the sequence through the Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Use the output corresponding to the class token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits

# Factory functions to instantiate different ViT variants with the hyperparameters from the paper

def vit_base(img_size=224, num_classes=1000, patch_size=16, dropout=0.0):
    """ViT-Base: 12 layers, 768-dim hidden size, 12 heads, MLP size 3072 (~86M params)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        dropout=dropout
    )

def vit_large(img_size=224, num_classes=1000, patch_size=16, dropout=0.0):
    """ViT-Large: 24 layers, 1024-dim hidden size, 16 heads, MLP size 4096 (~307M params)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        dropout=dropout
    )

def vit_huge(img_size=224, num_classes=1000, patch_size=16, dropout=0.0):
    """ViT-Huge: 32 layers, 1280-dim hidden size, 16 heads, MLP size 5120 (~632M params)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        dropout=dropout
    )
