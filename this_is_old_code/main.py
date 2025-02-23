# main.py
import torch
from model.vit import VisionTransformer

def main():
    # Hyperparameters (example settings based on the paper)
    image_size = 224      # Input image resolution
    patch_size = 16       # Each patch will be 16x16 pixels
    in_channels = 3       # RGB images
    num_classes = 1000    # e.g. for ImageNet classification
    embed_dim = 768       # Embedding dimension (ViT-Base)
    depth = 12            # Number of Transformer encoder layers
    num_heads = 12        # Number of attention heads
    mlp_ratio = 4         # MLP hidden dimension is 4x embed_dim
    dropout = 0.1         # Dropout probability
    drop_path_rate = 0.1  # Stochastic depth rate

    # Create the Vision Transformer model
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate
    )
    
    print(model)

if __name__ == "__main__":
    main()
