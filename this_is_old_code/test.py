# test.py
import torch
from model.vit import VisionTransformer

def test_model():
    image_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 10
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4
    dropout = 0.1
    drop_path_rate = 0.1
    batch_size = 2

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
    
    # Create a dummy input tensor simulating a batch of images
    dummy_input = torch.randn(batch_size, in_channels, image_size, image_size)
    output = model(dummy_input)
    print("Model output shape:", output.shape)
    # Expected output shape: (batch_size, num_classes)

if __name__ == "__main__":
    test_model()
