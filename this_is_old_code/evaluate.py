# evaluate.py
import torch
from model.vit import VisionTransformer
from data.data_loader import get_dataloader

def evaluate():
    # Hyperparameters (must match training configuration)
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
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model and load trained weights
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
    ).to(device)
    model.load_state_dict(torch.load("vit_cifar10.pth", map_location=device))
    model.eval()
    
    test_loader = get_dataloader(batch_size=batch_size, image_size=image_size, dataset_name="CIFAR10", train=False)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()
