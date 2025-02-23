# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model.vit import VisionTransformer
from data.data_loader import get_dataloader
from tqdm import tqdm

def train():
    # Hyperparameters
    image_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 10            # For CIFAR-10
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4
    dropout = 0.1
    drop_path_rate = 0.1
    num_epochs = 10
    batch_size = 64
    learning_rate = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the Vision Transformer model
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
    
    # Prepare data loader for training
    train_loader = get_dataloader(batch_size=batch_size, image_size=image_size, dataset_name="CIFAR10", train=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        
    # Save the trained model
    torch.save(model.state_dict(), "vit_cifar10.pth")
    print("Training complete. Model saved as vit_cifar10.pth.")

if __name__ == "__main__":
    train()
