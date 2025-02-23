import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
from models.vit import vit_base, vit_large, vit_huge
from data.data_loader import get_data_loaders

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-3):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Evaluate on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        epoch_acc = correct / total
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save the best model checkpoint
            torch.save(model.state_dict(), 'best_model.pth')
    print(f'Best Val Acc: {best_acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base', choices=['vit_base', 'vit_large', 'vit_huge'], help='Select model variant')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the selected model variant
    if args.model == 'vit_base':
        model = vit_base()
    elif args.model == 'vit_large':
        model = vit_large()
    elif args.model == 'vit_huge':
        model = vit_huge()
    
    train_loader, val_loader = get_data_loaders(dataset_name='CIFAR10', batch_size=args.batch_size, img_size=224)
    train_model(model, train_loader, val_loader, device, num_epochs=args.epochs, lr=args.lr)
