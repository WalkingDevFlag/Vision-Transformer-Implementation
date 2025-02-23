import torch
import argparse
from models.vit import vit_base, vit_large, vit_huge
from data.data_loader import get_data_loaders

def evaluate_model(model, val_loader, device):
    model.to(device)
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
    acc = correct / total
    print(f'Validation Accuracy: {acc:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base', choices=['vit_base', 'vit_large', 'vit_huge'], help='Select model variant')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Path to saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model == 'vit_base':
        model = vit_base()
    elif args.model == 'vit_large':
        model = vit_large()
    elif args.model == 'vit_huge':
        model = vit_huge()
    
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    _, val_loader = get_data_loaders(dataset_name='CIFAR10', batch_size=args.batch_size, img_size=224)
    evaluate_model(model, val_loader, device)
