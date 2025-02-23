# data/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(batch_size=32, image_size=224, dataset_name="CIFAR10", train=True):
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Using CIFAR-10 normalization values as a placeholder
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root="./data", train=train, download=True, transform=transform)
    elif dataset_name == "ImageNet":
        # For ImageNet, the directory structure should follow ImageFolder conventions.
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = datasets.ImageFolder(root="./imagenet/train" if train else "./imagenet/val", transform=transform)
    else:
        raise ValueError("Unsupported dataset")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return dataloader
