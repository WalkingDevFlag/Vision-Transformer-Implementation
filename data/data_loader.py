import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data_loaders(dataset_name='CIFAR10', data_dir='./data', batch_size=64, img_size=224):
    """
    Returns training and validation data loaders.
    For demonstration purposes, CIFAR10 is used. (For ImageNet or other datasets, adapt accordingly.)
    """
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_val)
    else:
        raise ValueError("Dataset not supported. Extend this function for other datasets.")
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader
