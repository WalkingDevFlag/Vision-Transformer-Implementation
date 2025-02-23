import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
from models.vit import vit_base, vit_large, vit_huge

def load_image(image_path, img_size=224):
    """
    Loads and preprocesses an image.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # add a batch dimension

def run_inference(model, image_tensor, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(device))
    _, pred = torch.max(output, 1)
    return pred.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit_base', choices=['vit_base', 'vit_large', 'vit_huge'], help='Select model variant')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # For demonstration, we assume the number of output classes is 10 (e.g., CIFAR10)
    if args.model == 'vit_base':
        model = vit_base(num_classes=10)
    elif args.model == 'vit_large':
        model = vit_large(num_classes=10)
    elif args.model == 'vit_huge':
        model = vit_huge(num_classes=10)
        
    # In a real test, you would load a trained checkpoint.
    # For now, this example runs inference with the randomly initialized model.
    image_tensor = load_image(args.image)
    pred_class = run_inference(model, image_tensor, device)
    print(f'Predicted class index: {pred_class}')
