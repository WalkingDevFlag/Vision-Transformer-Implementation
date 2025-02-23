## Vision Transformer (ViT) Implementation

This repository provides a complete implementation of the Vision Transformer (ViT) architecture as described in the paper "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (ICLR 2021).

## Project Structure

```
ViT/
├── models/
│   ├── __init__.py
│   ├── patch_embedding.py
│   ├── positional_embedding.py
│   ├── transformer_encoder.py
│   └── vit.py
├── data/
│   ├── __init__.py
│   └── data_loader.py
├── train.py
├── eval.py
├── test.py
└── README.md
```

## Components

### Main ViT Architecture (`models/vit.py`)

- Defines the overall Vision Transformer model
- Integrates patch embedding, positional embedding, and a stack of Transformer encoder layers
- Uses a [CLS] token whose final representation is fed into a classification head
- Provides factory functions for three variants:
  - **ViT-Base:** 12 layers, 768-dim hidden size, 12 heads (~86M params)
  - **ViT-Large:** 24 layers, 1024-dim hidden size, 16 heads (~307M params)
  - **ViT-Huge:** 32 layers, 1280-dim hidden size, 16 heads (~632M params)

### Other Components

- **Patch Embedding (`models/patch_embedding.py`):** Splits input images into fixed-size patches and projects them to the model's latent space.
- **Positional Embedding (`models/positional_embedding.py`):** Adds learnable positional embeddings to retain spatial information.
- **Transformer Encoder (`models/transformer_encoder.py`):** Implements the standard Transformer encoder block with multi-head self-attention and MLP layers.
- **Data Preprocessing (`data/data_loader.py`):** Provides functions to load and preprocess data (using CIFAR10 as an example).
- **Training Script (`train.py`):** Trains the selected ViT model variant on the dataset.
- **Evaluation Script (`eval.py`):** Loads a saved checkpoint and evaluates the model on the validation set.
- **Test Script (`test.py`):** Runs inference on a sample image and prints the predicted class index.

## How to Run

### Training

Run the training script with your chosen model variant and hyperparameters:

```bash
python train.py --model vit_base --epochs 10 --lr 0.001 --batch_size 64
```

To train ViT-Large or ViT-Huge, change the `--model` argument accordingly.

### Evaluation

To evaluate a pre-trained model:

```bash
python eval.py --model vit_base --checkpoint best_model.pth --batch_size 64
```

Ensure the checkpoint file (`best_model.pth`) exists in the working directory.

### Testing / Inference

To run inference on a sample image:

```bash
python test.py --model vit_base --image path_to_image.jpg
```

This will load the image, preprocess it, run it through the model, and print the predicted class index.

## Modifying Hyperparameters and Switching Variants

- **Switching Variants:** In all scripts, use the `--model` argument to select `vit_base`, `vit_large`, or `vit_huge`.
- **Hyperparameters:** Adjust learning rate, epochs, and batch size via command-line arguments in `train.py`.
- **Data Preprocessing:** Modify image size, normalization, or dataset selection in `data/data_loader.py` as needed.

## Sample Input and Expected Output

- **Input:** A sample image (e.g., a CIFAR-10 image) resized to 224×224.
- **Output:** The model outputs a predicted class index. Example:
  ```
  Predicted class index: 3
  ```

## Requirements and Installation

- Python 3.x
- PyTorch
- Torchvision
- Pillow

Install dependencies with:

```bash
pip install torch torchvision pillow
```

## Running the Entire Pipeline

1. **Training:**
   ```
   python train.py --model vit_base --epochs 10 --lr 0.001 --batch_size 64
   ```

2. **Evaluation:**
   ```
   python eval.py --model vit_base --checkpoint best_model.pth --batch_size 64
   ```

3. **Testing/Inference:**
   ```
   python test.py --model vit_base --image path_to_image.jpg
   ```

Switch between ViT-Base, ViT-Large, and ViT-Huge by changing the `--model` argument.

## Detailed Process Explanation

### Training Process

- **Command Structure:** Use `--model`, `--epochs`, `--lr`, and `--batch_size` arguments.
- **Examples:**
  - ViT-Base: `python train.py --model vit_base --epochs 10 --lr 0.001 --batch_size 64`
  - ViT-Large: `python train.py --model vit_large --epochs 10 --lr 0.001 --batch_size 64`
  - ViT-Huge: `python train.py --model vit_huge --epochs 10 --lr 0.001 --batch_size 64`
- **Process:** Loads dataset, instantiates model, trains using Adam optimizer, and saves best model.

### Evaluation Process

- **Command Structure:** Use `--model`, `--checkpoint`, and `--batch_size` arguments.
- **Example:** `python eval.py --model vit_base --checkpoint best_model.pth --batch_size 64`
- **Process:** Instantiates model, loads weights, and computes accuracy on validation set.

### Testing (Inference) Process

- **Command Structure:** Use `--model` and `--image` arguments.
- **Example:** `python test.py --model vit_base --image path_to_image.jpg`
- **Process:** Loads and preprocesses image, instantiates model, and outputs predicted class.

### Switching Between Variants and Modifying Hyperparameters

- **Switching Variants:** Use `--model` flag with `vit_base`, `vit_large`, or `vit_huge`.
- **Modifying Hyperparameters:** Adjust epochs, learning rate, and batch size in `train.py`. For deeper changes, modify factory functions in `models/vit.py`.