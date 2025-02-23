import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projects them to a latent embedding space.
    This is implemented using a convolutional layer with kernel size and stride equal to the patch size.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Using a Conv2d layer to both extract patches and linearly project them.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        x: tensor of shape (batch_size, in_channels, img_size, img_size)
        Returns: tensor of shape (batch_size, num_patches, embed_dim)
        """
        x = self.proj(x)              # (batch_size, embed_dim, H', W') where H'=W'=img_size/patch_size
        x = x.flatten(2)              # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (batch_size, num_patches, embed_dim)
        return x
