import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vit_pytorch import ViT

import architectures_pytorch.helpers.constants as constants

hyperparameters = constants.hyperparameters["vision_transformer"]

class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=hyperparameters["image_size"],
                 patch_size=hyperparameters["patch_size"],
                 in_channels=1,
                 num_classes=hyperparameters["num_classes"],
                 embed_dim=hyperparameters["projection_dim"],
                 depth=hyperparameters["transformer_layers"],
                 num_heads=hyperparameters["num_heads"],
                 dropout=hyperparameters["dropout"]):
        super().__init__()
        
        self.vit = ViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=embed_dim,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=embed_dim * 4,
            dropout=dropout,
            channels=in_channels
        )
        
        # MLP head with improved architecture for better gradient flow
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hyperparameters["mlp_head_units"][0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hyperparameters["mlp_head_units"][0]),
            nn.Linear(hyperparameters["mlp_head_units"][0], hyperparameters["mlp_head_units"][1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hyperparameters["mlp_head_units"][1]),
            nn.Linear(hyperparameters["mlp_head_units"][1], num_classes)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.head(x)
        return x

def create_vit_classifier():
    model = VisionTransformer()
    return model

def get_vit_model(load_weights=False, weights_path=None):
    print("Getting the PyTorch ViT model...")
    model = create_vit_classifier()
    
    if load_weights and weights_path:
        state_dict = torch.load(weights_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading ViT weights: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys when loading ViT weights: {unexpected}")
    
    return model

# Data augmentation transforms - enhanced for better generalization
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
])