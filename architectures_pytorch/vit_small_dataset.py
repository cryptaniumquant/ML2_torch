import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vit_pytorch.vit_for_small_dataset import ViT

import architectures_pytorch.helpers.constants as constants

# Hyperparameters specific to the small-dataset ViT
hyperparameters = constants.hyperparameters["vit_small"]


class VisionTransformerSmall(nn.Module):
    """Wrapper around vit_pytorch.vit_for_small_dataset.ViT that adds a custom MLP head
    compatible with the training pipeline (returns logits for num_classes)."""

    def __init__(
        self,
        img_size: int = hyperparameters["image_size"],
        patch_size: int = hyperparameters["patch_size"],
        in_channels: int = 1,
        num_classes: int = hyperparameters["num_classes"],
        embed_dim: int = hyperparameters["projection_dim"],
        depth: int = hyperparameters["transformer_layers"],
        num_heads: int = hyperparameters["num_heads"],
        dropout: float = hyperparameters["dropout"],
    ):
        super().__init__()

        # Base ViT backbone (with convolutional stem optimised for small images)
        self.vit = ViT(
            image_size=img_size,
            patch_size=patch_size,
            num_classes=embed_dim,  # use embed_dim as intermediate representation dimension
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=embed_dim * 4,
            channels=in_channels,
            dropout=dropout,
            pool="cls",
        )

        # Classification head – similar style to other models in the repo
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
            nn.Linear(hyperparameters["mlp_head_units"][1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.vit(x)
        x = self.head(x)
        return x


def create_vit_small_classifier() -> nn.Module:
    """Factory function used by training script."""
    return VisionTransformerSmall()


def get_vit_small_model(load_weights: bool = False, weights_path: str | None = None) -> nn.Module:
    """Return a ViT-Small model, optionally loading pretrained weights."""
    print("Getting the PyTorch ViT Small-Dataset model...")
    model = create_vit_small_classifier()

    if load_weights and weights_path:
        model.load_state_dict(torch.load(weights_path))

    return model


# Data-augmentation pipeline – mirrors the standard ViT transforms used elsewhere
# (you can tweak these later if required)
data_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5
        ),
    ]
)
