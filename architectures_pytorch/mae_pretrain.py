"""MAE pretraining utility.
Uses vit_pytorch.MAE to pretrain Vision Transformer encoder on unlabeled images.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from vit_pytorch import ViT, MAE

from architectures_pytorch.helpers.constants import hyperparameters, mae_weights_path

H = hyperparameters["vision_transformer"]


def build_mae_model(img_size: int, patch_size: int, in_channels: int = 1) -> MAE:
    encoder = ViT(
        image_size=img_size,
        patch_size=patch_size,
        num_classes=H["projection_dim"],
        dim=H["projection_dim"],
        depth=H["transformer_layers"],
        heads=H["num_heads"],
        mlp_dim=H["projection_dim"] * 4,
        channels=in_channels,
        dropout=H["dropout"],
        pool="cls",
    )
    mae = MAE(
        encoder=encoder,
        masking_ratio=H["mae_mask_ratio"],
        decoder_dim=H["mae_decoder_dim"],
        decoder_depth=H["mae_decoder_depth"],
    )
    return mae


def pretrain_mae(images: torch.Tensor | np.ndarray, epochs: int | None = None, batch_size: int | None = None, device: str | torch.device = "cuda") -> None:
    """Pretrain MAE on provided *unlabeled* images array of shape (N, H, W) or (N, C, H, W)."""
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    if images.ndim == 3:  # (N, H, W)
        images = images.unsqueeze(1)  # add channel dim

    epochs = epochs or H["mae_epochs_pretrain"]
    batch_size = batch_size or H["batch_size"]

    mae = build_mae_model(H["image_size"], H["patch_size"], in_channels=1).to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=H["learning_rate"], weight_decay=H["weight_decay"])

    ds = TensorDataset(images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    mae.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x,) in tqdm(dl, desc=f"MAE pretrain epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            loss = mae(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss {total_loss/len(dl):.4f}")

    # Save encoder weights
    encoder = mae.encoder
    enc_state = encoder.state_dict()
    # add 'vit.' prefix so it matches VisionTransformer submodule name
    prefixed_state = {f'vit.{k}': v for k, v in enc_state.items()}
    torch.save(prefixed_state, mae_weights_path)
    print(f"MAE encoder weights saved to {mae_weights_path} with 'vit.' prefix")
