import os
import wandb
from architectures_pytorch.helpers.constants import hyperparameters
from architectures_pytorch.helpers.constants import selected_model
from architectures_pytorch.helpers.constants import threshold

hyperparameters = hyperparameters[selected_model]


def initialize_wandb():
    # Set wandb to offline mode
    os.environ['WANDB_MODE'] = 'offline'
    
    if selected_model == "vision_transformer":
        wandb.init(project=f"{selected_model}", entity="ilya3009chernoglazov-cryptanium-fund",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "vit":
        wandb.init(project=f"{selected_model}", entity="ilya3009chernoglazov-cryptanium-fund",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "layer_norm_eps": hyperparameters["layer_norm_eps"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "vit_small":
        wandb.init(project=f"{selected_model}", entity="ilya3009chernoglazov-cryptanium-fund",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "threshold": f"0.{threshold}",
                   })
    elif selected_model == "cct":
        wandb.init(project=f"{selected_model}", entity="ilya3009chernoglazov-cryptanium-fund",
                   config={
                       "model": f"{selected_model}",
                       "learning_rate": hyperparameters["learning_rate_type"],
                       "epochs": hyperparameters["num_epochs"],
                       "batch_size": hyperparameters["batch_size"],
                       "weight_decay": hyperparameters["weight_decay"],
                       "image_size": hyperparameters["image_size"],
                       "projection_dim": hyperparameters["projection_dim"],
                       "num_heads": hyperparameters["num_heads"],
                       "patch_size": hyperparameters["patch_size"],
                       "transformer_layers": hyperparameters["transformer_layers"],
                       "num_conv_layers": hyperparameters["num_conv_layers"],
                       "conv_kernel_size": hyperparameters["conv_kernel_size"],
                       "pool_kernel_size": hyperparameters["pool_kernel_size"],
                       "pool_stride": hyperparameters["pool_stride"],
                       "threshold": f"0.{threshold}",
                   })
