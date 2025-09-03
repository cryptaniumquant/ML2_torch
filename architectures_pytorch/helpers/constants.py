from architectures_pytorch.helpers.warmup_cosine import WarmUpCosineScheduler
from architectures_pytorch.helpers.one_cycle import OneCycleLRScheduler

DB_CONFIG = {
    'username': '',
    'password': '',
    'host': '',
    'port': '', 
    'database': ''
}

DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

selected_model = "vision_transformer"
#selected_model = "vit"
#selected_model = "cct"
#selected_model = "vit_small"


direction = "3"  # long or short
coin_list = ['ETHUSDT1M']
threshold = f'03-015_{direction}_{coin_list}'

enable_mae_pretrain = True  # set to False to skip MAE pretraining
mae_weights_path = f'models/{coin_list[0]}_mae_encoder_50.pth'

hyperparameters = {
    "vision_transformer": {
        "learning_rate_type": "WarmUpCosine",
        "learning_rate": 0.001,  # Further reduced learning rate for better convergence
        "weight_decay": 0.0001,  # Minimal weight decay to prevent overfitting
        "batch_size": 128,         # Even smaller batch size for better gradient updates
        "num_epochs": 150,        # More epochs for better convergence
        "image_size": 50,
        "patch_size": 5,          # Smaller patch size for more detailed feature extraction
        "projection_dim": 64,    # Further increased projection dimension
        "num_heads": 4,          # More attention heads
        "transformer_layers": 8, # More transformer layers for complex pattern recognition
        "mlp_head_units": [2048, 1024, 512],  # Added an extra layer for better feature extraction
        "num_classes": 3,
        "layer_norm_eps": 1e-6,
        "dropout": 0.1,          # Slightly reduced dropout,
        # MAE pretraining params
        "mae_mask_ratio": 0.75,
        "mae_decoder_dim": 64,
        "mae_decoder_depth": 6,
        "mae_epochs_pretrain": 50       
    },
    "vit": {
        "learning_rate_type": "WarmUpCosine",
        "learning_rate": 0.0003,
        "weight_decay": 0.001,
        "batch_size": 64,
        "num_epochs": 50,
        "image_size": 50,
        "patch_size": 10,
        "projection_dim": 128,
        "num_heads": 8,
        "transformer_layers": 6,
        "mlp_head_units": [1024, 512],
        "num_classes": 2,
        "layer_norm_eps": 1e-6,
        "dropout": 0.2,
        "warmup_steps": 500
    },
    "cct": {
        "learning_rate_type": "WarmUpCosine",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 150,
        "image_size": 50,
        "patch_size": 3,          # Smaller patch size for CCT's convolutional tokenizer
        "projection_dim": 64,
        "num_heads": 4,
        "transformer_layers": 8,
        "mlp_head_units": [2048, 1024, 512],
        "num_classes": 3,
        "layer_norm_eps": 1e-6,
        "dropout": 0.1,
        "num_conv_layers": 2,     # Number of convolutional layers in tokenizer
        "conv_kernel_size": 3,     # Size of convolutional kernel in tokenizer
        "pool_kernel_size": 3,     # Size of pooling kernel in tokenizer
        "pool_stride": 2          # Stride of pooling in tokenizer
    },
    "vit_small": {
        "learning_rate_type": "WarmUpCosine",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "num_epochs": 150,
        "image_size": 50,
        "patch_size": 5,          # smaller patch size
        "projection_dim": 64,
        "num_heads": 4,
        "transformer_layers": 8,
        "mlp_head_units": [256, 128],
        "num_classes": 3,
        "layer_norm_eps": 1e-6,
        "dropout": 0.1
    }
}

# Calculate derived parameters for vision transformer
hyperparameters["vision_transformer"]["num_patches"] = (
    hyperparameters["vision_transformer"]["image_size"] // hyperparameters["vision_transformer"]["patch_size"]) ** 2
hyperparameters["vision_transformer"]["transformer_units"] = [
    hyperparameters["vision_transformer"]["projection_dim"] * 2,
    hyperparameters["vision_transformer"]["projection_dim"],
]
hyperparameters["vision_transformer"]["input_shape"] = (
    1,  # channels
    hyperparameters["vision_transformer"]["image_size"],
    hyperparameters["vision_transformer"]["image_size"]
)

# Calculate derived parameters for CCT
hyperparameters["cct"]["num_patches"] = (
    hyperparameters["cct"]["image_size"] // hyperparameters["cct"]["patch_size"]) ** 2
hyperparameters["cct"]["transformer_units"] = [
    hyperparameters["cct"]["projection_dim"] * 2,
    hyperparameters["cct"]["projection_dim"],
]
hyperparameters["cct"]["input_shape"] = (
    1,  # channels
    hyperparameters["cct"]["image_size"],
    hyperparameters["cct"]["image_size"]
)

# Calculate derived parameters for vit_small
hyperparameters["vit_small"]["num_patches"] = (
    hyperparameters["vit_small"]["image_size"] // hyperparameters["vit_small"]["patch_size"]) ** 2
hyperparameters["vit_small"]["transformer_units"] = [
    hyperparameters["vit_small"]["projection_dim"] * 2,
    hyperparameters["vit_small"]["projection_dim"],
]
hyperparameters["vit_small"]["input_shape"] = (
    1,
    hyperparameters["vit_small"]["image_size"],
    hyperparameters["vit_small"]["image_size"]
)
