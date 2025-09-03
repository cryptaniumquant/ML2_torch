import torch
import torch.nn as nn
import torchvision.transforms as transforms
from vit_pytorch.cct import CCT, Tokenizer
from einops import rearrange
from einops.layers.torch import Rearrange

import architectures_pytorch.helpers.constants as constants

hyperparameters = constants.hyperparameters["cct"]  # Using CCT-specific hyperparameters

# Создаем собственный токенизатор для одноканальных изображений
class CustomTokenizer(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=1,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False
    ):
        super().__init__()
        
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(
                    n_filter_list[i], n_filter_list[i + 1],
                    kernel_size=(kernel_size, kernel_size),
                    stride=(stride, stride),
                    padding=(padding, padding),
                    bias=conv_bias
                ),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(pooling_kernel_size, pooling_stride, pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ]
        )

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=1, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class CompactConvTransformer(nn.Module):
    def __init__(self,
                 img_size=hyperparameters["image_size"],
                 patch_size=hyperparameters["patch_size"],
                 in_channels=1,
                 num_classes=hyperparameters["num_classes"],
                 embed_dim=hyperparameters["projection_dim"],
                 depth=hyperparameters["transformer_layers"],
                 num_heads=hyperparameters["num_heads"],
                 dropout=hyperparameters["dropout"],
                 mlp_ratio=4,
                 num_conv_layers=hyperparameters["num_conv_layers"]):
        super().__init__()
        
        # Создаем собственный токенизатор для одноканальных изображений
        self.tokenizer = CustomTokenizer(
            kernel_size=hyperparameters["conv_kernel_size"],
            stride=1,
            padding=1,
            pooling_kernel_size=hyperparameters["pool_kernel_size"],
            pooling_stride=hyperparameters["pool_stride"],
            pooling_padding=1,
            n_conv_layers=num_conv_layers,
            n_input_channels=in_channels,
            n_output_channels=embed_dim,
            in_planes=embed_dim,
            activation=nn.GELU,
            max_pool=True
        )
        
        # Определяем длину последовательности после токенизации
        seq_length = self.tokenizer.sequence_length(in_channels, img_size, img_size)
        
        # Создаем трансформер
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Позиционное кодирование
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # MLP head with improved architecture similar to the ViT implementation
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
        # Токенизация входных данных
        x = self.tokenizer(x)
        
        # Добавляем позиционное кодирование
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Добавляем CLS токен
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Пропускаем через трансформер
        x = self.transformer(x)
        
        # Берем только CLS токен для классификации
        x = x[:, 0]
        
        # Пропускаем через MLP head
        x = self.head(x)
        return x

def create_cct_classifier():
    model = CompactConvTransformer()
    return model

def get_cct_model(load_weights=False, weights_path=None):
    print("Getting the PyTorch CCT model...")
    model = create_cct_classifier()
    
    if load_weights and weights_path:
        model.load_state_dict(torch.load(weights_path))
    
    return model

# Data augmentation transforms - enhanced for better generalization
# CCT can benefit from similar transforms as ViT
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
])
