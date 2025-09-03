from architectures_pytorch.helpers.constants import selected_model
from architectures_pytorch.vision_transformer import get_vit_model
from architectures_pytorch.vit import get_vit


def get_model():
    if selected_model == "vision_transformer":
        return get_vit_model()
    elif selected_model == "vit":
        return get_vit()
