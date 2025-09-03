import torch
import sys

print(f"Python версия: {sys.version}")
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"GPU устройство: {torch.cuda.get_device_name(0)}")
