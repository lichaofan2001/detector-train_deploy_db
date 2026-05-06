import torch
import torchvision
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print("Qwen3_5ForConditionalGeneration and AutoProcessor imported successfully!")