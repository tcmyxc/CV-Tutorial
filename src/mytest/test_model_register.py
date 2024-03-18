import argparse
import torch
from torchsummary import summary

from models import list_models, load_model

print(list_models())

parser = argparse.ArgumentParser()
# 模型架构
parser.add_argument("--model", default="resnet50_A1", type=str, help="model name")
parser.add_argument(
    "--model_lib",
    default="custom", type=str,
    choices=["torch", "timm", "cifar100", "qt"],
    help="model library",
)
parser.add_argument(
    "--act_layer",
    default="hgelu", type=str,
    help="activation function",
)

args = parser.parse_args()

model = load_model(args, num_classes=100)
print(model)
summary(model, input_size=(3, 32, 32), batch_size=8, device="cpu")

print(model(torch.rand(8, 3, 32, 32)).shape)
