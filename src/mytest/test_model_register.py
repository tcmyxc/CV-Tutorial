import argparse
import torch
from torchsummary import summary
from ptflops import get_model_complexity_info
from models import list_models, load_model

print(list_models())

parser = argparse.ArgumentParser()
# 模型架构
parser.add_argument("--model", default="resnet10t", type=str, help="model name")
parser.add_argument(
    "--model_lib",
    default="custom", type=str,
    choices=["torch", "timm", "cifar100", "qt"],
    help="model library",
)
parser.add_argument(
    "--act_layer",
    default="gelu", type=str,
    help="activation function",
)

args = parser.parse_args()

model = load_model(args, num_classes=100)
print(model)
input_size = (3, 32, 32)
summary(model, input_size=input_size, batch_size=8, device="cpu")

# with torch.cuda.device(0):
macs, params = get_model_complexity_info(
    model,
    input_size,
    as_strings=True,
    print_per_layer_stat=False,
    verbose=False,
    # flops_units="GMac",
)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))
