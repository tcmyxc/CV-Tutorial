import torch
import torch.nn as nn
from typing import Any
from functools import partial

from models._api import register_model


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, act_layer = partial(nn.ReLU, inplace=True)) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            act_layer(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            act_layer(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_model("alexnet_c100")
def alexnet(**kwargs: Any) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    return model
