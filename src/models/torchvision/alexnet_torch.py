from functools import partial
from typing import Any, Type

import torch
import torch.nn as nn

from .._api import register_model


class AlexNet(nn.Module):
    def __init__(
            self, 
            num_classes: int = 1000, 
            dropout: float = 0.5,
            act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
            **kwargs,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_layer(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            act_layer(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@register_model("alexnet_torch")
def alexnet(**kwargs: Any) -> AlexNet:
    """AlexNet model architecture from `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`__.

        AlexNet was originally introduced in the `ImageNet Classification with
        Deep Convolutional Neural Networks
        <https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>`__
        paper. Our implementation is based instead on the "One weird trick"
        paper above.
    """

    model = AlexNet(**kwargs)

    return model
