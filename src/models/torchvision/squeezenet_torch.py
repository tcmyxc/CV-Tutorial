from functools import partial
from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.nn.init as init

from .._api import register_model


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int, act_layer) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = act_layer()
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = act_layer()
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )


class SqueezeNet(nn.Module):
    def __init__(
            self, 
            version: str = "1_0", 
            num_classes: int = 1000, 
            dropout: float = 0.5,
            act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
            **kwargs: Any,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self.num_classes = num_classes
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                act_layer(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64, act_layer),
                Fire(128, 16, 64, 64, act_layer),
                Fire(128, 32, 128, 128, act_layer),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128, act_layer),
                Fire(256, 48, 192, 192, act_layer),
                Fire(384, 48, 192, 192, act_layer),
                Fire(384, 64, 256, 256, act_layer),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256, act_layer),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                act_layer(),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, act_layer),
                Fire(128, 16, 64, 64, act_layer),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, act_layer),
                Fire(256, 32, 128, 128, act_layer),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, act_layer),
                Fire(384, 48, 192, 192, act_layer),
                Fire(384, 64, 256, 256, act_layer),
                Fire(512, 64, 256, 256, act_layer),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout), final_conv, act_layer(), nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(
    version: str,
    **kwargs: Any,
) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    return model


@register_model("squeezenet1_0_torch")
def squeezenet1_0(**kwargs: Any) -> SqueezeNet:
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    return _squeezenet("1_0", **kwargs)


@register_model("squeezenet1_1_torch")
def squeezenet1_1(**kwargs: Any) -> SqueezeNet:
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.

    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    return _squeezenet("1_1", **kwargs)
