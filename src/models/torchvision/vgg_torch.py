from functools import partial
from typing import Any, cast, Dict, List, Type, Union

import torch
import torch.nn as nn

from .._api import register_model


class VGG(nn.Module):
    def __init__(
            self,
            features: nn.Module, 
            num_classes: int = 1000,
            init_weights: bool = True, 
            dropout: float = 0.5,
            act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
            **kwargs,
    ) -> None:
        super().__init__()
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self._act_layer = act_layer
        
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            act_layer(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(
        cfg: List[Union[str, int]], 
        batch_norm: bool = False,
        act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
        **kwargs,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act_layer()]
            else:
                layers += [conv2d, act_layer()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, **kwargs), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("A", False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    """VGG-11-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("A", True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("B", False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("B", True, **kwargs)


@register_model("vgg16_torch")
def vgg16(**kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("D", False, **kwargs)


@register_model("vgg16_bn_torch")
def vgg16_bn(**kwargs: Any) -> VGG:
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("D", True, **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("E", False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    """VGG-19_BN from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.
    """
    return _vgg("E", True, **kwargs)
