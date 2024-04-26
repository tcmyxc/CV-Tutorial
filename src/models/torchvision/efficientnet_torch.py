import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Type, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from .._api import register_model
from torchvision.models._utils import _make_divisible


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            act_layer,
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        # activation_layer = nn.SiLU
        activation_layer = act_layer

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                # activation=partial(nn.SiLU, inplace=True)
                activation=activation_layer,
            )
        )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf: FusedMBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            act_layer,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        # activation_layer = nn.SiLU
        activation_layer = act_layer

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
            act_layer: Type[nn.Module] = partial(nn.ReLU, inplace=True),
            **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class

        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                # activation_layer=nn.SiLU
                activation_layer=act_layer,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer, act_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                # activation_layer=nn.SiLU,
                activation_layer=act_layer,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    **kwargs: Any,
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)
    return model


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(MBConvConfig, width_mult=kwargs.pop("width_mult"), depth_mult=kwargs.pop("depth_mult"))
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def efficientnet_b0(**kwargs: Any) -> EfficientNet:
    """EfficientNet B0 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return _efficientnet(inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, **kwargs)


def efficientnet_b1(**kwargs: Any) -> EfficientNet:
    """EfficientNet B1 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return _efficientnet(inverted_residual_setting, kwargs.pop("dropout", 0.2), last_channel, **kwargs)


def efficientnet_b2(**kwargs: Any) -> EfficientNet:
    """EfficientNet B2 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return _efficientnet(inverted_residual_setting, kwargs.pop("dropout", 0.3), last_channel, **kwargs)


def efficientnet_b3(**kwargs: Any) -> EfficientNet:
    """EfficientNet B3 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return _efficientnet(inverted_residual_setting, kwargs.pop("dropout", 0.3), last_channel, **kwargs)


def efficientnet_b4(**kwargs: Any) -> EfficientNet:
    """EfficientNet B4 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        **kwargs,
    )


def efficientnet_b5(**kwargs: Any) -> EfficientNet:
    """EfficientNet B5 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(**kwargs: Any) -> EfficientNet:
    """EfficientNet B6 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(**kwargs: Any) -> EfficientNet:
    """EfficientNet B7 model architecture from the `EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks <https://arxiv.org/abs/1905.11946>`_ paper.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.5),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@register_model("efficientnet_v2_s_torch")
def efficientnet_v2_s(**kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-S architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_m(**kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-M architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.3),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )


def efficientnet_v2_l(**kwargs: Any) -> EfficientNet:
    """
    Constructs an EfficientNetV2-L architecture from
    `EfficientNetV2: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """

    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.4),
        last_channel,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        **kwargs,
    )
