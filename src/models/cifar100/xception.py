"""xception in pytorch


[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""

import torch.nn as nn
from functools import partial

from models._api import register_model


class SeperableConv2d(nn.Module):

    # ***Figure 4. An “extreme” version of our Inception module,
    # with one spatial convolution per output channel of the 1x1
    # convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class EntryFlow(nn.Module):

    def __init__(self, act_layer=None,):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            act_layer()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act_layer()
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            act_layer(),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            act_layer(),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            act_layer(),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        # no downsampling
        self.conv5_residual = nn.Sequential(
            act_layer(),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            act_layer(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        # no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x


class MiddleFLowBlock(nn.Module):

    def __init__(self, act_layer=None,):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            act_layer(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            act_layer(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            act_layer(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual


class MiddleFlow(nn.Module):
    def __init__(self, block, act_layer):
        super().__init__()

        # """then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8, act_layer)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times, act_layer):
        flows = []
        for i in range(times):
            flows.append(block(act_layer))

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self, act_layer=None,):
        super().__init__()
        self.residual = nn.Sequential(
            act_layer(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            act_layer(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            act_layer(),
            SeperableConv2d(1536, 2048, 3, padding=1),
            nn.BatchNorm2d(2048),
            act_layer()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        output = self.avgpool(output)

        return output


class Xception(nn.Module):

    def __init__(self, block, num_classes=100, act_layer=None, **kwargs):
        super().__init__()
        act_layer = act_layer or partial(nn.ReLU, inplace=True)
        self.entry_flow = EntryFlow(act_layer)
        self.middel_flow = MiddleFlow(block, act_layer)
        self.exit_flow = ExitFLow(act_layer)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@register_model("xception_c100")
def xception(**kwargs):
    return Xception(MiddleFLowBlock, **kwargs)
