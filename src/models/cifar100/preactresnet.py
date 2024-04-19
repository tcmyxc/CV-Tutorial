"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch.nn as nn
import torch.nn.functional as F

from models._api import register_model

class PreActBasic(nn.Module):

    expansion = 1
    def __init__(self, in_channels, out_channels, stride, act_layer=nn.GELU,):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut


class PreActBottleNeck(nn.Module):

    expansion = 4
    def __init__(self, in_channels, out_channels, stride, act_layer=nn.GELU,):
        super().__init__()

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            act_layer(),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            act_layer(),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return res + shortcut

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, in_channels=3, num_classes=100, act_layer=nn.GELU,):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        self.stage1 = self._make_layers(block, num_block[0], 64,  1, act_layer)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2, act_layer)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2, act_layer)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2, act_layer)
        
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.act_layer = act_layer()

        self.linear = nn.Linear(self.input_channels, num_classes)

    def _make_layers(self, block, block_num, out_channels, stride, act_layer):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride, act_layer))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1, act_layer))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.bn(x)
        x = self.act_layer(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


@register_model("preactresnet18_c100")
def preactresnet18(**kwargs):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], **kwargs)


def preactresnet34(**kwargs):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], **kwargs)


@register_model("preactresnet50_c100")
def preactresnet50(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], **kwargs)


def preactresnet101(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], **kwargs)


def preactresnet152(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], **kwargs)
