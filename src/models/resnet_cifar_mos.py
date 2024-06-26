"""
ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/baseline.py
-------------------------------------------------------------------------------
ResNet for CIFAR dataset proposed in He+15, p 7. and
https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
"""

import torch.nn as nn
from functools import partial

from ._api import register_model


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, act_layer=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = act_layer()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = act_layer()
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = nn.Identity()
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out


class PreActBasicBlock(BasicBlock):
    def __init__(self, inplanes, planes, stride, act_layer=None):
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self._act_layer = act_layer

        super().__init__(inplanes, planes, stride, act_layer)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False))
        else:
            self.downsample = nn.Identity()
        self.bn1 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, act_layer=None):
        super().__init__()
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self._act_layer = act_layer

        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = act_layer()
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        act_layer = self._act_layer

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, act_layer))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class PreActResNet(ResNet):
    def __init__(self, block, n_size, num_classes=10, act_layer=None):
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self._act_layer = act_layer

        super().__init__(block, n_size, num_classes, act_layer)

        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@register_model("resnet8_mos")
def resnet8(**kwargs):
    model = ResNet(BasicBlock, 1, **kwargs)
    return model


@register_model("resnet14_mos")
def resnet14(**kwargs):
    model = ResNet(BasicBlock, 2, **kwargs)
    return model


@register_model("resnet20_mos")
def resnet20(**kwargs):
    model = ResNet(BasicBlock, 3, **kwargs)
    return model


@register_model("resnet32_mos")
def resnet32(**kwargs):
    model = ResNet(BasicBlock, 5, **kwargs)
    return model


@register_model("resnet56_mos")
def resnet56(**kwargs):
    model = ResNet(BasicBlock, 9, **kwargs)
    return model


@register_model("resnet110_mos")
def resnet110(**kwargs):
    model = ResNet(BasicBlock, 18, **kwargs)
    return model


def preact_resnet20(**kwargs):
    model = PreActResNet(PreActBasicBlock, 3, **kwargs)
    return model


def preact_resnet32(**kwargs):
    model = PreActResNet(PreActBasicBlock, 5, **kwargs)
    return model


def preact_resnet56(**kwargs):
    model = PreActResNet(PreActBasicBlock, 9, **kwargs)
    return model


def preact_resnet110(**kwargs):
    model = PreActResNet(PreActBasicBlock, 18, **kwargs)
    return model
