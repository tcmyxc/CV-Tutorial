"""
ref: https://github.com/moskomule/senet.pytorch
"""

import torch.nn as nn
from functools import partial
from torchvision.models import ResNet
from layers import SequecialHGELUV4B
from ._api import register_model

SequecialHGELUV4 = partial(SequecialHGELUV4B, r=4)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SequecialHGELUV4(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SequecialHGELUV4(planes * 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1_000):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes=1_000):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes=1_000):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes=1_000):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes=1_000):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarSEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, act_layer=partial(nn.ReLU, inplace=True)):
        super(CifarSEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = act_layer()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = act_layer()
        self.se = SequecialHGELUV4(planes)
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
        out = out + self.se(out)

        out += residual
        out = self.relu2(out)

        return out


class CifarSEResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10, act_layer=None,):
        super(CifarSEResNet, self).__init__()
        if act_layer is None:
            act_layer = partial(nn.ReLU, inplace=True)
        self.inplane = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = act_layer()
        self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, act_layer=act_layer)
        self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, act_layer=act_layer)
        self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, act_layer=act_layer)
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride, act_layer):
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


@register_model("sehgelu_resnet14_v4")
def sehgelu_resnet14(**kwargs):
    model = CifarSEResNet(CifarSEBasicBlock, 2, **kwargs)
    return model


@register_model("sehgelu_resnet20_v4")
def sehgelu_resnet20(**kwargs):
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model


@register_model("sehgelu_resnet32_v4")
def sehgelu_resnet32(**kwargs):
    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)
    return model


@register_model("sehgelu_resnet56_v4")
def sehgelu_resnet56(**kwargs):
    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model


@register_model("sehgelu_resnet110_v4")
def sehgelu_resnet110(**kwargs):
    model = CifarSEResNet(CifarSEBasicBlock, 18, **kwargs)
    return model
