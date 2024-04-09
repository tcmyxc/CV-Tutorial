"""PyTorch ResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman
"""
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq

from ._api import register_model


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob: float = 0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=5, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=3, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if (stage_idx == 0 or stage_idx == 3) else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):
    """ResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of ResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This ResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon ResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    ResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision ResNet, NVIDIA ResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as ResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as ResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            avg_down: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = False,
            block_args: Optional[Dict[str, Any]] = None,
            last_stride: int = 1,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            if 'tiered' in stem_type:
                stem_chs = (3 * (stem_width // 4), stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=1, module='act1')]

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_resnet(variant, pretrained: bool = False, **kwargs) -> ResNet:
    return build_model_with_cfg(ResNet, variant, pretrained, **kwargs)


@register_model("resnet10t_timm")
def resnet10t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-10-T model.
    """
    model_args = dict(block=BasicBlock, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet10t', pretrained, **dict(model_args, **kwargs))



def resnet14t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-14-T model.
    """
    model_args = dict(block=Bottleneck, layers=[1, 1, 1, 1], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet14t', pretrained, **dict(model_args, **kwargs))



def resnet18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2])
    return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))


@register_model("resnet18dp_timm")
def resnet18dp(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], drop_path_rate=0.3)
    return _create_resnet('resnet18', pretrained, **dict(model_args, **kwargs))


def resnet18d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet18d', pretrained, **dict(model_args, **kwargs))



def resnet34(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3])
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))



@register_model("resnet34dp_timm")
def resnet34dp(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], drop_path_rate=0.4)
    return _create_resnet('resnet34', pretrained, **dict(model_args, **kwargs))


def resnet34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet34d', pretrained, **dict(model_args, **kwargs))



def resnet26(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2])
    return _create_resnet('resnet26', pretrained, **dict(model_args, **kwargs))



def resnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-T model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet26t', pretrained, **dict(model_args, **kwargs))



def resnet26d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-26-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet26d', pretrained, **dict(model_args, **kwargs))


@register_model("resnet50_timm")
def resnet50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3])
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))


@register_model("resnet50dp_timm")
def resnet50dp(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], drop_path_rate=0.5)
    return _create_resnet('resnet50', pretrained, **dict(model_args, **kwargs))


def resnet50c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet50c', pretrained, **dict(model_args, **kwargs))


@register_model("resnet50d_timm")
def resnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], avg_down=True)
    return _create_resnet('resnet50d', pretrained, **dict(model_args, **kwargs))


@register_model("resnet50d_dp_timm")
def resnet50d_dp(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], avg_down=True, drop_path_rate=0.5)
    return _create_resnet('resnet50d', pretrained, **dict(model_args, **kwargs))



def resnet50s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet50s', pretrained, **dict(model_args, **kwargs))



def resnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-T model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep_tiered', avg_down=True)
    return _create_resnet('resnet50t', pretrained, **dict(model_args, **kwargs))



def resnet101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3])
    return _create_resnet('resnet101', pretrained, **dict(model_args, **kwargs))



def resnet101c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet101c', pretrained, **dict(model_args, **kwargs))



def resnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet101d', pretrained, **dict(model_args, **kwargs))



def resnet101s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet101s', pretrained, **dict(model_args, **kwargs))



def resnet152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3])
    return _create_resnet('resnet152', pretrained, **dict(model_args, **kwargs))



def resnet152c(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-C model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep')
    return _create_resnet('resnet152c', pretrained, **dict(model_args, **kwargs))



def resnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet152d', pretrained, **dict(model_args, **kwargs))



def resnet152s(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-152-S model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], stem_width=64, stem_type='deep')
    return _create_resnet('resnet152s', pretrained, **dict(model_args, **kwargs))



def resnet200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200 model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3])
    return _create_resnet('resnet200', pretrained, **dict(model_args, **kwargs))



def resnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnet200d', pretrained, **dict(model_args, **kwargs))



def wide_resnet50_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-50-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], base_width=128)
    return _create_resnet('wide_resnet50_2', pretrained, **dict(model_args, **kwargs))



def wide_resnet101_2(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a Wide ResNet-101-2 model.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], base_width=128)
    return _create_resnet('wide_resnet101_2', pretrained, **dict(model_args, **kwargs))



def resnet50_gn(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model w/ GroupNorm
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], norm_layer='groupnorm')
    return _create_resnet('resnet50_gn', pretrained, **dict(model_args, **kwargs))



def resnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50-32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext50_32x4d', pretrained, **dict(model_args, **kwargs))



def resnext50d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt50d-32x4d model. ResNext50 w/ deep stem & avg pool downsample
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  cardinality=32, base_width=4,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnext50d_32x4d', pretrained, **dict(model_args, **kwargs))



def resnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4)
    return _create_resnet('resnext101_32x4d', pretrained, **dict(model_args, **kwargs))



def resnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x8d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8)
    return _create_resnet('resnext101_32x8d', pretrained, **dict(model_args, **kwargs))



def resnext101_32x16d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x16d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=16)
    return _create_resnet('resnext101_32x16d', pretrained, **dict(model_args, **kwargs))



def resnext101_32x32d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt-101 32x32d model
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=32)
    return _create_resnet('resnext101_32x32d', pretrained, **dict(model_args, **kwargs))



def resnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNeXt101-64x4d model.
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4)
    return _create_resnet('resnext101_64x4d', pretrained, **dict(model_args, **kwargs))



def ecaresnet26t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet26t', pretrained, **dict(model_args, **kwargs))



def ecaresnet50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d', pretrained, **dict(model_args, **kwargs))



def ecaresnet50d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model pruned with eca.
        The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))



def ecaresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNet-50-T model.
    Like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels in the deep stem and ECA attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet50t', pretrained, **dict(model_args, **kwargs))



def ecaresnetlight(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D light model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[1, 1, 11, 3], stem_width=32, avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnetlight', pretrained, **dict(model_args, **kwargs))



def ecaresnet101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with eca.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d', pretrained, **dict(model_args, **kwargs))



def ecaresnet101d_pruned(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model pruned with eca.
       The pruning has been obtained using https://arxiv.org/pdf/2002.08258.pdf
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet101d_pruned', pretrained, pruned=True, **dict(model_args, **kwargs))



def ecaresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet200d', pretrained, **dict(model_args, **kwargs))



def ecaresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with ECA.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnet269d', pretrained, **dict(model_args, **kwargs))



def ecaresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))



def ecaresnext50t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs an ECA-ResNeXt-50-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem. This model replaces SE module with the ECA module
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='eca'))
    return _create_resnet('ecaresnext50t_32x4d', pretrained, **dict(model_args, **kwargs))



def seresnet18(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet18', pretrained, **dict(model_args, **kwargs))



def seresnet34(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet34', pretrained, **dict(model_args, **kwargs))



def seresnet50(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50', pretrained, **dict(model_args, **kwargs))



def seresnet50t(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3],  stem_width=32, stem_type='deep_tiered',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet50t', pretrained, **dict(model_args, **kwargs))



def seresnet101(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 4, 23, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet101', pretrained, **dict(model_args, **kwargs))



def seresnet152(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(block=Bottleneck, layers=[3, 8, 36, 3], block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152', pretrained, **dict(model_args, **kwargs))



def seresnet152d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet152d', pretrained, **dict(model_args, **kwargs))



def seresnet200d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-200-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet200d', pretrained, **dict(model_args, **kwargs))



def seresnet269d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-269-D model with SE attn.
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 30, 48, 8], stem_width=32, stem_type='deep',
        avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnet269d', pretrained, **dict(model_args, **kwargs))



def seresnext26d_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNeXt-26-D model.`
    This is technically a 28 layer ResNet, using the 'D' modifier from Gluon / bag-of-tricks for
    combination of deep stem and avg_pool in downsample.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26d_32x4d', pretrained, **dict(model_args, **kwargs))



def seresnext26t_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE-ResNet-26-T model.
    This is technically a 28 layer ResNet, like a 'D' bag-of-tricks model but with tiered 24, 32, 64 channels
    in the deep stem.
    """
    model_args = dict(
        block=Bottleneck, layers=[2, 2, 2, 2], cardinality=32, base_width=4, stem_width=32,
        stem_type='deep_tiered', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext26t_32x4d', pretrained, **dict(model_args, **kwargs))



def seresnext50_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext50_32x4d', pretrained, **dict(model_args, **kwargs))



def seresnext101_32x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x4d', pretrained, **dict(model_args, **kwargs))



def seresnext101_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_32x8d', pretrained, **dict(model_args, **kwargs))



def seresnext101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101d_32x8d', pretrained, **dict(model_args, **kwargs))



def seresnext101_64x4d(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=64, base_width=4,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnext101_64x4d', pretrained, **dict(model_args, **kwargs))



def senet154(pretrained: bool = False, **kwargs) -> ResNet:
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], cardinality=64, base_width=4, stem_type='deep',
        down_kernel_size=3, block_reduce_first=2, block_args=dict(attn_layer='se'))
    return _create_resnet('senet154', pretrained, **dict(model_args, **kwargs))



def resnetblur18(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-18 model with blur anti-aliasing
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur18', pretrained, **dict(model_args, **kwargs))



def resnetblur50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with blur anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d)
    return _create_resnet('resnetblur50', pretrained, **dict(model_args, **kwargs))



def resnetblur50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur50d', pretrained, **dict(model_args, **kwargs))



def resnetblur101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with blur anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=BlurPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetblur101d', pretrained, **dict(model_args, **kwargs))



def resnetaa34d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-34-D model w/ avgpool anti-aliasing
    """
    model_args = dict(
        block=BasicBlock, layers=[3, 4, 6, 3],  aa_layer=nn.AvgPool2d, stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa34d', pretrained, **dict(model_args, **kwargs))



def resnetaa50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50 model with avgpool anti-aliasing
    """
    model_args = dict(block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d)
    return _create_resnet('resnetaa50', pretrained, **dict(model_args, **kwargs))



def resnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa50d', pretrained, **dict(model_args, **kwargs))



def resnetaa101d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-101-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True)
    return _create_resnet('resnetaa101d', pretrained, **dict(model_args, **kwargs))



def seresnetaa50d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE=ResNet-50-D model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], aa_layer=nn.AvgPool2d,
        stem_width=32, stem_type='deep', avg_down=True, block_args=dict(attn_layer='se'))
    return _create_resnet('seresnetaa50d', pretrained, **dict(model_args, **kwargs))



def seresnextaa101d_32x8d(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], cardinality=32, base_width=8,
        stem_width=32, stem_type='deep', avg_down=True, aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnextaa101d_32x8d', pretrained, **dict(model_args, **kwargs))



def seresnextaa201d_32x8d(pretrained: bool = False, **kwargs):
    """Constructs a SE=ResNeXt-101-D 32x8d model with avgpool anti-aliasing
    """
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 4], cardinality=32, base_width=8,
        stem_width=64, stem_type='deep', avg_down=True, aa_layer=nn.AvgPool2d,
        block_args=dict(attn_layer='se'))
    return _create_resnet('seresnextaa201d_32x8d', pretrained, **dict(model_args, **kwargs))



def resnetrs50(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-50 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 6, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs50', pretrained, **dict(model_args, **kwargs))



def resnetrs101(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-101 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 4, 23, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs101', pretrained, **dict(model_args, **kwargs))



def resnetrs152(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-152 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 8, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs152', pretrained, **dict(model_args, **kwargs))



def resnetrs200(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-200 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[3, 24, 36, 3], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs200', pretrained, **dict(model_args, **kwargs))


def resnetrs270(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-270 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 29, 53, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs270', pretrained, **dict(model_args, **kwargs))



def resnetrs350(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-350 model.
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 36, 72, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs350', pretrained, **dict(model_args, **kwargs))


def resnetrs420(pretrained: bool = False, **kwargs) -> ResNet:
    """Constructs a ResNet-RS-420 model
    Paper: Revisiting ResNets - https://arxiv.org/abs/2103.07579
    Pretrained weights from https://github.com/tensorflow/tpu/tree/bee9c4f6/models/official/resnet/resnet_rs
    """
    attn_layer = partial(get_attn('se'), rd_ratio=0.25)
    model_args = dict(
        block=Bottleneck, layers=[4, 44, 87, 4], stem_width=32, stem_type='deep', replace_stem_pool=True,
        avg_down=True,  block_args=dict(attn_layer=attn_layer))
    return _create_resnet('resnetrs420', pretrained, **dict(model_args, **kwargs))
