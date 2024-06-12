import sys
from pathlib import Path

import torch
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models._api import get_model, list_models
# `pytorch-cifar100` models
from models.cifar100 import (
    vgg,
    resnet,
    resnext,
    senet,
    squeezenet,
    wideresidual,
    shufflenetv2,
    densenet,
    googlenet,
    inceptionv3,
    xception,
    preactresnet,
    alexnet,
    sehgelu_resnet,
)
# code from cutout
from models import (
    wide_resnet,
)
# from qt
from models import (
    vision_transformer_timm,
    swin_transformer,
    cait,
    deit,
    tnt,
)
# custom
from models import(
    resnet_center_loss,
    resnet_bneck,
    resnet_cifar,
    resnet_e1,
    resnet_e1d,
    resnet_e2,
    resnet_e3,
    resnet_e4,
    resnet_e5,
    resnet_e6,
    resnet_e6_v2,
    resnet_e7,
    resnet_e8,
    resnet_e5_convnext,
    resnet_e5_trans,
    resnet_torch,
    pyramidnet,
    convnext,
    resnet_timm,
    coatnet,
    se_resnet,
    sehgelu_resnet,
    sehgelu_resnet_v3,
    sehgelu_resnet_v4,
    resnet_cifar_mos,
    simplenetv1,
    efficientnetv2,
)

# torchvision
from models.torchvision import (
    alexnet_torch,
    vgg_torch,
    shufflenetv2_torch,
    squeezenet_torch,
    googlenet_torch,
    efficientnet_torch,
    resnet_torchvision,
    rcnet,
)

# act layers
from layers import get_act_layer


# 支持自定义激活函数的模型列表
ACT_MODEL_LIST = [
    "simplenetv1", 
    "effnetv2_s", 
    "resnet20_mos", "resnet32_mos", "resnet14_mos", "resnet8_mos", "resnet56_mos",
    'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge',
    'sehgelu_resnet14_v4', 'sehgelu_resnet20_v4', 'sehgelu_resnet32_v4', 'sehgelu_resnet56_v4', 'sehgelu_resnet110_v4',
    'alexnet_torch',
    'efficientnet_v2_s_torch',
    'googlenet_torch',
    'shufflenet_v2_x0_5_torch',
    'squeezenet1_0_torch', 'squeezenet1_1_torch',
    'vgg16_torch', 'vgg16_bn_torch',
]


def load_model(args, num_classes=10, **kwargs):
    print(f'\n[INFO] load model: {args.model}, from lib: {args.model_lib}')
    act_layer = get_act_layer(args.act_layer)
    kwargs["act_layer"] = act_layer

    model = None
    if args.model_lib == "torch":
        if args.model in list_models() and args.model in ACT_MODEL_LIST:
            print('\n[INFO] act_layer:', args.act_layer)
            model = get_model(args.model, num_classes=num_classes, act_layer=act_layer)
        elif args.model in list_models():
            model = get_model(args.model, num_classes=num_classes)
    elif args.model_lib == "timm":
        pass
        # TODO
    elif args.model_lib == "cifar100":
        print('\n[INFO] act_layer:', args.act_layer)
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes, act_layer=act_layer)
    elif args.model_lib == "qt":
        print('\n[INFO] act_layer:', args.act_layer)
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes, act_layer=act_layer)
    elif args.model_lib == "custom":
        if "E5" in args.model or args.model in ACT_MODEL_LIST:
            print("[INFO] act_layer:", args.act_layer)
            model = get_model(args.model, num_classes=num_classes, act_layer=act_layer)
        elif args.model in list_models():
            drop_path = getattr(args, "drop_path", None)
            drop_block = getattr(args, "drop_block", None)
            if drop_path is not None:
                model = get_model(args.model, num_classes=num_classes, drop_path_rate=args.drop_path)
            elif drop_block is not None:
                model = get_model(args.model, num_classes=num_classes, drop_block_rate=args.drop_block)
            else:
                model = get_model(args.model, num_classes=num_classes)
    else:
        raise NotImplementedError(args.model)

    if model is not None:
        return model
    else:
        raise NotImplementedError(f"{args.model_lib} library, {args.model} arch not implemented")
