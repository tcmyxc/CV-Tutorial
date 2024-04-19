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
    shufflenet,
    shufflenetv2,
    mobilenetv2,
    densenet,
    googlenet,
    inceptionv3,
    inceptionv4,
    xception,
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
    resnet_e5_convnext,
    resnet_e5_trans,
    resnet_torch,
    pyramidnet,
    convnext,
    resnet_timm,
    coatnet,
)
# act layers
from layers import get_act_layer


def load_model(args, num_classes=10, **kwargs):
    print(f'\n[INFO] load model: {args.model}, from lib: {args.model_lib}')
    act_layer = get_act_layer(args.act_layer)
    kwargs["act_layer"] = act_layer

    model = None
    if args.model_lib == "torch":
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    elif args.model_lib == "timm":
        pass
        # TODO
    elif args.model_lib == "cifar100":
        print('\n[INFO] act_layer:', args.act_layer)
        # TODO: `act_layer` param current only for resnet
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes, **kwargs)
    elif args.model_lib == "qt":
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes)
    elif args.model_lib == "custom":
        if "E5" in args.model:
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
