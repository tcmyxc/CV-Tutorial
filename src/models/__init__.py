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
)
# custom
from models import(
    resnet_center_loss,
    resnet_bneck,
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
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes, **kwargs)
    elif args.model_lib == "qt":
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes)
    elif args.model_lib == "custom":
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes)
    else:
        raise NotImplementedError(args.model)

    if model is not None:
        return model
    else:
        raise NotImplementedError(f"{args.model_lib} library, {args.model} arch not implemented")
