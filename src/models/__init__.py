from pathlib import Path
import sys

import torch.nn as nn
from functools import partial

import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models._api import get_model, list_models
from models.cifar100 import (
    resnet,
)


def load_model(args, num_classes=10, **kwargs):
    print(f'\n[INFO] load model: {args.model}, from lib: {args.model_lib}')
    act_layer = kwargs.pop("act_layer", "relu")
    print('\n[INFO] act_layer:', act_layer)
    act_layer = get_act_layer(act_layer)
    kwargs["act_layer"] = act_layer

    model = None
    if args.model_lib == "torch":
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    elif args.model_lib == "timm":
        pass
        # TODO
    elif args.model_lib == "cifar100":
        if args.model in list_models():
            model = get_model(args.model, num_classes=num_classes)
    else:
        raise NotImplementedError(args.model)

    return model


def get_act_layer(act_layer):
    if act_layer == "relu":
        return partial(nn.ReLU, inplace=True)
    elif act_layer == "gelu":
        return nn.GELU
    else:
        raise NotImplementedError(act_layer)


if __name__ == "__main__":
    print(list_models())

