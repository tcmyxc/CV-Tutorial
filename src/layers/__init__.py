from functools import partial

from .activations import *


def get_act_layer(act_layer: str):
    if act_layer == "relu":
        return partial(nn.ReLU, inplace=True)
    elif act_layer == "gelu":
        return nn.GELU
    elif act_layer == "hgelu":
        return HGELU
    elif act_layer == "seqhgelu":
        return SequecialHGELU
    else:
        raise NotImplementedError(act_layer)
