from functools import partial

from .activations import *


def get_act_layer(act_layer: str):
    if act_layer == "relu":
        return partial(nn.ReLU, inplace=True)
    elif act_layer == "gelu":
        return GELU
    elif act_layer == "hgelu" or act_layer == "gclu":
        return HGELU
    elif act_layer == "quick_gclu":
        return QuickGCLU
    elif act_layer == "gclu_tanh":
        return GCLUTanh
    elif act_layer == "seqhgelu":
        return SequecialHGELU
    else:
        raise NotImplementedError(act_layer)
