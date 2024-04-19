"""
Activations
"""

import math
import torch
import torch.nn as nn


def hgelu(x, inplace: bool = False):
    # 保证大于0的部分导数不会为0
    p_out = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # 概率
    # 残差学习
    weight = torch.where(p_out < 0.5, p_out, 2 - p_out)

    return weight * x


class HGELU(nn.Module):
    """
    并行实现版本
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hgelu(x)


class SequecialHGELU(nn.Module):
    """
    串行实现版本
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        p_out = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # 概率
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        out = x + weight * x
        return self.relu(out)
