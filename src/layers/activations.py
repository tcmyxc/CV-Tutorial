"""
Activations
"""

import math
import torch
import torch.nn as nn

from torch.nn import functional as F


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
    串行实现版本，包含 ReLU层
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


class SequecialHGELUV2(nn.Module):
    """
    串行实现版本，不包含 ReLU层
    """
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        p_out = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # 概率
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x * 2


class SequecialHGELUV3(nn.Module):
    """
    串行实现版本，不包含 ReLU层
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(1, num_features))
        self.log_var = nn.Parameter(torch.zeros(1, num_features))
        self.eps = eps

    def forward(self, x):
        # 计算标准差
        std = torch.exp(0.5 * self.log_var)
        # 归一化
        x_dim = x.ndim
        if x_dim == 2:
            norm_out = (x - self.mu) / (std + self.eps)
        elif x_dim == 4:
            norm_out = (x - self.mu.reshape(1, -1, 1, 1)) / (std.reshape(1, -1, 1, 1) + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x * 2



class SequecialHGELUV4(nn.Module):
    """
    串行实现版本，不包含 ReLU层
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            reduction: int = 1,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(num_features, num_features // reduction)
        self.fc21 = nn.Linear(num_features // reduction, num_features)
        self.fc22 = nn.Linear(num_features // reduction, num_features)
        self.eps = eps

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # h1 = F.sigmoid(self.fc1(x))
        # h1 = self.fc1(x)
        h1 = x
        return self.fc21(h1), self.fc22(h1)

    def forward(self, x):
        mu, log_var = self.encode(torch.flatten(self.avg_pool(x), 1))
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 归一化
        x_dim = x.ndim
        if x_dim == 2:
            norm_out = (x - mu) / (std + self.eps)
        elif x_dim == 4:
            b, c, _, _ = x.size()
            norm_out = (x - mu.reshape(b, c, 1, 1)) / (std.reshape(b, c, 1, 1) + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x
