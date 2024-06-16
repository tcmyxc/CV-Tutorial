"""
Activations
"""

import math
import torch
import torch.nn as nn

from torch.nn import functional as F


def gelu(x, inplace: bool = False):
    p_out = 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # 概率
    return p_out * x


class GELU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return gelu(x)


def gclu_tanh(x, inplace: bool = False):
    p_out = 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))  # 概率
    weight = torch.where(p_out < 0.5, p_out, 2 - p_out)
    return weight * x


class GCLUTanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return gclu_tanh(x)


def quick_gclu(x, inplace: bool = False):
    # 使用cdf的近似形式计算概率
    p_out = torch.sigmoid(1.702 * x)  # 概率
    weight = torch.where(p_out < 0.5, p_out, 2 - p_out)
    return weight * x


class QuickGCLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return quick_gclu(x)


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
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc21 = nn.Linear(num_features, num_features)
        self.fc22 = nn.Linear(num_features, num_features)
        self.eps = eps

    def encode(self, x):
        return self.fc21(x), self.fc22(x)

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


class SequecialHGELUV4B(nn.Module):
    """
    串行实现版本，不包含 ReLU层
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            r: int = 16,
            dropout_p: float = 0,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_features, num_features//r)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0. else nn.Identity()
        self.fc21 = nn.Linear(num_features//r, num_features)
        self.fc22 = nn.Linear(num_features//r, num_features)
        self.eps = eps

    def encode(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc21(x), self.fc22(x)

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


class SequecialHGELUV4C(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            r: int = 16,
            dropout_p: float = 0,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_features, num_features//r, 1)
        self.fc21 = nn.Conv2d(num_features//r, num_features, 1)
        self.fc22 = nn.Conv2d(num_features//r, num_features, 1)
        self.eps = eps

    def encode(self, x):
        x = self.fc1(x)
        return self.fc21(x), self.fc22(x)

    def forward(self, x):
        mu, log_var = self.encode(self.avg_pool(x))
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 归一化
        norm_out = (x - mu) / (std + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x


class SequecialHGELUV5(nn.Module):
    """
    串行实现版本，分组卷积实现
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, groups=num_features)
        self.fc2 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, groups=num_features)
        self.eps = eps

    def encode(self, x):
        return self.fc1(x), self.fc2(x)

    def forward(self, x):
        mu, log_var = self.encode(self.avg_pool(x))
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 归一化
        norm_out = (x - mu) / (std + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x


class SequecialHGELUV6(nn.Module):
    """
    串行实现版本，分组卷积实现
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, groups=num_features)
        self.fc2 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=1, padding=0, groups=num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eps = eps

    def encode(self, x):
        return self.avg_pool(self.fc1(x)), self.avg_pool(self.fc2(x))

    def forward(self, x):
        mu, log_var = self.encode(x)
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 归一化
        norm_out = (x - mu) / (std + self.eps)
        # 计算概率
        p_out = 0.5 * (1 + torch.erf(norm_out / math.sqrt(2)))
        # 残差学习
        weight = torch.where(p_out < 0.5, p_out, 1 - p_out)

        return weight * x
