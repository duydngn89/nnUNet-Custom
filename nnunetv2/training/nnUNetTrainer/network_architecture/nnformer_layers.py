"""Small timm-free layer subset required by the official nnFormer network."""

import math

import torch
from torch import nn


def to_3tuple(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(f"expected three spatial values, got {value}")
        return tuple(value)
    return (value, value, value)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """PyTorch equivalent of timm's truncated-normal initializer."""
    if std <= 0:
        raise ValueError("std must be positive")
    with torch.no_grad():
        low = (a - mean) / std
        high = (b - mean) / std
        tensor.uniform_(2 * 0.5 * (1 + math.erf(low / math.sqrt(2))) - 1,
                        2 * 0.5 * (1 + math.erf(high / math.sqrt(2))) - 1)
        tensor.erfinv_().mul_(std * math.sqrt(2)).add_(mean).clamp_(min=a, max=b)
    return tensor
