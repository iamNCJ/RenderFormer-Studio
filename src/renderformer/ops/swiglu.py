from typing import Literal

import torch
import torch.nn as nn

from ._triton.swiglu import LigerSiLUMulFunction
from renderformer.models.attention import FeedForwardSwiGLU


class LigerSwiGLUMLP(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
        ):
        super().__init__()
        self.hidden_size = dim
        self.intermediate_size = hidden_dim
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(
            LigerSiLUMulFunction.apply(self.w1(x), self.w3(x))
        )

    @staticmethod
    def can_be_applied(module: FeedForwardSwiGLU):
        return isinstance(module.dropout, nn.Identity) and module.w1.bias is None

    @staticmethod
    def from_torch_module(torch_module: FeedForwardSwiGLU):
        if not __class__.can_be_applied(torch_module):
            print("Warning: LigerSwiGLUMLP does not support dropout or bias.")
            return torch_module
        dim = torch_module.w1.in_features
        hidden_dim = torch_module.w1.out_features
        liger_module = __class__(
            dim=dim,
            hidden_dim=hidden_dim,
        )
        liger_module.load_state_dict(torch_module.state_dict())
        return liger_module
