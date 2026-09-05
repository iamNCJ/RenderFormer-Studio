from typing import Literal

import torch
import torch.nn as nn

from ._triton.rms_norm import LigerRMSNormFunction


class LigerRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        casting_mode: Literal["llama", "gemma", "none"] = "gemma",
        init_fn: Literal["ones", "zeros"] = "ones",
        in_place=True,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon, self.offset, self.casting_mode, self.in_place = (
            eps,
            offset,
            casting_mode,
            in_place,
        )

    @staticmethod
    def from_torch_module(torch_module: nn.RMSNorm):
        eps = torch_module.eps
        if eps is None:
            # eps = torch.finfo(torch.float16).eps
            eps = 1e-6
        liger_module = __class__(
            hidden_size=torch_module.normalized_shape,
            eps=eps,
            init_fn="ones" if torch_module.weight[0] == 1.0 else "zeros",
        )
        liger_module.load_state_dict(torch_module.state_dict())
        return liger_module

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
            self.in_place,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}, in_place={self.in_place}"
