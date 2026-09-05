# Modified from https://github.com/lucidrains/rotary-embedding-torch/blob/e2224b5102a045998b4131bcac52c9ebde779b5d/rotary_embedding_torch/rotary_embedding_torch.py

# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from math import pi, log

import torch
from torch.nn import Module
from torch.amp import autocast
from torch import nn, einsum, broadcast_tensors, Tensor

from einops import rearrange, repeat

from typing import Literal


# helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcat, as tortoise-tts was using it
def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


# rotary embedding helper functions
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rotate_half_hf(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (
        rotate_half(t_middle) * freqs.sin() * scale
    )

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


def freqs_to_cos_sin(freqs, scale=1.0, start_index=0, head_dim=None):
    """
    Convert frequencies to cos and sin for rotary embeddings.

    Args:
        freqs (torch.Tensor): The frequency tensor of shape (..., n_freqs).
        scale (float): The scaling factor for the frequencies.
        start_index (int): The starting index of the frequencies.
        head_dim (int): The dimension of the head.
    """
    if head_dim is not None:
        # pad the freqs to match the head_dim
        freqs = freqs[..., : freqs.shape[-1] // 2]
        left_pad = start_index
        right_pad = head_dim // 2 - (left_pad + freqs.shape[-1])
        freqs = torch.cat(
            (torch.zeros((*freqs.shape[:-1], left_pad), device=freqs.device),
             freqs,
             torch.zeros((*freqs.shape[:-1], right_pad), device=freqs.device)),
            dim=-1,
        )
        freqs = torch.cat([freqs, freqs], dim=-1)

    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    return cos, sin


@autocast("cuda", enabled=False)
def apply_rotary_emb_cossin(q, k, cos, sin):
    """
    q size: (bsz, n_q_head, seq_len, head_dim)
    k size: (bsz, n_kv_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == q.shape[-1]
    ), f"feature dimension {q.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    q = (q * cos) + (
        rotate_half_hf(q) * sin
    )
    k = (k * cos) + (
        rotate_half_hf(k) * sin
    )

    return q.type(dtype), k.type(dtype)


@autocast("cuda", enabled=False)
def apply_rotary_emb_one_cossin(one_tensor, cos, sin):
    """
    one_tensor size: (bsz, n_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = one_tensor.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == one_tensor.shape[-1]
    ), f"feature dimension {one_tensor.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    one_tensor = (one_tensor * cos) + (
        rotate_half_hf(one_tensor) * sin
    )

    return one_tensor.type(dtype)


# learned rotation helpers
def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes
class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        custom_freqs: Tensor | None = None,
        freqs_for: Literal[
            "lang",
            "pixel",
            "constant",
            "triangle",
            "triangle_learned",
            "triangle_mixed",
            "triangle_center"
        ] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        num_heads=1,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
        cache_max_seq_len=8192,
        hf_format=True,
        double_max_freq=False,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for
        self.hf_format = hf_format

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for in ["triangle", "triangle_learned", "triangle_center"]:
            print(
                f"Using {freqs_for} frequencies, ignoring max_freq and using `dim // 2` instead"
            )
            # log spaced frequencies
            max_freq = log(dim // 2 - 1, 2) if not double_max_freq else log(dim - 1, 2)
            freqs = 2 ** torch.linspace(0, max_freq, dim // 2)
            print("Using freqs: ", freqs)
            if freqs_for == "triangle_learned":
                # repeat for each head
                freqs = repeat(freqs, "f -> h f", h=num_heads).clone()  # to force allocate new memory
                learned_freq = True  # force learned freqs
        elif freqs_for == "triangle_mixed":
            print(
                f"Using {freqs_for} frequencies, ignoring max_freq and using `dim // 2` instead"
            )
            # log spaced frequencies
            init_dim = dim // 2 // 9
            print("init per triangle dim: ", init_dim)
            max_freq = log(init_dim - 1, 2)
            init_freqs = 2 ** torch.linspace(0, max_freq, init_dim)

            # init into triangle frequencies
            freqs = torch.zeros(9, dim // 2)
            for i in range(9):
                freqs[i, init_dim * i:init_dim * (i + 1)] = init_freqs
            # repeat each frequency for each dimension and each head
            freqs = repeat(freqs, "d f -> h d f", h=num_heads).clone()  # to force allocate new memory
            learned_freq = True  # force learned freqs
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer(
            "cached_freqs", torch.zeros(cache_max_seq_len, dim), persistent=False
        )
        self.register_buffer("cached_freqs_seq_len", torch.tensor(0), persistent=False)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)
        self.learned_freq = learned_freq
        print(f"Using learned freqs: {learned_freq}")

        # dummy for device
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer(
            "cached_scales", torch.zeros(cache_max_seq_len, dim), persistent=False
        )
        self.register_buffer("cached_scales_seq_len", torch.tensor(0), persistent=False)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(
            scale
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        dtype, device, seq_dim = (
            q.dtype,
            q.device,
            default(seq_dim, self.default_seq_dim),
        )

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.0

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(
            q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset
        )
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale**-1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(self, t: Tensor, seq_len: int | None = None, offset=0):
        assert self.use_xpos

        should_cache = (
            self.cache_if_possible
            and exists(seq_len)
            and (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache
            and exists(self.cached_scales)
            and (seq_len + offset) <= self.cached_scales_seq_len.item()
        ):
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = repeat(scale, "n d -> n (d r)", r=2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len.copy_(seq_len)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    def get_triangle_freqs(self, pos: Tensor):
        if self.freqs_for == "triangle":
            # generate all frequencies for all triangles
            freqs = self.forward(pos)
            freqs = rearrange(
                freqs, "batch n_tris n_verts d -> batch 1 n_tris (n_verts d)"
            )  # 1 for head dim
        elif self.freqs_for == "triangle_center":
            freqs = self.forward(pos) # [batch, n_tris, n_verts, d]
            freqs = rearrange(freqs, "batch n_tris n_dim d -> batch 1 n_tris (n_dim d)") # [batch, 1, n_tris, (n_dim d)]
        elif self.freqs_for == "triangle_mixed":
            freqs = self.forward_mixed_freqs(pos)  # [batch, n_tris, n_heads, n_freqs]
            freqs = rearrange(freqs, "batch n_tris n_heads d -> batch n_heads n_tris d")
        elif self.freqs_for == "triangle_learned":
            freqs = self.forward_learned_freqs(
                pos
            )  # [batch, n_tris, n_heads, n_verts, n_freqs]
            freqs = rearrange(
                freqs,
                "batch n_tris n_heads n_verts d -> batch n_heads n_tris (n_verts d)",
            )
        else:
            raise NotImplementedError

        if self.hf_format:
            freqs = torch.cat([freqs, freqs], dim=-1)
        else:
            freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        return freqs

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible
            and not self.learned_freq
            and exists(seq_len)
            and (self.freqs_for != "pixel" and self.freqs_for != "triangle")
            and (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache
            and exists(self.cached_freqs)
            and (offset + seq_len) <= self.cached_freqs_seq_len.item()
        ):
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len.copy_(seq_len)

        return freqs

    @autocast("cuda", enabled=False)
    def forward_learned_freqs(self, t: Tensor, seq_len=None, offset=0):
        assert self.freqs_for == "triangle_learned"

        freqs = self.freqs  # [n_heads, n_freqs]
        freqs = einsum("... t, h f -> ... h t f", t, freqs)

        return freqs

    @autocast("cuda", enabled=False)
    def forward_mixed_freqs(self, t: Tensor, seq_len=None, offset=0):
        """
        This is the forward pass for the mixed triangle frequencies.
        Args:
            t: Tensor, shape [batch_size, seq_len, 9]
        """
        assert self.freqs_for == "triangle_mixed"

        freqs = self.freqs  # [n_heads, 9, n_freqs]
        freqs = einsum("... t, h t f -> ... h f", t, freqs)

        return freqs
