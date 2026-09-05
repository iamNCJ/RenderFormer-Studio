import os
from typing import Dict, List, Literal, Optional
import random
import math

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.attention.flex_attention import and_masks, or_masks, create_block_mask, flex_attention

from .ordering import encode_mask_batch
from .rope import (
    RotaryEmbedding,
    freqs_to_cos_sin,
    apply_rotary_emb_cossin,
    apply_rotary_emb_one_cossin
)


EPS = 1e-6
ATTN = os.environ.get("ATTN_IMPL", "flash_attn")
if os.environ.get("USE_SDPA", "0") == "1":
    ATTN = "sdpa"
if ATTN not in {"flash_attn", "sdpa"}:
    raise ValueError("ATTN_IMPL must be either 'flash_attn' or 'sdpa'")

if ATTN == "flash_attn":
    try:
        from flash_attn import (
            flash_attn_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_kvpacked_func,
        )
        from flash_attn.bert_padding import pad_input, unpad_input
    except ImportError:
        print(
            "flash-attn is unavailable; falling back to PyTorch SDPA. "
            "Install flash-attn for accelerated training and inference.",
            flush=True,
        )
        ATTN = "sdpa"
# print('Using attention type: ', ATTN)

if os.environ.get('DISABLE_FLEX_COMPILE', '0') == '1':
    print('Disabling flex-attention compile, will be slower')
else:
    flex_attention = torch.compile(flex_attention)
    create_block_mask = torch.compile(create_block_mask)


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with SwiGLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x))))


class FeedForwardGeLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with GeLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.gelu(self.w1(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        num_heads,
        num_kv_heads=None,
        kv_dim=None,
        bias=True,
        qk_norm=False,
        norm_type='layer_norm',
        use_local_attention=False,
        local_attention_window_size_half=-1,
        use_sdpa=False,
        use_flex_attention=False,
    ):
        super().__init__()
        self.apply_rope_cossin = apply_rotary_emb_cossin

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.is_self_attn = kv_dim is None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.is_gqa = self.num_kv_heads != self.num_heads
        if self.is_gqa:
            assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
            self.num_head_per_group = self.num_heads // self.num_kv_heads
        kv_dim = query_dim if kv_dim is None else kv_dim

        if self.is_self_attn and not self.is_gqa:
            self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(query_dim, self.num_heads * self.head_dim, bias=bias)
            self.k_proj = nn.Linear(kv_dim, self.num_kv_heads * self.head_dim, bias=bias)
            self.v_proj = nn.Linear(kv_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        if qk_norm:
            if norm_type == 'layer_norm':
                norm_module = nn.LayerNorm
            elif norm_type == 'rms_norm':
                norm_module = nn.RMSNorm
            else:
                raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")
            self.q_norm = norm_module(self.num_heads * self.head_dim, eps=EPS)
            self.k_norm = norm_module(self.num_kv_heads * self.head_dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.use_local_attention = use_local_attention
        self.local_attention_window_size_half = local_attention_window_size_half
        if not self.use_local_attention:
            self.local_attention_window_size_half = -1
            # default -1 means full attention

        self.use_flex_attention = use_flex_attention

        self._dump_attn_weight = False
        self._attn_weight_path = None

    @property
    def dump_attn_weight(self):
        return self._dump_attn_weight

    @dump_attn_weight.setter
    def dump_attn_weight(self, value):
        self._dump_attn_weight = value

    @property
    def attn_weight_path(self):
        return self._attn_weight_path

    @attn_weight_path.setter
    def attn_weight_path(self, value):
        self._attn_weight_path = value

    def forward(self, q, k, v, src_key_padding_mask=None, rope_cos=None, rope_sin=None, rope_ctx_cos=None, rope_ctx_sin=None, force_sdpa=False, block_mask=None):
        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        bs, src_len = q.shape[0], q.shape[1]
        ctx_len = k.shape[1]

        if self.is_self_attn and not self.is_gqa:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
        else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        # qk normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(bs, src_len, self.num_heads, -1).transpose(1, 2)  # (bs, num_heads, src_len, head_dim)
        k = k.view(bs, ctx_len, self.num_kv_heads, -1).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_kv_heads, -1).transpose(1, 2)

        # apply rope
        if rope_cos is not None:
            if rope_ctx_cos is None:
                q, k = self.apply_rope_cossin(q, k, rope_cos, rope_sin)
            else:
                q = apply_rotary_emb_one_cossin(q, rope_cos, rope_sin)
                k = apply_rotary_emb_one_cossin(k, rope_ctx_cos, rope_ctx_sin)

        if self.use_flex_attention:
            assert block_mask is not None, "block_mask must be provided if use_flex_attention is True"
            # Use flex attention with block mask
            attn_output = flex_attention(q.type(v.dtype), k.type(v.dtype), v, block_mask=block_mask).transpose(1, 2).contiguous().view(bs, src_len, -1)
        elif ATTN == 'sdpa' or force_sdpa or self.dump_attn_weight:
            # create attention mask
            if src_key_padding_mask is not None:
                assert src_key_padding_mask.shape == (bs, ctx_len), \
                    f"expecting key_padding_mask shape of {(bs, ctx_len)}, but got {src_key_padding_mask.shape}"
                attn_mask = (
                    src_key_padding_mask.view(bs, 1, 1, ctx_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bs, self.num_heads, 1, ctx_len)
                )
            else:
                attn_mask = None

            if self.dump_attn_weight:
                assert self.attn_weight_path is not None, "attn_weight_path must be provided if dump_attn_weight is True"
                scale_factor = 1 / math.sqrt(q.size(-1))
                _attn_mask = 0.
                if attn_mask is not None:
                    _attn_mask = torch.zeros_like(attn_mask, dtype=torch.float32)
                    _attn_mask = _attn_mask.masked_fill(attn_mask.logical_not(), float("-inf"))
                attn_weight = q @ k.transpose(-2, -1) * scale_factor
                attn_weight += _attn_mask
                attn_weight = torch.softmax(attn_weight, dim=-1)
                torch.save(attn_weight, self.attn_weight_path)

            attn_output = F.scaled_dot_product_attention(
                query=q.type(v.dtype),
                key=k.type(v.dtype),
                value=v,
                attn_mask=attn_mask,
                enable_gqa=self.is_gqa,
            ).transpose(1, 2).contiguous().view(bs, src_len, -1)
            # TODO: local attention for SDPA
        elif ATTN == 'flash_attn':
            # self-attn
            if self.is_self_attn:
                if src_key_padding_mask is not None:
                    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q.transpose(1, 2), src_key_padding_mask)
                    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k.transpose(1, 2), src_key_padding_mask)
                    v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v.transpose(1, 2), src_key_padding_mask)
                    if not self.is_gqa:
                        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
                        out_unpad = flash_attn_varlen_qkvpacked_func(
                            qkv_unpad, cu_seqlens_q, max_seqlen_q,
                            window_size=(self.local_attention_window_size_half, self.local_attention_window_size_half)
                        )
                    else:  # GQA/MQA
                        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
                        out_unpad = flash_attn_varlen_kvpacked_func(
                            q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_q, max_seqlen_q, max_seqlen_q,  # in self-attn, cu_seqlens_k == cu_seqlens_q
                            window_size=(self.local_attention_window_size_half, self.local_attention_window_size_half)
                        )
                    attn_output = pad_input(out_unpad, indices_q, bs, src_len).contiguous().view(bs, src_len, -1)
                else:
                    qkv = torch.stack([
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                    ], dim=2)
                    attn_output = flash_attn_qkvpacked_func(
                        qkv,
                        window_size=(self.local_attention_window_size_half, self.local_attention_window_size_half)
                    ).contiguous().view(bs, src_len, -1)
            # cross-attn
            else:
                q_unpad = rearrange(q, "b h s d -> (b s) h d")
                cu_seqlens_q = torch.arange(
                    0, (bs + 1) * src_len, step=src_len, dtype=torch.int32, device=q_unpad.device
                )
                max_seqlen_q = src_len

                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k.transpose(1, 2), src_key_padding_mask)
                v_unpad, _, _, _, _ = unpad_input(v.transpose(1, 2), src_key_padding_mask)

                kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
                out_unpad = flash_attn_varlen_kvpacked_func(
                    q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                )
                attn_output = rearrange(
                    out_unpad, "(b s) h d -> b s h d", b=bs
                ).contiguous().view(bs, src_len, -1)
        else:
            raise ValueError("Unsupported attention type. Choose from 'flash_attn' and 'sdpa'.")

        return self.out_proj(attn_output)

    def forward_varlen(
        self,
        x: torch.Tensor,           # [total_tokens, D]
        cu_seqlens: torch.Tensor,  # [B+1]
        max_seqlen: int,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """Self attention with flash_attn_varlen for packed sequences."""
        total_tokens, D = x.shape

        # Project to Q, K, V
        if self.is_self_attn and not self.is_gqa:
            qkv = self.in_proj(x)  # [total_tokens, 3 * D]
            qkv = qkv.reshape(total_tokens, 3, self.num_heads, self.head_dim)
            q = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
            k = qkv[:, 1]
            v = qkv[:, 2]
        else:
            q = self.q_proj(x)  # [total_tokens, num_heads * head_dim]
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        # qk normalization
        q = self.q_norm(q.view(total_tokens, -1)).view(total_tokens, self.num_heads, self.head_dim)
        k = self.k_norm(k.view(total_tokens, -1)).view(total_tokens, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            # Expand rope_cos/sin to match num_heads if needed
            if rope_cos.ndim == 2:  # [total_tokens, head_dim]
                rope_cos = rope_cos.unsqueeze(1).expand(-1, self.num_heads, -1)
                rope_sin = rope_sin.unsqueeze(1).expand(-1, self.num_heads, -1)
            q = apply_rotary_emb_one_cossin(q, rope_cos, rope_sin)
            # For k, use same rope if self-attn, or expand for kv_heads
            if self.is_self_attn:
                k = apply_rotary_emb_one_cossin(k, rope_cos[:, :self.num_kv_heads, :], rope_sin[:, :self.num_kv_heads, :])
            else:
                # For cross-attn, k might have different rope, but for now use same
                k_rope_cos = rope_cos[:, :self.num_kv_heads, :] if rope_cos.shape[1] >= self.num_kv_heads else rope_cos
                k_rope_sin = rope_sin[:, :self.num_kv_heads, :] if rope_sin.shape[1] >= self.num_kv_heads else rope_sin
                k = apply_rotary_emb_one_cossin(k, k_rope_cos, k_rope_sin)

        # Group k, v for GQA if needed
        if self.is_gqa:
            k = k.repeat_interleave(self.num_head_per_group, dim=1)
            v = v.repeat_interleave(self.num_head_per_group, dim=1)

        # Use flash attention varlen
        if ATTN == 'flash_attn':
            if self.is_self_attn and not self.is_gqa:
                qkv = torch.stack([q, k, v], dim=1)  # [total_tokens, 3, num_heads, head_dim]
                if self.local_attention_window_size_half > 0:
                    out = flash_attn_varlen_qkvpacked_func(
                        qkv, cu_seqlens, max_seqlen,
                        window_size=(self.local_attention_window_size_half, self.local_attention_window_size_half),
                    )
                else:
                    out = flash_attn_varlen_qkvpacked_func(
                        qkv, cu_seqlens, max_seqlen,
                    )
            else:
                kv = torch.stack([k, v], dim=1)  # [total_tokens, 2, num_heads, head_dim]
                if self.local_attention_window_size_half > 0:
                    out = flash_attn_varlen_kvpacked_func(
                        q, kv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                        window_size=(self.local_attention_window_size_half, self.local_attention_window_size_half),
                    )
                else:
                    out = flash_attn_varlen_kvpacked_func(
                        q, kv, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                    )
        else:
            raise NotImplementedError("varlen attention only supports flash_attn for now")

        # Project output
        out = out.contiguous().view(total_tokens, -1)
        return self.out_proj(out)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


SWIN_ATTN_MASK_CACHE = {}
def get_swin_attn_mask(H, W, window_size, shift_size, device):
    """
    Get the attention mask for Swin Transformer. (Original implementation)
    Args:
        H (int): height of image
        W (int): width of image
        window_size (int): window size
        shift_size (int): shift size
        device (torch.device): device to store the attention mask
    Returns:
        attn_mask: (num_windows, num_windows)
    """
    if (H, W, window_size, shift_size) in SWIN_ATTN_MASK_CACHE:
        return SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)]
    else:
        print(f"Generating Swin attention mask for {H}x{W} with window size {window_size} and shift size {shift_size}")
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (attn_mask == 0).to(torch.bool)  # nW, window_size * window_size, window_size * window_size
        SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)] = attn_mask
        return attn_mask


class SwinSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size: int = 0,
        bias=True,
        qk_norm=False,
        norm_type='layer_norm'
    ):
        """
        Args:
            dim (int): input dimension
            num_heads (int): number of attention heads
            window_size (int): window size
            shift_size (int): shift size, if None, no shift
            bias (bool): whether to use bias, default True
            qk_norm (bool): whether to normalize query and key, default False
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.in_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if qk_norm:
            if norm_type == 'layer_norm':
                norm_module = nn.LayerNorm
            elif norm_type == 'rms_norm':
                norm_module = nn.RMSNorm
            else:
                raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")
            self.q_norm = norm_module(dim, eps=EPS)
            self.k_norm = norm_module(dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self._dump_attn_weight = False
        self._attn_weight_path = None

    @property
    def dump_attn_weight(self):
        return self._dump_attn_weight

    @dump_attn_weight.setter
    def dump_attn_weight(self, value):
        self._dump_attn_weight = value

    @property
    def attn_weight_path(self):
        return self._attn_weight_path

    @attn_weight_path.setter
    def attn_weight_path(self, value):
        self._attn_weight_path = value

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        nW = H * W // self.window_size // self.window_size

        # swin related operations
        if self.shift_size > 0:
            attn_mask = get_swin_attn_mask(H, W, self.window_size, self.shift_size, x.device)  # nW, window_size * window_size, window_size * window_size
            attn_mask = attn_mask.repeat(B, 1, 1)[:, None]  # B * nW, 1, window_size * window_size, window_size * window_size
        else:
            attn_mask = None

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # B * nW, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # B * nW, window_size * window_size, C

        q, k, v = self.in_proj(x_windows).chunk(3, dim=-1)

        # qk normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B * nW, num_heads, window_size * window_size, head_dim)
        q = q.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)
        k = k.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)
        v = v.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)

        if self.dump_attn_weight:
            assert self.attn_weight_path is not None, "attn_weight_path must be provided if dump_attn_weight is True"
            scale_factor = 1 / math.sqrt(q.size(-1))
            attn_bias = torch.zeros(B * nW, 1, self.window_size * self.window_size, self.window_size * self.window_size, dtype=q.dtype, device=q.device)
            attn_weight = q @ k.transpose(-2, -1) * scale_factor
            if attn_mask is not None:
                attn_weight.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_weight = torch.softmax(attn_weight, dim=-1)  # [B * nW, num_heads, window_size * window_size, window_size * window_size]
            attn_weight = attn_weight.view(B, nW, self.num_heads, self.window_size * self.window_size, self.window_size * self.window_size)
            # print(attn_weight.shape)
            full_attn_map = torch.zeros(B, self.num_heads, nW, self.window_size * self.window_size, nW, self.window_size * self.window_size)
            for i in range(nW):
                full_attn_map[:, :, i, :, i, :] = attn_weight[:, i, :, :]
            full_attn_map = rearrange(full_attn_map, 'B heads (hNW1 wNW1) (hWS1 wWS1) (hNW2 wNW2) (hWS2 wWS2) -> B heads (hNW1 hWS1) (wNW1 wWS1) (hNW2 hWS2) (wNW2 wWS2)', hNW1=H // self.window_size, wNW1=W // self.window_size, hWS1=self.window_size, wWS1=self.window_size, hNW2=H // self.window_size, wNW2=W // self.window_size, hWS2=self.window_size, wWS2=self.window_size)
            # full_attn_map.shape  # [B * nW, num_heads, H, window_size, window_size]
            if self.shift_size > 0:
                full_attn_map = torch.roll(full_attn_map, shifts=(self.shift_size, self.shift_size, self.shift_size, self.shift_size), dims=(2, 3, 4, 5))
            torch.save(full_attn_map, self.attn_weight_path)

        # apply attention
        attn_output = F.scaled_dot_product_attention(
            query=q.type(v.dtype),
            key=k.type(v.dtype),
            value=v,
            attn_mask=attn_mask,
        ).transpose(1, 2).contiguous().view(B * nW, self.window_size * self.window_size, -1)

        attn_windows = self.out_proj(attn_output)  # B * nW, window_size * window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H, W, C)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        num_kv_heads: Optional[int] = None,
        kv_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        bias_kv: bool = False,
        activation: str = 'swiglu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        disable_q_norm: bool = False,
        disable_kv_norm: bool = False,
        qk_norm: bool = False,
        add_self_attn: bool = False,
        use_swin_attn: bool = False,
        window_size: int = 8,
        shift_size: int = 0,
        use_local_attention: bool = False,
        local_attention_window_size_half: int = 256,
        use_flex_attention: bool = False,
        self_attn_before_cross: bool = False,
    ):
        """
        Attention layer with feed forward and pre-norm.
        Args:
            query_dim (int): input dimension
            kv_dim (int): key and value dimension, if None, set to query_dim (self-attention)
            num_heads (int): number of attention heads
            hidden_dim (int): feed forward hidden dim
            num_kv_heads (int): number of key and value heads (for GQA/MQA), if None, set to num_heads (MHA)
            dropout (float): dropout, default 0.1
            bias (bool): whether to use bias, default True
            bias_kv (bool): whether to use bias for key and value, default False
            activation (str): activation function, choose from 'gelu' and 'swiglu', default 'swiglu'
            norm_type (str): normalization type, choose from 'layer_norm' and 'rms_norm', default 'layer_norm'
            disable_q_norm (bool): disable query normalization, default False
            disable_kv_norm (bool): disable key and value normalization, default False
            qk_norm (bool): whether to apply normalization to query and key, default False
            add_self_attn (bool): whether to add self-attention after cross-attention (cross-attn, self-attn, ffn), default False
            self_attn_before_cross (bool): whether to put self-attention before cross-attention, default False (self-attn after cross-attn)
            use_swin_attn (bool): whether to use swin self-attention, default False
            window_size (int): window size for swin self-attention, default 8
            shift_size (int): shift size for swin self-attention, default 0 (no shift)
            use_local_attention (bool): whether to use local attention, default False
            local_attention_window_size_half (int): half of the window size for local attention, default 256
        Returns:
            torch.Tensor: (B, N, query_dim)
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            kv_dim=kv_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type,
            use_local_attention=use_local_attention,
            local_attention_window_size_half=local_attention_window_size_half,
            use_flex_attention=use_flex_attention,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if bias_kv:
            raise NotImplementedError("Bias for key and value is not supported for now")

        if norm_type == 'layer_norm':
            norm_module = nn.LayerNorm
        elif norm_type == 'rms_norm':
            norm_module = nn.RMSNorm
        else:
            raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")

        self.query_norm = norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()
        kv_dim = query_dim if kv_dim is None else kv_dim
        if not self.multihead_attn.is_self_attn:
            self.kv_norm = norm_module(kv_dim, eps=EPS) if not disable_kv_norm else nn.Identity()

        self.add_self_attn = add_self_attn
        self.use_swin_attn = use_swin_attn
        self.self_attn_before_cross = self_attn_before_cross
        if add_self_attn:
            if use_swin_attn:
                self.self_attn = SwinSelfAttention(
                    dim=query_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type
                )
            else:
                self.self_attn = MultiHeadAttention(
                    query_dim=query_dim,
                    num_heads=num_heads,
                    kv_dim=None,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type,
                )
            self.self_attn_norm = norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()

        if activation == 'swiglu':
            self.ffn = FeedForwardSwiGLU(
                query_dim,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias
            )
        elif activation == 'gelu':
            self.ffn = FeedForwardGeLU(
                query_dim,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias
            )
        else:
            raise ValueError("Unsupported activation function. Choose from 'gelu' and 'swiglu'.")
        
        self.ffn_norm = norm_module(query_dim, eps=EPS)

    def _do_self_attn(self, query, rope_cos=None, rope_sin=None, force_sdpa=False, patch_h=None, patch_w=None):
        """Apply self-attention to query."""
        bs = query.shape[0]
        q = self.self_attn_norm(query)
        if self.use_swin_attn:
            q = q.view(bs, patch_h, patch_w, -1)
            self_attn_output = self.self_attn(q)
            self_attn_output = self_attn_output.view(bs, patch_h * patch_w, -1)
        else:
            self_attn_output = self.self_attn(q, q, q, None, rope_cos, rope_sin, force_sdpa=force_sdpa)
        return query + self.dropout(self_attn_output)

    def forward(self, query, kv=None, src_key_padding_mask=None, rope_cos=None, rope_sin=None, rope_ctx_cos=None, rope_ctx_sin=None, force_sdpa=False, patch_h=None, patch_w=None, block_mask=None):
        """
        Args:
            query (torch.Tensor): (B, N, query_dim)
            kv (torch.Tensor): (B, N, kv_dim), key and value, if None, set to query (self-attention)
            src_key_padding_mask (torch.Tensor): (B, N), key padding mask, things you want to attend to is True
            rope_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            rope_ctx_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_ctx_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            patch_h (int): height of the patch, used for swin self-attention
            patch_w (int): width of the patch, used for swin self-attention
        Returns:
            torch.Tensor: (B, N, query_dim)
        """
        # Self-attention before cross-attention (if configured)
        if self.add_self_attn and self.self_attn_before_cross:
            query = self._do_self_attn(query, rope_cos, rope_sin, force_sdpa, patch_h, patch_w)

        q = self.query_norm(query)
        if self.multihead_attn.is_self_attn:
            kv = q
        else:
            kv = self.kv_norm(kv)

        # multihead attention (cross-attention)
        attn_output = self.dropout(self.multihead_attn(q, kv, kv, src_key_padding_mask, rope_cos, rope_sin, rope_ctx_cos, rope_ctx_sin, force_sdpa=force_sdpa, block_mask=block_mask))
        query = query + attn_output

        # Self-attention after cross-attention (default behavior)
        if self.add_self_attn and not self.self_attn_before_cross:
            query = self._do_self_attn(query, rope_cos, rope_sin, force_sdpa, patch_h, patch_w)

        # feed forward
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query

    def forward_varlen(
        self,
        x: torch.Tensor,           # [total_tokens, D]
        cu_seqlens: torch.Tensor,  # [B+1]
        max_seqlen: int,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward with varlen attention."""
        # Self attention
        residual = x
        x = self.query_norm(x)
        x = self.multihead_attn.forward_varlen(
            x, cu_seqlens, max_seqlen, rope_cos, rope_sin
        )
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x

    def forward_cross_varlen(
        self,
        query: torch.Tensor,           # [B, num_queries, D]
        context_packed: torch.Tensor,  # [total_context_tokens, D]
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        rope_ctx_cos: torch.Tensor = None,
        rope_ctx_sin: torch.Tensor = None,
        patch_h: int = None,
        patch_w: int = None,
        use_reentrant: bool = False,
    ) -> torch.Tensor:
        """
        Cross attention with varlen packed context.

        Args:
            query: Query tokens [B, num_queries, D] - typically ray tokens (equal length per sample)
            context_packed: Packed context tokens [total_context_tokens, D] - variable length per sample
            cu_seqlens_q: Cumulative sequence lengths for queries [B+1]
            cu_seqlens_k: Cumulative sequence lengths for keys/values [B+1]
            max_seqlen_q: Maximum query sequence length
            max_seqlen_k: Maximum key/value sequence length
            rope_cos: RoPE cosine for queries. Supported shapes:
                - [B*num_queries, head_dim]: Pre-flattened, shared across heads
                - [B, num_queries, head_dim]: Per-sample, shared across heads (triangle_center mode)
                - [B, num_queries, n_heads, head_dim]: Per-sample, per-head (triangle_mixed mode)
            rope_sin: RoPE sine for queries (same shape options as rope_cos)
            rope_ctx_cos: RoPE cosine for context keys. Supported shapes:
                - [total_context_tokens, head_dim]: Pre-flattened packed format, shared across heads
                - [B, seq_len, head_dim]: Per-sample padded format, shared across heads
                - [B, seq_len, n_heads, head_dim]: Per-sample padded format, per-head
            rope_ctx_sin: RoPE sine for context keys (same shape options as rope_ctx_cos)
            patch_h: Height of patches for SwinSelfAttention (optional)
            patch_w: Width of patches for SwinSelfAttention (optional)
            use_reentrant: Whether to use reentrant checkpointing (unused)

        Returns:
            Output tensor [B, num_queries, D]
        """
        B, num_queries, D = query.shape

        # Cross attention
        residual = query
        query = self.query_norm(query)
        context = self.kv_norm(context_packed)

        # Project query and context
        q = self.multihead_attn.q_proj(query)  # [B, num_queries, num_heads * head_dim]
        k = self.multihead_attn.k_proj(context)  # [total_context, num_kv_heads * head_dim]
        v = self.multihead_attn.v_proj(context)  # [total_context, num_kv_heads * head_dim]

        # Apply qk normalization
        q = self.multihead_attn.q_norm(q)
        k = self.multihead_attn.k_norm(k)

        # Reshape
        q = q.view(B * num_queries, self.multihead_attn.num_heads, self.multihead_attn.head_dim)
        k = k.view(-1, self.multihead_attn.num_kv_heads, self.multihead_attn.head_dim)
        v = v.view(-1, self.multihead_attn.num_kv_heads, self.multihead_attn.head_dim)

        # ========== Apply RoPE to queries ==========
        # Target shape for q after RoPE: [B*num_queries, num_heads, head_dim]
        # We normalize all input formats to this target shape before applying RoPE.
        if rope_cos is not None:
            if rope_cos.ndim == 2:
                # Case 1: Pre-flattened, head-shared
                # Input: [B*num_queries, head_dim] -> broadcast to all heads
                rope_cos_q = rope_cos.unsqueeze(1).expand(-1, self.multihead_attn.num_heads, -1)
                rope_sin_q = rope_sin.unsqueeze(1).expand(-1, self.multihead_attn.num_heads, -1)
            elif rope_cos.ndim == 3:
                # Case 2: Per-sample, head-shared (triangle_center mode)
                # Input: [B, num_queries, head_dim] -> flatten then broadcast to all heads
                rope_cos_flat = rope_cos.view(B * num_queries, -1)
                rope_cos_q = rope_cos_flat.unsqueeze(1).expand(-1, self.multihead_attn.num_heads, -1)
                rope_sin_flat = rope_sin.view(B * num_queries, -1)
                rope_sin_q = rope_sin_flat.unsqueeze(1).expand(-1, self.multihead_attn.num_heads, -1)
            elif rope_cos.ndim == 4:
                # Case 3: Per-sample, per-head (triangle_mixed mode)
                # Input: [B, num_queries, n_heads, head_dim] -> just flatten batch dim
                rope_cos_q = rope_cos.view(B * num_queries, self.multihead_attn.num_heads, -1)
                rope_sin_q = rope_sin.view(B * num_queries, self.multihead_attn.num_heads, -1)
            else:
                raise ValueError(f"Unexpected rope_cos shape: {rope_cos.shape}, expected 2, 3, or 4 dimensions")
            q = apply_rotary_emb_one_cossin(q, rope_cos_q, rope_sin_q)

        # ========== Apply RoPE to context keys ==========
        # Target shape for k after RoPE: [total_context_tokens, num_kv_heads, head_dim]
        # Context is packed (variable length per sample), so we handle both packed and padded inputs.
        if rope_ctx_cos is not None:
            total_context_tokens = k.shape[0]

            if rope_ctx_cos.ndim == 2:
                # Case 1: Pre-flattened packed format, head-shared
                # Input: [total_context_tokens, head_dim] -> broadcast to all kv_heads
                if rope_ctx_cos.shape[0] != total_context_tokens:
                    raise ValueError(f"rope_ctx_cos shape mismatch: expected {total_context_tokens} tokens, got {rope_ctx_cos.shape[0]}")
                rope_ctx_cos_k = rope_ctx_cos.unsqueeze(1).expand(-1, self.multihead_attn.num_kv_heads, -1)
                rope_ctx_sin_k = rope_ctx_sin.unsqueeze(1).expand(-1, self.multihead_attn.num_kv_heads, -1)
            elif rope_ctx_cos.ndim == 3:
                # Case 2: Per-sample padded format, head-shared
                # Input: [B, seq_len, head_dim] -> flatten to packed then broadcast
                # Note: Caller must ensure padded format matches packed token count
                rope_ctx_cos_flat = rope_ctx_cos.view(-1, rope_ctx_cos.shape[-1])
                if rope_ctx_cos_flat.shape[0] != total_context_tokens:
                    raise ValueError(f"rope_ctx_cos shape mismatch: expected {total_context_tokens} tokens after flattening, got {rope_ctx_cos_flat.shape[0]}")
                rope_ctx_cos_k = rope_ctx_cos_flat.unsqueeze(1).expand(-1, self.multihead_attn.num_kv_heads, -1)
                rope_ctx_sin_flat = rope_ctx_sin.view(-1, rope_ctx_sin.shape[-1])
                rope_ctx_sin_k = rope_ctx_sin_flat.unsqueeze(1).expand(-1, self.multihead_attn.num_kv_heads, -1)
            elif rope_ctx_cos.ndim == 4:
                # Case 3: Per-sample padded format, per-head
                # Input: [B, seq_len, n_heads, head_dim] -> flatten and slice to kv_heads
                rope_ctx_cos_flat = rope_ctx_cos.view(-1, rope_ctx_cos.shape[2], rope_ctx_cos.shape[3])
                if rope_ctx_cos_flat.shape[0] != total_context_tokens:
                    raise ValueError(f"rope_ctx_cos shape mismatch: expected {total_context_tokens} tokens after flattening, got {rope_ctx_cos_flat.shape[0]}")
                rope_ctx_cos_k = rope_ctx_cos_flat[:, :self.multihead_attn.num_kv_heads, :]
                rope_ctx_sin_flat = rope_ctx_sin.view(-1, rope_ctx_sin.shape[2], rope_ctx_sin.shape[3])
                rope_ctx_sin_k = rope_ctx_sin_flat[:, :self.multihead_attn.num_kv_heads, :]
            else:
                raise ValueError(f"Unexpected rope_ctx_cos shape: {rope_ctx_cos.shape}, expected 2, 3, or 4 dimensions")
            k = apply_rotary_emb_one_cossin(k, rope_ctx_cos_k, rope_ctx_sin_k)

        # Flash attention varlen cross
        if ATTN == 'flash_attn':
            # For GQA, repeat k and v
            if self.multihead_attn.is_gqa:
                k = k.repeat_interleave(self.multihead_attn.num_head_per_group, dim=1)
                v = v.repeat_interleave(self.multihead_attn.num_head_per_group, dim=1)
            
            kv = torch.stack([k, v], dim=1)  # [total_context, 2, num_heads, head_dim]
            out = flash_attn_varlen_kvpacked_func(
                q=q,
                kv=kv,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=1.0 / math.sqrt(self.multihead_attn.head_dim),
                causal=False,
            )
        else:
            raise NotImplementedError("varlen cross attention only supports flash_attn for now")

        out = out.view(B, num_queries, D)
        out = self.multihead_attn.out_proj(out)
        query = residual + self.dropout(out)

        # Self attention on queries (if enabled) - match padded forward order
        # Note: queries are always equal length (ray tokens), so we can use regular attention
        # SwinSelfAttention doesn't support varlen, but it works with equal-length sequences
        if self.add_self_attn:
            residual = query
            query = self.self_attn_norm(query)
            if isinstance(self.self_attn, SwinSelfAttention):
                if patch_h is not None and patch_w is not None:
                    q_swin = query.view(B, patch_h, patch_w, D)
                    q_swin = self.self_attn(q_swin)
                    query = q_swin.view(B, num_queries, D)
            else:
                query_self = query.view(B * num_queries, D)
                cu_seqlens_q_self = torch.arange(
                    0, (B + 1) * num_queries, step=num_queries,
                    dtype=torch.int32, device=query.device
                )
                query_self = self.self_attn.forward_varlen(
                    query_self, cu_seqlens_q_self, max_seqlen_q, rope_cos, rope_sin
                )
                query = query_self.view(B, num_queries, D)
            query = residual + self.dropout(query)

        # FFN
        residual = query
        query = self.ffn_norm(query)
        query = self.ffn(query)
        query = residual + self.dropout(query)

        return query


def create_sliding_sink_mask_mod(
    sink_num: torch.Tensor,
    window_size: int,
    lengths: torch.Tensor,
):
    # if isinstance(sink_num, int):
    #     sink_num = [sink_num] * len(lengths)
    def sliding_window(b, h, q_idx, kv_idx):
        return abs(q_idx - kv_idx) <= window_size

    def attn_sink(b, h, q_idx, kv_idx):
        return kv_idx < sink_num[b]

    def sink_tokens(b, h, q_idx, kv_idx):
        return q_idx < sink_num[b]

    def padding_mask_kv(b, h, q_idx, kv_idx):
        return kv_idx < lengths[b]

    def padding_mask_q(b, h, q_idx, kv_idx):
        return q_idx < lengths[b]

    combined_mask = or_masks(sliding_window, attn_sink, sink_tokens)
    combined_mask = and_masks(combined_mask, padding_mask_kv, padding_mask_q)

    return combined_mask


def create_sliding_summary_sink_mask_mod(
    sink_num: int,
    window_size: int,
    lengths: torch.Tensor,
    max_summary_token_cnt: int,
    summary_token_cnts: torch.Tensor,
):
    def sliding_window(b, h, q_idx, kv_idx):
        not_summary_token = (q_idx > max_summary_token_cnt) & (kv_idx > max_summary_token_cnt)
        return (abs(q_idx - kv_idx) <= window_size) & not_summary_token

    def attn_sink(b, h, q_idx, kv_idx):
        # q_is_reg_token = (q_idx >= max_summary_token_cnt - summary_token_cnts[b]) & (q_idx < (sink_num + max_summary_token_cnt))
        # kv_is_reg_token = (kv_idx >= max_summary_token_cnt - summary_token_cnts[b]) & (kv_idx < (sink_num + max_summary_token_cnt))
        q_is_reg_token = (q_idx < (sink_num + max_summary_token_cnt))
        kv_is_reg_token = (kv_idx < (sink_num + max_summary_token_cnt))
        return q_is_reg_token | kv_is_reg_token

    def non_padding_mask(b, h, q_idx, kv_idx):
        q_is_non_padding = (q_idx < lengths[b] + max_summary_token_cnt) & (q_idx >= max_summary_token_cnt - summary_token_cnts[b])
        kv_is_non_padding = (kv_idx < lengths[b] + max_summary_token_cnt) & (kv_idx >= max_summary_token_cnt - summary_token_cnts[b])
        return q_is_non_padding & kv_is_non_padding

    combined_mask = or_masks(sliding_window, attn_sink)
    combined_mask = and_masks(combined_mask, non_padding_mask)

    return combined_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        bias_kv: bool = False,
        activation: str = 'gelu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        norm_first: bool = True,
        rope_dim: Optional[int] = None,
        rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed', 'triangle_center'] = 'triangle',
        rope_double_max_freq: bool = False,
        qk_norm: bool = False,
        use_triangle_ordering: bool = False,
        triangle_ordering_order: Literal['z', 'z-trans', 'hilbert', 'hilbert-trans', 'shuffle-order'] = 'z',
        use_local_attention: bool = False,
        local_attention_window_size_half: int = 256,
        use_flex_attention: bool = False,
        flex_attention_block_size: int = 128,
        add_summary_tokens: bool = False,
        summary_block_size: int = 64,
        input_tri_center_pos: bool = False,
        deepstack_injection_map: Optional[Dict[int, int]] = None,
        deepstack_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        assert norm_first, "Only support norm_first=True"

        # DeepStack texture injection - create projectors internally
        self.deepstack_layer_to_proj = None
        self.deepstack_projectors = None

        if deepstack_injection_map is not None and deepstack_feature_dim is not None:
            # Create per-layer projectors
            self.deepstack_projectors = nn.ModuleList([
                nn.Linear(deepstack_feature_dim, hidden_dim)
                for _ in range(len(deepstack_injection_map))
            ])

            # Map layer_idx -> (projector_idx, feature_idx)
            self.deepstack_layer_to_proj = {
                layer_idx: (proj_idx, feature_idx)
                for proj_idx, (layer_idx, feature_idx) in enumerate(sorted(deepstack_injection_map.items()))
            }
        
        # Store flex attention config
        self.use_flex_attention = use_flex_attention
        self.flex_attention_block_size = flex_attention_block_size

        self.head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList([
            AttentionLayer(
                query_dim=hidden_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                bias=bias,
                bias_kv=bias_kv,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
                use_local_attention=use_local_attention,
                local_attention_window_size_half=local_attention_window_size_half,
                use_flex_attention=use_flex_attention,
            ) for _ in range(num_layers)
        ])
        self.rope_dim = rope_dim
        if rope_dim is not None:
            print(f'Using RoPE with dim {rope_dim} and type {rope_type}')
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type not in ['triangle_mixed', 'triangle_center']:
                assert rope_dim // 2 * 9 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            elif rope_type == 'triangle_center':
                assert rope_dim // 2 * 3 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            elif rope_type == 'triangle_mixed':
                print(f"Overriding rope_dim {rope_dim} with {hidden_dim // num_heads} for triangle_mixed")
                rope_dim = hidden_dim // num_heads
            self.rope_emb = RotaryEmbedding(
                dim=rope_dim,
                freqs_for=rope_type,
                double_max_freq=rope_double_max_freq,
                num_heads=num_heads
            )

        # Ordering
        self.single_ordering = False
        self.ordering_order = None
        self.shuffle_ordering = False
        self.shift_ordering = False
        self.orderings = ['z', 'z-trans', 'hilbert', 'hilbert-trans']
        if use_triangle_ordering:
            if triangle_ordering_order == 'shuffle-order':
                self.shuffle_ordering = True
            elif triangle_ordering_order == 'shift-order':
                self.shift_ordering = True
            else:
                self.single_ordering = True
                self.ordering_order = triangle_ordering_order

        self.local_attention_window_size_half = local_attention_window_size_half

        self.add_summary_tokens = add_summary_tokens
        self.summary_block_size = summary_block_size
        if self.add_summary_tokens:
            self.summary_token_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_tri_center_pos = input_tri_center_pos

        self._grad_checkpointing = False
        self._dump_ordering_result = False
        self._ordering_result_path = None

    @property
    def grad_checkpointing(self):
        return self._grad_checkpointing

    @grad_checkpointing.setter
    def grad_checkpointing(self, value):
        self._grad_checkpointing = value

    @property
    def dump_ordering_result(self):
        return self._dump_ordering_result
    
    @dump_ordering_result.setter
    def dump_ordering_result(self, value):
        self._dump_ordering_result = value

    @property
    def ordering_result_path(self):
        return self._ordering_result_path

    @ordering_result_path.setter
    def ordering_result_path(self, value):
        self._ordering_result_path = value

    def forward(
        self,
        x,
        src_key_padding_mask=None,
        triangle_pos=None,
        num_register_tokens=0,
        env_token_num=0,
        out_layers=[],
        light_token_mask=None,
        deepstack_texture_embs=None,
        triangle_valid_mask=None,
        pre_ordered=False,
        sink_lens=None,
    ):
        """
        Args:
            pre_ordered: If True, sequence is already ordered/summarized by SequenceBuilder.
                        Skip internal reordering and summary generation.
            sink_lens: [B] tensor, sink length per sample. Only used when pre_ordered=True.
        """
        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        bs, src_len = x.shape[:2]

        # Store index_order from spatial reordering for DeepStack
        deepstack_index_order = None
        summary_token_max_cnt = 0
        summary_token_cnts = None

        if pre_ordered:
            # Sequence is already ordered and summarized by SequenceBuilder
            # Skip all reordering and summary generation
            # Just compute RoPE and generate attention mask
            pass
        elif self.single_ordering:
            # Reorder only for an explicitly configured spatial ordering. The
            # full-attention path must continue to accept triangle_pos=None.
            assert triangle_pos is not None, "triangle_pos must be provided if single_ordering is True"
            tri_centers = triangle_pos.reshape(bs, src_len, 3, 3).mean(dim=-2) if not self.input_tri_center_pos else triangle_pos
            index_order, reverse_idx = encode_mask_batch(
                tri_centers[:, num_register_tokens:],
                mask=src_key_padding_mask[:, num_register_tokens:],
                order=self.ordering_order,
                prefix_mask=light_token_mask  # light token mask starts from real triangles
            )
            # Store index_order for DeepStack injection (NOT reverse_idx!)
            deepstack_index_order = index_order
            # print('index_order: ', index_order.shape)
            # print('first sample first 80 indices: ', index_order[0, :80])
            batch_indices = torch.arange(bs).unsqueeze(1).expand(-1, src_len - num_register_tokens)
            x[:, num_register_tokens:] = x[batch_indices, index_order + num_register_tokens]
            tri_centers[:, num_register_tokens:] = tri_centers[batch_indices, index_order + num_register_tokens]
            triangle_pos[:, num_register_tokens:] = triangle_pos[batch_indices, index_order + num_register_tokens]

            # Also reorder masks for DeepStack
            if triangle_valid_mask is not None:
                triangle_valid_mask = triangle_valid_mask[batch_indices, index_order]
            if light_token_mask is not None:
                light_token_mask = light_token_mask[batch_indices, index_order]
            # if self.rope_dim is not None:
            #     rope_cos[:, 0, num_register_tokens:] = rope_cos[batch_indices, 0, index_order + num_register_tokens]
            #     rope_sin[:, 0, num_register_tokens:] = rope_sin[batch_indices, 0, index_order + num_register_tokens]
            if self.dump_ordering_result:
                # _triangle_pos = triangle_pos.clone()
                # _triangle_pos[batch_indices, num_register_tokens:] = _triangle_pos[batch_indices, index_order + num_register_tokens]
                torch.save(triangle_pos, self.ordering_result_path)

            # Add summary tokens
            if self.add_summary_tokens:
                assert src_key_padding_mask is not None, "src_key_padding_mask must be provided if add_summary_tokens is True"
                assert triangle_pos is not None, "triangle_pos must be provided if add_summary_tokens is True"
                assert light_token_mask is None, "can't work with light tokens at front of the sequence"
                summary_tokens = F.avg_pool1d(x[:, num_register_tokens:].transpose(1, 2), kernel_size=self.summary_block_size, stride=self.summary_block_size).transpose(1, 2) + self.summary_token_embed  # [bs, summary_token_max_cnt, hidden_dim]
                summary_token_max_cnt = summary_tokens.shape[1]
                summary_token_cnts = src_key_padding_mask[:, num_register_tokens:].sum(dim=-1) // self.summary_block_size  # only use the full blocks, [bs]
                # Handle different triangle_pos formats: [B, L, 9] or [B, L, 3]
                if self.input_tri_center_pos:
                    # triangle_pos is already [B, L, 3] (center positions only)
                    summary_token_pos = F.avg_pool1d(tri_centers[:, num_register_tokens:].transpose(1, 2), kernel_size=self.summary_block_size, stride=self.summary_block_size).transpose(1, 2)  # [bs, summary_token_max_cnt, 3]
                else:
                    # triangle_pos is [B, L, 9] (full vertex positions), need to expand to 9
                    summary_token_pos = F.avg_pool1d(tri_centers[:, num_register_tokens:].transpose(1, 2), kernel_size=self.summary_block_size, stride=self.summary_block_size).transpose(1, 2).repeat(1, 1, 3)  # [bs, summary_token_max_cnt, 9]
                # reverse concat (so that padding tokens are at the front and the end)
                x = torch.cat([summary_tokens.flip(dims=[1]), x], dim=1)  # [bs, src_len + summary_token_max_cnt, hidden_dim]
                triangle_pos = torch.cat([summary_token_pos.flip(dims=[1]), triangle_pos], dim=1)  # [bs, src_len + summary_token_max_cnt, ...]

            # with open('tmp/debug_summary_forward.pt', 'wb') as f:
            #     torch.save({
            #         'summary_token_cnts': summary_token_cnts,
            #         'summary_token_max_cnt': summary_token_max_cnt,
            #         # 'summary_tokens': summary_tokens,  # Uncomment if needed
            #         'triangle_pos': triangle_pos,
            #         'src_key_padding_mask': src_key_padding_mask,
            #     }, f)
            # exit(0)

        # after altering all the tokens, we can compute the rope cos and sin
        if self.rope_dim is not None:
            assert triangle_pos is not None, "triangle_pos must be provided if rope_dim is not None"
            rope_freqs = self.rope_emb.get_triangle_freqs(triangle_pos)
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
        else:
            rope_cos = rope_sin = None

        # Generate block mask for flex attention at the beginning of pipeline
        block_mask = None
        if self.use_flex_attention:
            assert src_key_padding_mask is not None, "src_key_padding_mask must be provided if use_flex_attention is True"
            if pre_ordered:
                # Use sink_lens from SequenceBuilder
                assert sink_lens is not None, "sink_lens must be provided when pre_ordered=True"
                sink_num = sink_lens.to(x.device)  # [B]
                block_mask_mod = create_sliding_sink_mask_mod(
                    sink_num=sink_num,
                    window_size=self.local_attention_window_size_half,
                    lengths=src_key_padding_mask.sum(dim=-1),
                )
                block_mask = create_block_mask(block_mask_mod, bs, None, src_len, src_len, device=x.device, BLOCK_SIZE=self.flex_attention_block_size)
                src_key_padding_mask = None
            elif self.add_summary_tokens:
                block_mask_mod = create_sliding_summary_sink_mask_mod(
                    sink_num=num_register_tokens,
                    window_size=self.local_attention_window_size_half,
                    lengths=src_key_padding_mask.sum(dim=-1),
                    summary_token_cnts=summary_token_cnts,
                    max_summary_token_cnt=summary_token_max_cnt,
                )
                block_mask = create_block_mask(block_mask_mod, bs, None, src_len + summary_token_max_cnt, src_len + summary_token_max_cnt, device=x.device, BLOCK_SIZE=self.flex_attention_block_size)
                src_key_padding_mask = None
            else:
                if light_token_mask is not None:
                    # put light tokens into the sink
                    light_token_cnt = light_token_mask.sum(dim=-1)
                    sink_num = num_register_tokens + light_token_cnt
                    # print('sink_num: ', sink_num)
                else:
                    sink_num = torch.tensor([num_register_tokens] * bs, device=x.device)
                block_mask_mod = create_sliding_sink_mask_mod(
                    sink_num=sink_num,
                    window_size=self.local_attention_window_size_half,
                    lengths=src_key_padding_mask.sum(dim=-1),
                )
                block_mask = create_block_mask(block_mask_mod, bs, None, src_len , src_len, device=x.device, BLOCK_SIZE=self.flex_attention_block_size)
                src_key_padding_mask = None

        # Per-layer Ordering (skip if pre_ordered)
        if not pre_ordered and (self.shuffle_ordering or self.shift_ordering):
            assert triangle_pos is not None, "triangle_pos must be provided if shuffle_ordering is True"
            bs, src_len = x.shape[:2]
            tri_centers = triangle_pos.reshape(bs, src_len, 3, 3).mean(dim=-2)
            order_cache = {}
            for ordering in self.orderings:
                index_order, reverse_idx = encode_mask_batch(tri_centers, mask=src_key_padding_mask, order=ordering)
                batch_indices = torch.arange(bs).unsqueeze(1).expand(-1, src_len)
                _rope_cos = rope_cos[batch_indices, :, index_order] if rope_cos is not None else None
                _rope_sin = rope_sin[batch_indices, :, index_order] if rope_sin is not None else None
                order_cache[ordering] = (index_order, reverse_idx, _rope_cos, _rope_sin)

        out_list = []

        for i, layer in enumerate(self.layers):
            # ===== DeepStack Injection =====
            if self.deepstack_layer_to_proj is not None and \
               i in self.deepstack_layer_to_proj and \
               deepstack_texture_embs is not None:

                projector_idx, feature_idx = self.deepstack_layer_to_proj[i]
                feature_key = str(feature_idx)

                # Check if this feature is available
                if feature_key in deepstack_texture_embs:
                    # 1. Project texture feature using this layer's projector
                    texture_residual = self.deepstack_projectors[projector_idx](
                        deepstack_texture_embs[feature_key]
                    )  # [B, max_tri, latent_dim]

                    # 2. Reorder texture to match token order (if reordering enabled)
                    # Use index_order (NOT reverse_idx!) to apply the same reordering as tokens
                    if deepstack_index_order is not None:
                        batch_indices_ds = torch.arange(
                            x.size(0), device=x.device
                        ).unsqueeze(1).expand(-1, deepstack_index_order.size(1))
                        texture_residual = texture_residual[batch_indices_ds, deepstack_index_order]

                    # 3. Mask invalid triangles AND light triangles
                    # Note: masks are already reordered if single_ordering is enabled
                    if triangle_valid_mask is not None:
                        texture_residual = texture_residual * triangle_valid_mask.unsqueeze(-1)
                    if light_token_mask is not None:
                        # light_token_mask is [B, max_tri], True for light triangles
                        # Zero out light triangle residuals (only inject to normal triangles)
                        texture_residual = texture_residual * (~light_token_mask).unsqueeze(-1)

                    # 4. Add residual to triangle token positions
                    # If texture_residual is aligned with x (same length), add directly
                    if texture_residual.size(1) == x.size(1):
                         x = x + texture_residual
                    else:
                        # Otherwise assume it maps to the "triangle" section
                        # Skip: summary_tokens (if any) + register_tokens + env_tokens
                        # Token order after summary insertion: [summary] [register] [env] [triangles]
                        tri_start = summary_token_max_cnt + num_register_tokens + env_token_num
                        tri_end = tri_start + texture_residual.size(1)
                        
                        # Ensure we don't go out of bounds (can happen if x is truncated/packed differently)
                        valid_end = min(tri_end, x.size(1))
                        if valid_end > tri_start:
                            x[:, tri_start:valid_end] += texture_residual[:, :valid_end-tri_start]

            if self.shuffle_ordering:
                layer_ordering = random.choice(self.orderings)
            elif self.shift_ordering:
                layer_ordering = self.orderings[i % len(self.orderings)]
            if self.shuffle_ordering or self.shift_ordering:
                index_order, reverse_idx, rope_cos, rope_sin = order_cache[layer_ordering]
                x = x[batch_indices, index_order]

            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin, block_mask=block_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin, block_mask=block_mask)
            if i in out_layers:
                out_list.append(x)
            if self.shuffle_ordering or self.shift_ordering:
                x = x[batch_indices, reverse_idx]

        # Remove summary tokens and reverse ordering (skip if pre_ordered)
        if not pre_ordered:
            if self.add_summary_tokens:
                x = x[:, summary_token_max_cnt:]  # remove summary tokens

            if self.single_ordering:
                x[:, num_register_tokens:] = x[batch_indices, reverse_idx + num_register_tokens]
        
        return x if not out_list else torch.stack(out_list, dim=0)

    def forward_packed(
        self,
        packed_batch,
        rope_emb: Optional[RotaryEmbedding] = None,
        deepstack_texture_embs=None,
        triangle_valid_mask=None,
        light_token_mask=None,
    ):
        """
        Forward pass with packed batch using flash_attn_varlen.

        Args:
            packed_batch: PackedBatch with packed_seq, packed_pos, cu_seqlens, sink_lens
            rope_emb: Optional rotary embedding module (should be self.rope_emb)
            deepstack_texture_embs: [B, max_tri, deepstack_input_dim] for DeepStack injection
            triangle_valid_mask: [B, max_tri] for masking invalid triangles
            light_token_mask: [B, max_tri] True for light triangles - we skip these

        Returns:
            Updated PackedBatch with encoded packed_seq
        """
        from .packed_batch import PackedBatch

        x = packed_batch.packed_seq  # [total_tokens, D]
        cu_seqlens = packed_batch.cu_seqlens
        max_seqlen = packed_batch.max_seq_len
        sink_lens = packed_batch.sink_lens
        B = packed_batch.batch_size

        # Compute RoPE from positions
        # For packed format, we need to compute RoPE per sample and then concatenate
        rope_cos = rope_sin = None
        if rope_emb is not None and self.rope_dim is not None:
            # Compute RoPE for all packed positions at once to avoid per-sample loops.
            packed_pos_batch = packed_batch.packed_pos.unsqueeze(0)  # [1, total_tokens, 3]
            rope_freqs = rope_emb.get_triangle_freqs(packed_pos_batch)  # [1, 1/heads, total_tokens, ...]
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
            if rope_cos.ndim == 4:
                if rope_cos.shape[1] == 1:
                    # [1, 1, total_tokens, head_dim] -> [total_tokens, head_dim]
                    rope_cos = rope_cos.squeeze(0).squeeze(0)
                    rope_sin = rope_sin.squeeze(0).squeeze(0)
                else:
                    # [1, n_heads, total_tokens, head_dim] -> [total_tokens, n_heads, head_dim]
                    rope_cos = rope_cos.squeeze(0).transpose(0, 1)
                    rope_sin = rope_sin.squeeze(0).transpose(0, 1)

        # Forward through layers
        for i, layer in enumerate(self.layers):
            # ===== DeepStack Injection for Varlen =====
            if self.deepstack_layer_to_proj is not None and \
               i in self.deepstack_layer_to_proj and \
               deepstack_texture_embs is not None:

                projector_idx, feature_idx = self.deepstack_layer_to_proj[i]
                feature_key = str(feature_idx)

                # Check if this feature is available
                if feature_key in deepstack_texture_embs:
                    # 1. Project texture feature using this layer's projector
                    texture_residual = self.deepstack_projectors[projector_idx](
                        deepstack_texture_embs[feature_key]
                    )  # [B, max_tri, latent_dim]

                    # 2. Mask invalid and light triangles
                    if triangle_valid_mask is not None:
                        texture_residual = texture_residual * triangle_valid_mask.unsqueeze(-1)
                    if light_token_mask is not None:
                        texture_residual = texture_residual * (~light_token_mask).unsqueeze(-1)

                    # 3. Scatter texture residual into packed sequence at triangle positions
                    # Use is_triangle_mask to identify which packed tokens are triangles
                    if packed_batch.is_triangle_mask is not None:
                        is_triangle_mask = packed_batch.is_triangle_mask  # [B, max_normal_geo]
                        n_normal_geo = packed_batch.n_normal_geo  # [B]

                        # For each sample, inject texture into triangle tokens in normal_geo
                        for b in range(B):
                            start_idx = cu_seqlens[b].item()
                            sink_len = sink_lens[b].item()
                            normal_geo_start = start_idx + sink_len
                            normal_geo_count = n_normal_geo[b].item()

                            # Get triangle mask for this sample's normal_geo
                            sample_tri_mask = is_triangle_mask[b, :normal_geo_count]

                            # Find triangle indices in normal_geo
                            tri_indices_in_normal = torch.where(sample_tri_mask)[0]

                            # Map to global packed positions
                            tri_global_indices = normal_geo_start + tri_indices_in_normal

                            # Get corresponding texture residuals
                            n_valid_tri = min(tri_indices_in_normal.numel(), texture_residual.size(1))

                            if n_valid_tri > 0:
                                # Add texture residual to packed sequence at triangle positions
                                x[tri_global_indices[:n_valid_tri]] += texture_residual[b, :n_valid_tri]

            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer.forward_varlen, x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    use_reentrant=False,
                )
            else:
                x = layer.forward_varlen(
                    x, cu_seqlens, max_seqlen, rope_cos, rope_sin
                )

        # Return updated packed batch
        return PackedBatch(
            packed_seq=x,
            packed_pos=packed_batch.packed_pos,
            cu_seqlens=packed_batch.cu_seqlens,
            sink_lens=packed_batch.sink_lens,
            max_seq_len=packed_batch.max_seq_len,
            is_triangle_mask=packed_batch.is_triangle_mask,
            n_normal_geo=packed_batch.n_normal_geo,
        )

    def forward_packed_with_out_layers(
        self,
        packed_batch,
        rope_emb: Optional[RotaryEmbedding] = None,
        out_layers: Optional[list] = None,
        deepstack_texture_embs=None,
        triangle_valid_mask=None,
        light_token_mask=None,
    ):
        """
        Forward pass with packed batch and return selected layer outputs.

        Args:
            deepstack_texture_embs: [B, max_tri, deepstack_input_dim] for DeepStack injection
            triangle_valid_mask: [B, max_tri] for masking invalid triangles
            light_token_mask: [B, max_tri] True for light triangles

        Returns:
            packed_batch_out: PackedBatch with encoded packed_seq
            out_list: list of packed_seq tensors for requested layers
        """
        from .packed_batch import PackedBatch

        x = packed_batch.packed_seq
        cu_seqlens = packed_batch.cu_seqlens
        max_seqlen = packed_batch.max_seq_len
        sink_lens = packed_batch.sink_lens
        B = packed_batch.batch_size

        rope_cos = rope_sin = None
        if rope_emb is not None and self.rope_dim is not None:
            packed_pos_batch = packed_batch.packed_pos.unsqueeze(0)  # [1, total_tokens, 3]
            rope_freqs = rope_emb.get_triangle_freqs(packed_pos_batch)
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
            if rope_cos.ndim == 4:
                if rope_cos.shape[1] == 1:
                    rope_cos = rope_cos.squeeze(0).squeeze(0)
                    rope_sin = rope_sin.squeeze(0).squeeze(0)
                else:
                    rope_cos = rope_cos.squeeze(0).transpose(0, 1)
                    rope_sin = rope_sin.squeeze(0).transpose(0, 1)

        out_list = []
        if out_layers is None:
            out_layers = []

        for idx, layer in enumerate(self.layers):
            # ===== DeepStack Injection for Varlen =====
            if self.deepstack_layer_to_proj is not None and \
               idx in self.deepstack_layer_to_proj and \
               deepstack_texture_embs is not None:

                projector_idx, feature_idx = self.deepstack_layer_to_proj[idx]
                feature_key = str(feature_idx)

                # Check if this feature is available
                if feature_key in deepstack_texture_embs:
                    # 1. Project texture feature using this layer's projector
                    texture_residual = self.deepstack_projectors[projector_idx](
                        deepstack_texture_embs[feature_key]
                    )  # [B, max_tri, latent_dim]

                    # 2. Mask invalid and light triangles
                    if triangle_valid_mask is not None:
                        texture_residual = texture_residual * triangle_valid_mask.unsqueeze(-1)
                    if light_token_mask is not None:
                        texture_residual = texture_residual * (~light_token_mask).unsqueeze(-1)

                    # 3. Scatter texture residual into packed sequence at triangle positions
                    # Use is_triangle_mask to identify which packed tokens are triangles
                    if packed_batch.is_triangle_mask is not None:
                        is_triangle_mask = packed_batch.is_triangle_mask  # [B, max_normal_geo]
                        n_normal_geo = packed_batch.n_normal_geo  # [B]

                        # For each sample, inject texture into triangle tokens in normal_geo
                        for b in range(B):
                            start_idx = cu_seqlens[b].item()
                            sink_len = sink_lens[b].item()
                            normal_geo_start = start_idx + sink_len
                            normal_geo_count = n_normal_geo[b].item()

                            # Get triangle mask for this sample's normal_geo
                            sample_tri_mask = is_triangle_mask[b, :normal_geo_count]

                            # Find triangle indices in normal_geo
                            tri_indices_in_normal = torch.where(sample_tri_mask)[0]

                            # Map to global packed positions
                            tri_global_indices = normal_geo_start + tri_indices_in_normal

                            # Get corresponding texture residuals
                            n_valid_tri = min(tri_indices_in_normal.numel(), texture_residual.size(1))

                            if n_valid_tri > 0:
                                # Add texture residual to packed sequence at triangle positions
                                x[tri_global_indices[:n_valid_tri]] += texture_residual[b, :n_valid_tri]

            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer.forward_varlen,
                    x,
                    cu_seqlens,
                    max_seqlen,
                    rope_cos,
                    rope_sin,
                    use_reentrant=False,
                )
            else:
                x = layer.forward_varlen(x, cu_seqlens, max_seqlen, rope_cos, rope_sin)
            if idx in out_layers:
                out_list.append(x)

        packed_out = PackedBatch(
            packed_seq=x,
            packed_pos=packed_batch.packed_pos,
            cu_seqlens=packed_batch.cu_seqlens,
            sink_lens=packed_batch.sink_lens,
            max_seq_len=packed_batch.max_seq_len,
        )
        return packed_out, out_list


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        num_kv_heads: Optional[int] = None,
        ctx_dim: Optional[int] = None,
        dropout: float = 0.1,
        include_self_attn: bool = True,
        self_attn_before_cross: bool = False,
        use_swin_attn: bool = False,
        window_size: int = 8,
        shift_size: int = 4,
        activation: str = 'gelu',
        norm_first: bool = True,
        bias: bool = True,
        bias_kv: bool = False,
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        qk_norm: bool = False,
        rope_dim: Optional[int] = None,
        rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed', 'triangle_center'] = 'triangle',
        rope_double_max_freq: bool = False,
        use_triangle_ordering: bool = False,
        triangle_ordering_order: Literal['z', 'z-trans', 'hilbert', 'hilbert-trans', 'shuffle-order', 'shift-order'] = 'z',
        deepstack_injection_map: Optional[Dict[int, int]] = None,
        deepstack_feature_dim: Optional[int] = None,
    ):
        """
        Transformer decoder. Each layer has cross-attention and self-attention.
        Args:
            num_layers (int): number of decoder layers
            num_heads (int): number of attention heads
            hidden_dim (int): hidden dimension
            ffn_hidden_dim (int): feed forward hidden dimension
            ctx_dim (int): context dimension, if None, set to hidden_dim
            dropout (float): dropout rate, default 0.1
            include_self_attn (bool): whether to include self-attention after cross-attention, default True
            self_attn_before_cross (bool): whether to put self-attention before cross-attention, default False
            use_swin_attn (bool): whether to use swin self-attention, default False
            activation (str): activation function, default 'gelu'
            norm_first (bool): whether to apply normalization before attention/ffn, default True
            bias (bool): whether to use bias, default True
            bias_kv (bool): whether to use bias for key and value, default False
            norm_type (str): normalization type, choose from 'layer_norm' and 'rms_norm', default 'layer_norm'
            qk_norm (bool): whether to apply normalization to query and key, default False
            rope_dim (int): rotary position embedding dimension, if None, no RoPE is used
            rope_type (str): rotary position embedding type, choose from 'triangle', 'triangle_learned', 'triangle_mixed', default 'triangle'
            rope_double_max_freq (bool): whether to double the max frequency for RoPE, default False
            use_triangle_ordering (bool): whether to use triangle ordering, default False
            triangle_ordering_order (str): the order to use for triangle ordering, default 'z'
        """
        super().__init__()
        ctx_dim = hidden_dim if ctx_dim is None else ctx_dim
        self.head_dim = hidden_dim // num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            AttentionLayer(
                query_dim=hidden_dim,
                kv_dim=ctx_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias,
                bias_kv=bias_kv,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
                add_self_attn=include_self_attn,
                self_attn_before_cross=self_attn_before_cross,
                use_swin_attn=use_swin_attn,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else shift_size,  # w-attn and swin-attn are on alternate layers
            ) for i in range(num_layers)
        ])

        self.rope_dim = rope_dim
        if rope_dim is not None:
            print(f'Using RoPE with dim {rope_dim} and type {rope_type}')
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type not in ['triangle_mixed', 'triangle_center']:
                assert rope_dim // 2 * 9 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            elif rope_type == 'triangle_center':
                assert rope_dim // 2 * 3 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            elif rope_type == 'triangle_mixed':
                print(f"Overriding rope_dim {rope_dim} with {hidden_dim // num_heads} for triangle_mixed")
                rope_dim = hidden_dim // num_heads
            self.rope_emb = RotaryEmbedding(
                dim=rope_dim,
                freqs_for=rope_type,
                double_max_freq=rope_double_max_freq,
                num_heads=num_heads
            )

        # Triangle ordering (single ordering only, not per-layer like encoder)
        self.use_triangle_ordering = use_triangle_ordering
        self.triangle_ordering_order = triangle_ordering_order

        # DeepStack texture injection - create projectors internally
        self.deepstack_layer_to_proj = None
        self.deepstack_projectors = None

        if deepstack_injection_map is not None and deepstack_feature_dim is not None:
            # Create per-layer projectors
            self.deepstack_projectors = nn.ModuleList([
                nn.Linear(deepstack_feature_dim, hidden_dim)
                for _ in range(len(deepstack_injection_map))
            ])

            # Map layer_idx -> (projector_idx, feature_idx)
            self.deepstack_layer_to_proj = {
                layer_idx: (proj_idx, feature_idx)
                for proj_idx, (layer_idx, feature_idx) in enumerate(sorted(deepstack_injection_map.items()))
            }

        self._grad_checkpointing = False

    @property
    def grad_checkpointing(self):
        return self._grad_checkpointing

    @grad_checkpointing.setter
    def grad_checkpointing(self, value):
        self._grad_checkpointing = value

    def forward(self, x, ctx, src_key_padding_mask=None, triangle_pos=None, ray_pos=None, out_layers=[], tf32_mode=False, patch_h=None, patch_w=None, use_indices=[], num_register_tokens=0, env_token_num=0, deepstack_texture_embs=None, triangle_valid_mask=None, light_token_mask=None):
        if self.rope_dim is not None:
            assert triangle_pos is not None and ray_pos is not None, "triangle_pos and ray_pos must be provided if rope_dim is not None"
            rope_freqs = self.rope_emb.get_triangle_freqs(ray_pos)
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
            rope_ctx_freqs = self.rope_emb.get_triangle_freqs(triangle_pos)
            rope_ctx_cos, rope_ctx_sin = freqs_to_cos_sin(rope_ctx_freqs, head_dim=self.head_dim)
        else:
            rope_cos = rope_sin = rope_ctx_cos = rope_ctx_sin = None

        # Triangle ordering (single ordering, not per-layer like encoder)
        deepstack_index_order = None
        if self.use_triangle_ordering:
            assert triangle_pos is not None, "triangle_pos must be provided if use_triangle_ordering is True"
            bs, ctx_len = ctx.shape[:2]
            tri_centers = triangle_pos.reshape(bs, ctx_len, 3, 3).mean(dim=-2)

            # For decoder, we only support single ordering (not per-layer like encoder)
            # If shuffle-order or shift-order is specified, fall back to 'z' ordering
            ordering = self.triangle_ordering_order
            if ordering in ['shuffle-order', 'shift-order']:
                ordering = 'z'

            # Handle register tokens - exclude them from reordering (similar to encoder)
            index_order, reverse_idx = encode_mask_batch(tri_centers[:, num_register_tokens:], mask=src_key_padding_mask[:, num_register_tokens:], order=ordering)
            deepstack_index_order = index_order  # Store for DeepStack (NOT reverse_idx!)
            batch_indices = torch.arange(bs).unsqueeze(1).expand(-1, ctx_len - num_register_tokens)

            # Only reorder the non-register tokens
            ctx[:, num_register_tokens:] = ctx[batch_indices, index_order + num_register_tokens]

            # Also reorder masks for DeepStack
            if triangle_valid_mask is not None:
                triangle_valid_mask = triangle_valid_mask[batch_indices, index_order]
            if light_token_mask is not None:
                light_token_mask = light_token_mask[batch_indices, index_order]
            if self.rope_dim is not None:
                rope_ctx_cos[:, 0, num_register_tokens:] = rope_ctx_cos[batch_indices, 0, index_order + num_register_tokens]
                rope_ctx_sin[:, 0, num_register_tokens:] = rope_ctx_sin[batch_indices, 0, index_order + num_register_tokens]

        out_list = []
        if not use_indices:
            use_indices = [...] * self.num_layers  # use Ellipsis for default non-interleaved behavior
        for idx, layer in enumerate(self.layers):
            # ===== DeepStack Injection =====
            if self.deepstack_layer_to_proj is not None and \
               idx in self.deepstack_layer_to_proj and \
               deepstack_texture_embs is not None:

                projector_idx, feature_idx = self.deepstack_layer_to_proj[idx]
                feature_key = str(feature_idx)

                if feature_key in deepstack_texture_embs:
                    # 1. Project texture embeddings using this layer's projector
                    texture_residual = self.deepstack_projectors[projector_idx](
                        deepstack_texture_embs[feature_key]
                    )  # [B, max_tri, D]

                    # 2. Reorder texture to match context token order (if reordering enabled)
                    # Use index_order (NOT reverse_idx!) to apply the same reordering as ctx tokens
                    if deepstack_index_order is not None:
                        batch_indices = torch.arange(
                            ctx.size(0), device=ctx.device
                        ).unsqueeze(1).expand(-1, deepstack_index_order.size(1))
                        texture_residual = texture_residual[batch_indices, deepstack_index_order]

                    # 3. Mask invalid triangles AND light triangles
                    # Note: masks are already reordered if triangle_ordering is enabled
                    if triangle_valid_mask is not None:
                        texture_residual = texture_residual * triangle_valid_mask.unsqueeze(-1)
                    if light_token_mask is not None:
                        # Zero out light triangle residuals (only inject to normal triangles)
                        texture_residual = texture_residual * (~light_token_mask).unsqueeze(-1)

                    # 4. Add residual to context triangle token positions
                    # If texture_residual is aligned with ctx (same length), add directly
                    if texture_residual.size(1) == ctx.size(1):
                        ctx = ctx + texture_residual
                    else:
                        # Skip: register_tokens + env_tokens (decoder ctx doesn't have summary tokens)
                        tri_start = num_register_tokens + env_token_num
                        tri_end = tri_start + texture_residual.size(1)
                        
                        # Ensure we don't go out of bounds
                        valid_end = min(tri_end, ctx.size(1))
                        if valid_end > tri_start:
                             ctx[:, tri_start:valid_end] += texture_residual[:, :valid_end-tri_start]
            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, ctx[use_indices[idx]], src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin, rope_ctx_cos=rope_ctx_cos, rope_ctx_sin=rope_ctx_sin, force_sdpa=tf32_mode, patch_h=patch_h, patch_w=patch_w, use_reentrant=False)
            else:
                x = layer(x, ctx[use_indices[idx]], src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin, rope_ctx_cos=rope_ctx_cos, rope_ctx_sin=rope_ctx_sin, force_sdpa=tf32_mode, patch_h=patch_h, patch_w=patch_w)
            if idx in out_layers:
                out_list.append([x])
        return x if not out_list else out_list

    def forward_with_packed_context(
        self,
        query: torch.Tensor,           # [B, num_queries, D] (ray tokens)
        context_packed,                # PackedBatch - encoder output
        query_positions: torch.Tensor = None,  # [B, num_queries, 3] for query RoPE
        patch_h: int = None,          # Height of patches for SwinSelfAttention
        patch_w: int = None,          # Width of patches for SwinSelfAttention
        out_layers: list = None,      # List of layer indices to return outputs from (for DPT)
    ) -> torch.Tensor:
        """
        Cross attention with packed context.

        Args:
            query: [B, num_queries, D] - ray tokens (queries)
            context_packed: PackedBatch - encoder output
            query_positions: Optional positions for query RoPE

        Returns:
            output: [B, num_queries, D]
        """
        from .packed_batch import PackedBatch
        
        B_query, num_queries, D = query.shape
        device = query.device

        # Context batch size (may be different from query batch size if query is flattened)
        B_context = context_packed.batch_size

        # Query cu_seqlens (all samples have same num_queries)
        cu_seqlens_q = torch.arange(
            0, (B_query + 1) * num_queries, step=num_queries,
            dtype=torch.int32, device=device
        )

        # Context cu_seqlens from packed batch
        cu_seqlens_k_context = context_packed.cu_seqlens  # [B_context + 1]
        max_seqlen_k = context_packed.max_seq_len
        
        # For cross attention, cu_seqlens_k must match cu_seqlens_q's batch size
        # If query batch size is larger than context batch size (e.g., query is [B*V, ...] vs context [B, ...]),
        # we need to expand cu_seqlens_k to match query batch size
        if B_query > B_context:
            # Repeat context for each query sample (vectorized, no Python loops).
            num_repeats = B_query // B_context
            lengths = (cu_seqlens_k_context[1:] - cu_seqlens_k_context[:-1]).to(torch.long)  # [B_context]
            lengths_repeated = lengths.repeat_interleave(num_repeats)  # [B_query]
            cu_seqlens_k = torch.zeros(B_query + 1, device=device, dtype=torch.int32)
            if lengths_repeated.numel() > 0:
                cu_seqlens_k[1:] = lengths_repeated.cumsum(0).to(torch.int32)

            # Build indices to repeat each sample's packed block in order.
            max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
            if max_len > 0:
                offsets = torch.arange(max_len, device=device).unsqueeze(0)  # [1, max_len]
                valid = offsets < lengths.unsqueeze(1)  # [B_context, max_len]
                base = cu_seqlens_k_context[:-1].to(torch.long).unsqueeze(1)  # [B_context, 1]
                indices = base + offsets  # [B_context, max_len]
                if num_repeats > 1:
                    indices = indices.repeat_interleave(num_repeats, dim=0)
                    valid = valid.repeat_interleave(num_repeats, dim=0)
                repeat_idx = indices[valid]
            else:
                repeat_idx = torch.empty(0, device=device, dtype=torch.long)
        else:
            cu_seqlens_k = cu_seqlens_k_context
            repeat_idx = None

        # Compute RoPE for queries if needed
        rope_cos = rope_sin = rope_ctx_cos = rope_ctx_sin = None
        if self.rope_dim is not None:
            if query_positions is not None:
                rope_freqs = self.rope_emb.get_triangle_freqs(query_positions)
                rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
                # get_triangle_freqs returns [B, 1, num_queries, ...] or [B, n_heads, num_queries, ...]
                # Squeeze the head dimension if it's 1, or handle multi-head case
                if rope_cos.ndim == 4:
                    if rope_cos.shape[1] == 1:
                        # [B, 1, num_queries, head_dim] -> [B, num_queries, head_dim]
                        rope_cos = rope_cos.squeeze(1)
                        rope_sin = rope_sin.squeeze(1)
                    else:
                        # [B, n_heads, num_queries, head_dim] -> [B, num_queries, n_heads, head_dim]
                        rope_cos = rope_cos.transpose(1, 2)
                        rope_sin = rope_sin.transpose(1, 2)
            
            # Compute RoPE for context in one shot (packed positions).
            packed_ctx_pos = context_packed.packed_pos.unsqueeze(0)  # [1, total_context_tokens, 3]
            rope_ctx_freqs = self.rope_emb.get_triangle_freqs(packed_ctx_pos)
            rope_ctx_cos_packed, rope_ctx_sin_packed = freqs_to_cos_sin(rope_ctx_freqs, head_dim=self.head_dim)
            if rope_ctx_cos_packed.ndim == 4:
                if rope_ctx_cos_packed.shape[1] == 1:
                    rope_ctx_cos_packed = rope_ctx_cos_packed.squeeze(0).squeeze(0)  # [total_context_tokens, head_dim]
                    rope_ctx_sin_packed = rope_ctx_sin_packed.squeeze(0).squeeze(0)
                else:
                    rope_ctx_cos_packed = rope_ctx_cos_packed.squeeze(0).transpose(0, 1)  # [total_context_tokens, n_heads, head_dim]
                    rope_ctx_sin_packed = rope_ctx_sin_packed.squeeze(0).transpose(0, 1)
            
            # If query batch size is larger than context batch size, we need to repeat context RoPE
            # to match the expanded cu_seqlens_k
            # This should match the expansion logic for context_packed_seq_expanded
            if B_query > B_context:
                rope_ctx_cos = rope_ctx_cos_packed.index_select(0, repeat_idx)
                rope_ctx_sin = rope_ctx_sin_packed.index_select(0, repeat_idx)
            else:
                rope_ctx_cos = rope_ctx_cos_packed
                rope_ctx_sin = rope_ctx_sin_packed

        # Forward through layers
        # If query batch size is larger than context batch size, we need to repeat context
        # to match the expanded cu_seqlens_k
        if B_query > B_context:
            context_packed_seq_expanded = context_packed.packed_seq.index_select(0, repeat_idx)
        else:
            context_packed_seq_expanded = context_packed.packed_seq
        
        x = query
        out_list = []
        if out_layers is None:
            out_layers = []
        
        for idx, layer in enumerate(self.layers):
            if self.training and self.grad_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer.forward_cross_varlen,
                    x,
                    context_packed_seq_expanded,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    num_queries,
                    max_seqlen_k,
                    rope_cos,
                    rope_sin,
                    rope_ctx_cos,
                    rope_ctx_sin,
                    patch_h,
                    patch_w,
                    False,  # use_reentrant
                    use_reentrant=False,
                )
            else:
                x = layer.forward_cross_varlen(
                    x,
                    context_packed_seq_expanded,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    num_queries,
                    max_seqlen_k,
                    rope_cos,
                    rope_sin,
                    rope_ctx_cos,
                    rope_ctx_sin,
                    patch_h,
                    patch_w,
                    False,  # use_reentrant
                )
            
            if idx in out_layers:
                out_list.append([x])
        
        if out_list:
            return out_list
        else:
            return x


def map_weights(ref_ckpt):
    """
    Map weights from the reference transformer checkpoint to a new checkpoint format,
    keeping all other keys intact and automatically detecting the transformer prefix.

    :param ref_ckpt: The checkpoint dictionary from the reference model
    :return: A new checkpoint dictionary with mapped transformer weights and all other keys preserved
    """
    new_ckpt = ref_ckpt.copy()  # Start with a copy of the original checkpoint
    
    # Automatically detect the transformer prefix
    transformer_prefixes = [k.rsplit('layers.', 1)[0] for k in ref_ckpt.keys() if 'layers.' in k and ('.self_attn.in_proj_weight' in k or '.mha.in_proj_weight' in k or '.in_proj.bias' in k)]
    if not transformer_prefixes:
        raise ValueError("Could not detect transformer layers in the checkpoint.")
    transformer_prefix = transformer_prefixes[0]  # Use the first detected prefix
    print(f"Detected transformer prefix: {transformer_prefix}")
    
    # Determine the number of layers
    layer_keys = [k for k in ref_ckpt.keys() if k.startswith(f'{transformer_prefix}layers.') and ('.self_attn.in_proj_weight' in k or '.mha.in_proj_weight' in k or '.in_proj.bias' in k)]
    num_layers = len(layer_keys)
    
    for i in range(num_layers):
        old_prefix = f'{transformer_prefix}layers.{i}.'
        new_prefix = f'{transformer_prefix}layers.{i}.'
        
        # Mapping dictionary
        mapping = {
            # nn.TransformerEncoderLayer
            'self_attn.in_proj_weight': 'multihead_attn.in_proj.weight',
            'self_attn.in_proj_bias': 'multihead_attn.in_proj.bias',
            'self_attn.out_proj.weight': 'multihead_attn.out_proj.weight',
            'self_attn.out_proj.bias': 'multihead_attn.out_proj.bias',
            # old implementation
            'in_proj.weight': 'multihead_attn.in_proj.weight',
            'in_proj.bias': 'multihead_attn.in_proj.bias',
            'out_proj.weight': 'multihead_attn.out_proj.weight',
            'out_proj.bias': 'multihead_attn.out_proj.bias',
            # nn.MultiheadAttention
            'mha.in_proj_weight': 'multihead_attn.in_proj.weight',
            'mha.in_proj_bias': 'multihead_attn.in_proj.bias',
            'mha.out_proj.weight': 'multihead_attn.out_proj.weight',
            'mha.out_proj.bias': 'multihead_attn.out_proj.bias',
            # other layers are the same
            'linear1.weight': 'ffn.w1.weight',
            'linear1.bias': 'ffn.w1.bias',
            'linear2.weight': 'ffn.w2.weight',
            'linear2.bias': 'ffn.w2.bias',
            'norm1.weight': 'query_norm.weight',
            'norm1.bias': 'query_norm.bias',
            'norm2.weight': 'ffn_norm.weight',
            'norm2.bias': 'ffn_norm.bias'
        }

        for old_key, new_key in mapping.items():
            old_full_key = f'{old_prefix}{old_key}'
            new_full_key = f'{new_prefix}{new_key}'
            if old_full_key in new_ckpt:
                new_ckpt[new_full_key] = new_ckpt.pop(old_full_key)

    return new_ckpt


def map_cross_attn_layer(old_layer_state_dict):
    
    mapping = {
        'mha.q_proj_weight': 'q_proj.weight',
        'mha.k_proj_weight': 'k_proj.weight',
        'mha.v_proj_weight': 'v_proj.weight',
        'mha.out_proj.weight': 'out_proj.weight',
        'mha.out_proj.bias': 'out_proj.bias',
        'query_norm.weight': 'query_norm.weight',
        'query_norm.bias': 'query_norm.bias',
        'kv_norm.weight': 'kv_norm.weight',
        'kv_norm.bias': 'kv_norm.bias',
        'ffn.w1.weight': 'ffn.w1.weight',
        'ffn.w1.bias': 'ffn.w1.bias',
        'ffn.w2.weight': 'ffn.w2.weight',
        'ffn.w2.bias': 'ffn.w2.bias',
        'ffn.w3.weight': 'ffn.w3.weight',
        'ffn.w3.bias': 'ffn.w3.bias',
        'ffn_norm.weight': 'ffn_norm.weight',
        'ffn_norm.bias': 'ffn_norm.bias'
    }
    
    new_state_dict = {}
    for old_key, new_key in mapping.items():
        new_state_dict[new_key] = old_layer_state_dict[old_key]

    # handle in_proj_bias separately
    q_bias, k_bias, v_bias = old_layer_state_dict['mha.in_proj_bias'].chunk(3)
    new_state_dict['q_proj.bias'] = q_bias
    new_state_dict['k_proj.bias'] = k_bias
    new_state_dict['v_proj.bias'] = v_bias

    return new_state_dict
