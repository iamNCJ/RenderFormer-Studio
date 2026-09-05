# Modified from PointTransformerV3 serialization utilities at commit
# 37a3ddc031de127240ed558a450f6e684647ebe1. Copyright 2023 Pointcept.
# Licensed under MIT; see THIRD_PARTY_NOTICES.md and licenses/PointTransformerV3-MIT.txt.

import os

import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_

# torch.compile the hilbert path on first CUDA call. The function has a tight
# 48-iter Python loop bound by kernel-launch overhead; compile fuses it for a
# bit-exact ~2-3x speedup on real shapes. Set RF_DISABLE_HILBERT_COMPILE=1 to
# opt out (e.g. to debug, or on cpu-only runs).
_compiled_hilbert_encode = None
_compile_hilbert_disabled = os.environ.get("RF_DISABLE_HILBERT_COMPILE", "0") == "1"


def _get_compiled_hilbert_encode():
    global _compiled_hilbert_encode
    if _compiled_hilbert_encode is None:
        _compiled_hilbert_encode = torch.compile(
            hilbert_encode_, dynamic=True, fullgraph=False
        )
    return _compiled_hilbert_encode


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.no_grad()
def encode_mask_batch(xyz, mask=None, depth=16, order="z", stable=False, prefix_mask=None):
    """
    Args:
        xyz: (bs, n_tris, 3)
        mask: (bs, n_tris), things you want to attend to is True
        prefix_mask: (bs, n_tris), things you want to put at the front of the sequence is True
    Returns:
        tris_order: (bs, n_tris)
    """

    bs, n_tris = xyz.shape[:2]
    xyz_max = xyz.max(dim=1).values[:, None]
    xyz_min = xyz.min(dim=1).values[:, None]
    grid_coord = ((xyz - xyz_min) / (xyz_max - xyz_min) * (2 ** depth - 1)).long()
    grid_coord = grid_coord.reshape(-1, 3)

    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    code = code.reshape(bs, n_tris)
    if mask is not None:
        code = code.masked_fill(~mask, torch.iinfo(torch.long).max) # fill masked area with max value to ensure they are at the end of the sequence
    if prefix_mask is not None:
        code = code.masked_fill(prefix_mask, torch.iinfo(torch.long).min) # fill prefix mask with min value to ensure they are at the front of the sequence
    tris_order = torch.argsort(code, dim=1, stable=stable)
    reverse_idx = torch.argsort(tris_order, dim=1)
    return tris_order, reverse_idx


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    global _compile_hilbert_disabled
    if grid_coord.is_cuda and not _compile_hilbert_disabled:
        try:
            return _get_compiled_hilbert_encode()(grid_coord, num_dims=3, num_bits=depth)
        except Exception:
            # On compile failure, fall through to eager. Disable for subsequent
            # calls so we don't pay the failure cost repeatedly.
            _compile_hilbert_disabled = True
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)


# TODO:
# 1. support batch and var len mask (fill masked area with inf)
# 2. then reordering (now after sorting, the masked parts are still behind)
