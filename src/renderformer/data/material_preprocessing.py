"""Physical-HDR transform used by the released material autoencoder."""

from __future__ import annotations

from typing import Literal

import torch


MaterialPreprocessingMode = Literal["log10_1p"]
MATERIAL_PREPROCESSING_MODES: tuple[MaterialPreprocessingMode, ...] = (
    "log10_1p",
)


def _validate_floating_image(images: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(images):
        raise TypeError(f"{name} must be a torch.Tensor")
    if images.ndim not in (3, 4):
        raise ValueError(f"{name} must be CHW or BCHW, got shape {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {images.dtype}")
    if not bool(torch.isfinite(images).all()):
        raise ValueError(f"{name} contains non-finite values")


def encode_material_hdr(
    images: torch.Tensor,
    *,
    mode: MaterialPreprocessingMode = "log10_1p",
) -> torch.Tensor:
    """Map nonnegative linear RGB into material-autoencoder network space.

    The release transform is ``log10(1 + image)``. Alpha compositing belongs
    to EXR loading and must happen before this function.
    """

    if mode not in MATERIAL_PREPROCESSING_MODES:
        raise ValueError(
            f"unsupported material preprocessing mode {mode!r}; "
            f"choose one of {MATERIAL_PREPROCESSING_MODES}"
        )
    _validate_floating_image(images, name="images")
    if not bool((images >= 0).all()):
        raise ValueError("material HDR input must be nonnegative linear RGB")
    encoded = torch.log10(images + 1.0)
    if not bool(torch.isfinite(encoded).all()):
        raise ValueError("material HDR encoding produced non-finite values")
    return encoded


def decode_material_hdr(
    images: torch.Tensor,
    *,
    mode: MaterialPreprocessingMode = "log10_1p",
) -> torch.Tensor:
    """Invert material network space with ``10**image - 1``."""

    if mode not in MATERIAL_PREPROCESSING_MODES:
        raise ValueError(
            f"unsupported material preprocessing mode {mode!r}; "
            f"choose one of {MATERIAL_PREPROCESSING_MODES}"
        )
    _validate_floating_image(images, name="images")
    decoded = torch.pow(10.0, images) - 1.0
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("material HDR decoding produced non-finite values")
    return decoded


class MaterialPreprocessor:
    """Callable explicit preprocessing contract stored in new checkpoints."""

    def __init__(self, mode: MaterialPreprocessingMode) -> None:
        if mode not in MATERIAL_PREPROCESSING_MODES:
            raise ValueError(
                f"unsupported material preprocessing mode {mode!r}; "
                f"choose one of {MATERIAL_PREPROCESSING_MODES}"
            )
        self.mode = mode

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return encode_material_hdr(images, mode=self.mode)

    def inverse(self, images: torch.Tensor) -> torch.Tensor:
        return decode_material_hdr(images, mode=self.mode)

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode}


__all__ = [
    "MATERIAL_PREPROCESSING_MODES",
    "MaterialPreprocessingMode",
    "MaterialPreprocessor",
    "decode_material_hdr",
    "encode_material_hdr",
]
