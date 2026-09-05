"""Strict image I/O shared by RenderFormer inference entrypoints."""

from __future__ import annotations

import os

# OpenCV disables its OpenEXR codec unless this opt-in exists before the
# backend is initialized. Keep this above the cv2 import and import this module
# before imageio in callers that also write PNG/video outputs.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from pathlib import Path

import cv2
import numpy as np


def write_hdr_exr(path: str | Path, image: np.ndarray) -> Path:
    """Write one finite HWC RGB float32 array as a float32 OpenEXR image."""

    path = Path(path)
    if path.suffix.lower() != ".exr":
        raise ValueError(f"HDR output path must end in .exr: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"HDR output directory does not exist: {path.parent}")

    array = np.asarray(image)
    if array.dtype != np.float32:
        raise TypeError(f"HDR EXR input must be float32, got {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError(f"HDR EXR input must have nonempty HWC RGB shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("HDR EXR input contains NaN or infinite values")

    # OpenCV's EXR backend accepts BGR. Explicitly request FLOAT so a backend
    # default cannot silently reduce the linear-HDR receipt to half precision.
    bgr = np.ascontiguousarray(array[..., ::-1])
    try:
        written = cv2.imwrite(
            os.fspath(path),
            bgr,
            [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT],
        )
    except cv2.error as exc:
        raise RuntimeError(f"could not encode float32 OpenEXR: {path}") from exc
    if not written:
        raise RuntimeError(f"could not encode float32 OpenEXR: {path}")
    return path
