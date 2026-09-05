"""Schema validation for generated RenderFormer H5 files."""

from pathlib import Path

import h5py
import numpy as np


class H5ValidationError(ValueError):
    pass


REQUIRED_KEYS = {
    "v1": ("triangles", "texture", "vn", "c2w", "fov"),
    "v2": ("triangles", "texture", "vn", "c2w", "fov", "mvp"),
}


def validate_h5_file(path: str | Path, export_format: str) -> None:
    if export_format not in REQUIRED_KEYS:
        raise H5ValidationError(f"unsupported h5 format: {export_format}")

    with h5py.File(path, "r") as f:
        for key in REQUIRED_KEYS[export_format]:
            if key not in f:
                raise H5ValidationError(f"missing required key: {key}")

        triangles = f["triangles"]
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise H5ValidationError(
                f"triangles must have shape (N, 3, 3), got {triangles.shape}"
            )
        if triangles.dtype != np.dtype("float32"):
            raise H5ValidationError(
                f"triangles must have dtype float32, got {triangles.dtype}"
            )

        texture = f["texture"]
        if texture.shape[0] != triangles.shape[0]:
            raise H5ValidationError(
                "texture and triangles must have the same first dimension"
            )
        if export_format == "v1":
            if texture.dtype != np.dtype("float32"):
                raise H5ValidationError(
                    f"v1 texture must have dtype float32, got {texture.dtype}"
                )
            if texture.shape != (triangles.shape[0], 13, 32, 32):
                raise H5ValidationError(
                    f"v1 texture must have shape (N, 13, 32, 32), got {texture.shape}"
                )
        else:
            if texture.dtype not in (np.dtype("float16"), np.dtype("float32")):
                raise H5ValidationError(
                    "v2 texture must have floating dtype float16 or float32, "
                    f"got {texture.dtype}"
                )
            if (
                texture.ndim != 4
                or texture.shape[1] <= 0
                or texture.shape[2] <= 0
                or texture.shape[3] <= 0
            ):
                raise H5ValidationError(
                    f"v2 texture must have shape (N, C, H, W), got {texture.shape}"
                )

        vn = f["vn"]
        if vn.shape != triangles.shape:
            raise H5ValidationError(
                f"vn must match triangles shape, got {vn.shape} and {triangles.shape}"
            )
        if vn.dtype != np.dtype("float32"):
            raise H5ValidationError(f"vn must have dtype float32, got {vn.dtype}")

        c2w = f["c2w"]
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise H5ValidationError(
                f"c2w must have shape (V, 4, 4), got {c2w.shape}"
            )
        if c2w.dtype != np.dtype("float32"):
            raise H5ValidationError(f"c2w must have dtype float32, got {c2w.dtype}")

        fov = f["fov"]
        if fov.ndim != 1 or fov.shape[0] != c2w.shape[0]:
            raise H5ValidationError(
                f"fov must have shape (V,), got {fov.shape} for c2w views {c2w.shape[0]}"
            )
        if fov.dtype != np.dtype("float32"):
            raise H5ValidationError(f"fov must have dtype float32, got {fov.dtype}")

        if "mvp" in f:
            mvp = f["mvp"]
            if (
                mvp.ndim != 3
                or mvp.shape[1:] != (4, 4)
                or mvp.shape[0] != c2w.shape[0]
            ):
                raise H5ValidationError(
                    f"mvp must have shape (V, 4, 4), got {mvp.shape}"
                )
            if mvp.dtype != np.dtype("float32"):
                raise H5ValidationError(
                    f"mvp must have dtype float32, got {mvp.dtype}"
                )

__all__ = ["H5ValidationError", "validate_h5_file"]
