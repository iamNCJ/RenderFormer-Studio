"""Pure RF1 schema helpers used by the Blender extension.

This module deliberately has no ``bpy`` dependency so its compatibility contract can be
validated in a normal Python environment.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


TEXTURE_SIZE = 32
TEXTURE_CHANNELS = 13


def angle_weighted_face_vertex_normals(
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_angles: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    """Return one angle-weighted unit normal per face corner without SciPy."""

    summed = np.zeros((vertex_count, 3), dtype=np.float64)
    contributions = face_normals[:, None, :] * face_angles[:, :, None]
    np.add.at(summed, faces.reshape(-1), contributions.reshape(-1, 3))
    lengths = np.linalg.norm(summed, axis=1, keepdims=True)
    unit = np.divide(
        summed,
        lengths,
        out=np.zeros_like(summed),
        where=lengths > np.finfo(np.float64).eps,
    )
    return unit[faces].astype(np.float32)


def lower_triangular_mask(size: int = TEXTURE_SIZE) -> np.ndarray:
    """Return the spatial mask used by the released RF1 texture representation."""

    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return x + y <= size


def make_texture_tile(values: Iterable[float]) -> np.ndarray:
    """Expand one RF1 material vector to a strict float32 ``(13, 32, 32)`` tile."""

    vector = np.asarray(tuple(values), dtype=np.float32)
    if vector.shape != (TEXTURE_CHANNELS,):
        raise ValueError(
            f"RF1 material vector must have shape ({TEXTURE_CHANNELS},), got {vector.shape}"
        )
    tile = np.zeros(
        (TEXTURE_CHANNELS, TEXTURE_SIZE, TEXTURE_SIZE), dtype=np.float32
    )
    tile[:, lower_triangular_mask()] = vector[:, None]
    return tile


def unsupported_v1_features(
    *,
    has_uv_texture: bool = False,
    has_metallic: bool = False,
    has_transmission: bool = False,
    has_displacement: bool = False,
    has_environment_texture: bool = False,
    has_light_object: bool = False,
    has_volume_object: bool = False,
    has_multiple_materials: bool = False,
    has_untriangulated_vertex_colors: bool = False,
    vertex_colors_with_topology_processing: bool = False,
) -> tuple[str, ...]:
    """Describe Blender features that cannot be represented by the RF1 H5 schema."""

    checks = (
        (has_uv_texture, "UV/image textures"),
        (has_metallic, "metallic materials"),
        (has_transmission, "transmission materials"),
        (has_displacement, "height/displacement maps"),
        (has_environment_texture, "environment textures"),
        (has_light_object, "Blender LIGHT objects (use emissive mesh geometry)"),
        (has_volume_object, "volume objects"),
        (has_multiple_materials, "multiple material slots on one mesh"),
        (
            has_untriangulated_vertex_colors,
            "vertex colors on non-triangulated geometry",
        ),
        (
            vertex_colors_with_topology_processing,
            "vertex colors together with normal/topology recomputation",
        ),
    )
    return tuple(label for enabled, label in checks if enabled)


def require_v1_features(**features: bool) -> None:
    unsupported = unsupported_v1_features(**features)
    if unsupported:
        raise ValueError(
            "RF1 export cannot represent: "
            + ", ".join(unsupported)
            + ". Simplify the scene or use the RenderFormer V2 exporter."
        )


def validate_rf1_arrays(
    triangles: np.ndarray,
    vn: np.ndarray,
    texture: np.ndarray,
    c2w: np.ndarray,
    fov: np.ndarray,
) -> None:
    """Validate the exact array contract consumed by the public RF1 pipeline."""

    arrays = {
        "triangles": triangles,
        "vn": vn,
        "texture": texture,
        "c2w": c2w,
        "fov": fov,
    }
    for name, value in arrays.items():
        if value.dtype != np.dtype("float32"):
            raise ValueError(f"RF1 {name} must be float32, got {value.dtype}")

    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(f"RF1 triangles must have shape (N, 3, 3), got {triangles.shape}")
    if vn.shape != triangles.shape:
        raise ValueError(f"RF1 vn must match triangles, got {vn.shape}")
    expected_texture = (triangles.shape[0], TEXTURE_CHANNELS, TEXTURE_SIZE, TEXTURE_SIZE)
    if texture.shape != expected_texture:
        raise ValueError(
            f"RF1 texture must have shape {expected_texture}, got {texture.shape}"
        )
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"RF1 c2w must have shape (V, 4, 4), got {c2w.shape}")
    if fov.shape != (c2w.shape[0],):
        raise ValueError(f"RF1 fov must have shape ({c2w.shape[0]},), got {fov.shape}")


__all__ = [
    "TEXTURE_CHANNELS",
    "TEXTURE_SIZE",
    "angle_weighted_face_vertex_normals",
    "lower_triangular_mask",
    "make_texture_tile",
    "require_v1_features",
    "unsupported_v1_features",
    "validate_rf1_arrays",
]
