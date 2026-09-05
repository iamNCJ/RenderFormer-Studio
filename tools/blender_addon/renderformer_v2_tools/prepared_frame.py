"""Pure helpers for the RenderFormer V2 Blender prepared-frame exporter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


SVBRDF_DIFFUSE_SPECULAR = "svbrdf_diffuse_specular"
SVBRDF_METALLIC_ROUGHNESS = "svbrdf_metallic_roughness"
HOMO_DIFFUSE_SPECULAR = "homo_diffuse_specular"
HOMO_METALLIC_ROUGHNESS = "homo_metallic_roughness"
HOMO_METALLIC_ROUGHNESS_TRANSMISSION = "homo_metallic_roughness_transmission"


def first_not_none(*values: Any) -> Any:
    """Return the first non-None value without coercing NumPy arrays to bool."""

    for value in values:
        if value is not None:
            return value
    return None


def constant_image(value: float | list[float] | tuple[float, ...], size: int) -> np.ndarray:
    """Create an RGB float32 image for a missing material map."""

    color = np.asarray(value, dtype=np.float32).reshape(-1)
    if color.size == 1:
        color = np.repeat(color, 3)
    if color.shape != (3,):
        raise ValueError(f"constant image value must be scalar or RGB, got {color.shape}")
    return np.broadcast_to(color, (size, size, 3)).astype(np.float32, copy=True)


def image_or_constant(
    image: np.ndarray | None,
    fallback: float | list[float] | tuple[float, ...],
    size: int,
) -> np.ndarray:
    """Choose an extracted image or a fully materialized constant fallback."""

    if image is None:
        return constant_image(fallback, size)
    data = np.asarray(image, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3 or data.shape[2] == 0:
        raise ValueError(f"texture image must have shape (H, W, C), got {data.shape}")
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    return data[..., :3]


def required_texture_maps(material_type: str, *, use_heightmap: bool = False) -> tuple[str, ...]:
    if material_type == SVBRDF_DIFFUSE_SPECULAR:
        names = ("diffuse.png", "specular.png", "roughness.png", "normal.png")
    elif material_type == SVBRDF_METALLIC_ROUGHNESS:
        names = ("basecolor.png", "metallic.png", "roughness.png", "normal.png")
    else:
        names = ()
    if use_heightmap:
        names += ("height.png",)
    return names


def sanitize_object_key(name: str) -> str:
    """Create a portable, non-empty filename/config key from a Blender object name."""

    key = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return key or "object"


def prepare_frame_directory(path: str | Path) -> Path:
    """Create an output frame only when it cannot retain stale export artifacts."""

    frame_dir = Path(path)
    if frame_dir.exists() and any(frame_dir.iterdir()):
        raise FileExistsError(
            f"prepared-frame output already exists and is not empty: {frame_dir}"
        )
    frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir


def identity_transform() -> dict[str, float | bool]:
    return {
        "translation_x": 0.0,
        "translation_y": 0.0,
        "translation_z": 0.0,
        "rotation_x": 0.0,
        "rotation_y": 0.0,
        "rotation_z": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "scale_z": 1.0,
        "normalize": False,
    }


def build_object_config(
    *,
    obj_key: str,
    material_type: str,
    is_lighting: bool,
    emissive: list[float],
    texture_map_path: str | None = None,
    use_heightmap: bool = False,
    smooth_shading: bool = False,
    diffuse_specular_params: dict[str, Any] | None = None,
    metallic_roughness_params: dict[str, Any] | None = None,
    transmission_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the strict ``GeneratedConfig.ObjectConfig`` JSON representation."""

    if material_type.startswith("svbrdf_") and texture_map_path is None:
        raise ValueError(f"{material_type} requires texture_map_path")
    if not material_type.startswith("svbrdf_") and texture_map_path is not None:
        raise ValueError(f"{material_type} cannot have texture_map_path")
    if use_heightmap and texture_map_path is None:
        raise ValueError("heightmap export requires a texture directory")

    return {
        "mesh_path": f"split/{obj_key}.obj",
        "transform": identity_transform(),
        "is_background": False,
        "is_lighting": is_lighting,
        "material": {
            "material_type": material_type,
            "texture_map_path": texture_map_path,
            "texture_map_aug_random_seed": None,
            "use_heightmap": use_heightmap,
            "diffuse_specular_params": diffuse_specular_params,
            "metallic_roughness_params": metallic_roughness_params,
            "metallic_roughness_transmission_params": transmission_params,
            "measured_brdf_npy_path": None,
            "emissive": emissive,
            "smooth_shading": smooth_shading,
            "rand_tri_diffuse_seed": None,
        },
    }


__all__ = [
    "HOMO_DIFFUSE_SPECULAR",
    "HOMO_METALLIC_ROUGHNESS",
    "HOMO_METALLIC_ROUGHNESS_TRANSMISSION",
    "SVBRDF_DIFFUSE_SPECULAR",
    "SVBRDF_METALLIC_ROUGHNESS",
    "build_object_config",
    "constant_image",
    "first_not_none",
    "identity_transform",
    "image_or_constant",
    "prepare_frame_directory",
    "required_texture_maps",
    "sanitize_object_key",
]
