from pathlib import Path

import numpy as np

from renderformer.data.capabilities.validator import validate_scene_for_export
from renderformer.data.exporters.common import ExportResult, ObjectPreparedData, PreparedScene
from renderformer.data.h5.io import atomic_write_h5
from renderformer.data.schemas.export_profile import ExportProfile, TextureExportMode
from renderformer.data.textures.encoders import build_texture_encoder


_RENDER_PASS_KEYS = ("img", "diffuse_img", "glossy_img", "normal_img", "albedo_img")

# v2 latent layout: [latent(9), normal(3), emissive(3) [+ heightmap(1)]]
_V2_LATENT_CHANNELS = 9
_V2_BASE_CHANNELS = 15  # latent(9) + normal(3) + emissive(3)
_V2_HEIGHTMAP_CHANNELS = 16  # base + heightmap(1)


def _load_measured_latent(material) -> np.ndarray:
    if material.measured_brdf_npy_path is None:
        raise ValueError("homo_measured_brdf requires measured_brdf_npy_path")
    latent = np.asarray(
        np.load(Path(material.measured_brdf_npy_path).expanduser()), dtype=np.float32
    ).reshape(-1)
    if latent.shape[0] != _V2_LATENT_CHANNELS:
        raise ValueError(f"measured BRDF latent must have 9 values, got shape {latent.shape}")
    return latent


def _homogeneous_latent(material) -> np.ndarray:
    """Encode the homogeneous material BRDF parameters to a (9,) latent.

    Lazy-imports brdf_utils so v2.py can be loaded without the BRDF mapper's
    torch / huggingface_hub deps when only doing schema work.
    """
    from renderformer.data.blender.material_types import MATERIAL_TYPE_HOMO_MEASURED_BRDF

    if material.material_type == MATERIAL_TYPE_HOMO_MEASURED_BRDF:
        return _load_measured_latent(material)

    from renderformer.data.blender.brdf_utils import (
        extract_brdf_params_from_material_config,
        map_brdf_to_latent,
    )

    brdf_params = extract_brdf_params_from_material_config(material)
    return map_brdf_to_latent(brdf_params, material.material_type).astype(np.float32)


def _v2_texture_for_object(
    obj: ObjectPreparedData,
    *,
    target_channels: int,
    size: int,
) -> np.ndarray:
    """Return (N_i, target_channels, size, size) v2 texture for one object."""
    if target_channels not in (_V2_BASE_CHANNELS, _V2_HEIGHTMAP_CHANNELS):
        raise ValueError(
            f"v2 target_channels must be 15 or 16, got {target_channels}"
        )

    if obj.svbrdf_texture is not None:
        tex = obj.svbrdf_texture.astype(np.float32, copy=False)
        if tex.shape[1] == target_channels:
            return tex
        if tex.shape[1] == _V2_BASE_CHANNELS and target_channels == _V2_HEIGHTMAP_CHANNELS:
            pad = np.zeros(
                (tex.shape[0], 1, tex.shape[2], tex.shape[3]), dtype=np.float32
            )
            return np.concatenate([tex, pad], axis=1)
        if tex.shape[1] == _V2_HEIGHTMAP_CHANNELS and target_channels == _V2_BASE_CHANNELS:
            return tex[:, :_V2_BASE_CHANNELS]
        raise ValueError(
            f"svbrdf_texture has {tex.shape[1]} channels; cannot adapt to {target_channels}"
        )

    # Homogeneous path
    n = obj.triangles.shape[0]
    latent = _homogeneous_latent(obj.material)  # (9,)
    normal_linear = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # [-1, 1] identity
    emissive = np.asarray(obj.material.emissive, dtype=np.float32)
    if emissive.shape != (3,):
        raise ValueError(f"material.emissive must be 3 floats, got shape {emissive.shape}")

    parts = [latent, normal_linear, emissive]
    if target_channels == _V2_HEIGHTMAP_CHANNELS:
        parts.append(np.array([0.0], dtype=np.float32))  # heightmap=0 means no displacement
    per_face = np.concatenate(parts)  # (target_channels,)
    return np.broadcast_to(
        per_face[None, :, None, None], (n, target_channels, size, size)
    ).astype(np.float32, copy=True)


def _texture_size_for(prepared: PreparedScene) -> int:
    for obj in prepared.objects:
        if obj.svbrdf_texture is not None:
            return int(obj.svbrdf_texture.shape[-1])
    # Default for fully-homogeneous scenes when nothing pins the spatial size.
    return 32


def _light_strength_from_raw_texture(raw_texture: np.ndarray) -> np.ndarray:
    if raw_texture.shape[1] >= 15:
        return np.clip(raw_texture[:, 12:15, 0, 0], 0.0, np.inf).astype(np.float16)
    return np.zeros((raw_texture.shape[0], 3), dtype=np.float16)


def _encode_texture(raw_texture: np.ndarray, profile: ExportProfile) -> tuple[np.ndarray, str]:
    encoder_config = profile.texture_export.texture_encoder or {"type": "raw"}
    encoder = build_texture_encoder(encoder_config)
    triangle_batch_size = int(encoder_config.get("triangle_batch_size", 256))
    encoded = encoder.encode(raw_texture, triangle_batch_size=triangle_batch_size)
    return encoded, encoder.name


def _add_extra_dataset(
    datasets: dict[str, np.ndarray], profile: ExportProfile, key: str, value: np.ndarray
) -> None:
    if key == "env_map" or key.startswith("env_"):
        if not profile.allow_env_map:
            raise ValueError(f"{key} extras require profile.allow_env_map=True")
        datasets[key] = value
        return
    if key.startswith("volume_") or key == "voxel_indices":
        if not profile.allow_volume:
            raise ValueError(f"{key} extras require profile.allow_volume=True")
        datasets[key] = value
        return
    raise ValueError(f"extra dataset {key!r} is not enabled by the export profile")


def export_v2_h5(
    prepared: PreparedScene,
    profile: ExportProfile,
    output_path: str | Path,
) -> ExportResult:
    validate_scene_for_export(prepared.scene, profile)

    rr = prepared.render_results

    target_channels = _V2_HEIGHTMAP_CHANNELS if profile.allow_heightmap else _V2_BASE_CHANNELS
    size = _texture_size_for(prepared)

    texture_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    vn_chunks: list[np.ndarray] = []
    for obj in prepared.objects:
        texture_chunks.append(_v2_texture_for_object(obj, target_channels=target_channels, size=size))
        triangle_chunks.append(obj.triangles)
        vn_chunks.append(obj.vn)
    raw_texture = np.concatenate(texture_chunks, axis=0).astype(np.float32)

    mode = profile.texture_export.mode
    encoder_name = "raw"
    if mode is TextureExportMode.RAW_ONLY:
        primary_texture = raw_texture
    else:
        primary_texture, encoder_name = _encode_texture(raw_texture, profile)

    datasets = {
        "triangles": np.concatenate(triangle_chunks, axis=0).astype(np.float32),
        "vn": np.concatenate(vn_chunks, axis=0).astype(np.float32),
        "texture": primary_texture.astype(np.float16),
        "c2w": rr.c2w.astype(np.float32),
        "fov": rr.fov.astype(np.float32),
        "mvp": (
            rr.mvp.astype(np.float32)
            if rr.mvp is not None
            else np.zeros((rr.c2w.shape[0], 4, 4), dtype=np.float32)
        ),
        "light_strength": _light_strength_from_raw_texture(raw_texture),
    }

    if mode is TextureExportMode.RAW_AND_ENCODED:
        datasets["texture_raw"] = raw_texture.astype(np.float16)

    render_passes = profile.render_passes
    include_all_passes = bool(render_passes.get("all", False))
    for key in _RENDER_PASS_KEYS:
        value = getattr(rr, key)
        if value is not None and (include_all_passes or bool(render_passes.get(key, False))):
            datasets[key] = value.astype(np.float16)

    for key, value in prepared.extra.items():
        if isinstance(value, np.ndarray):
            _add_extra_dataset(datasets, profile, key, value)

    metadata = {
        "format": "v2",
        "schema_version": profile.schema_version,
        "scene_name": prepared.scene.scene_name,
        "texture_export_mode": mode.value,
        "texture_encoder": encoder_name,
        "feature_set": sorted(prepared.scene.feature_set()),
    }

    output_path = Path(output_path)
    atomic_write_h5(output_path, datasets=datasets, attrs={"export_metadata": metadata})
    return ExportResult(output_path=output_path, metadata=metadata)
