from pathlib import Path

import numpy as np

from renderformer.data.capabilities.validator import validate_scene_for_export
from renderformer.data.exporters.common import ExportResult, ObjectPreparedData, PreparedScene
from renderformer.data.h5.io import atomic_write_h5
from renderformer.data.schemas.export_profile import ExportProfile

V1_TEXTURE_SIZE = 32
V1_NUM_CHANNELS = 13


def _rf1_lower_triangular_mask(size: int = V1_TEXTURE_SIZE) -> np.ndarray:
    """Mask used by reference RF1 to_h5.py: cells where x + y > size are zeroed."""
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return x + y <= size


def _v1_texture_for_object(
    obj: ObjectPreparedData,
    *,
    legacy_texture_quantization: bool = False,
) -> np.ndarray:
    """Build (N_i, 13, 32, 32) RF1 texture for one object.

    Channels: [diffuse(3), specular(3), roughness(1), normal(3), irradiance(3)].
    Diffuse is per-face from `obj.face_diffuse`; specular/roughness/normal/emissive
    are homogeneous per object. Normal is the NDC identity (0.5, 0.5, 1.0).
    """
    material = obj.material
    params = material.diffuse_specular_params or {}
    n = obj.triangles.shape[0]
    working_dtype = np.float64 if legacy_texture_quantization else np.float32

    diffuse = obj.face_diffuse.astype(working_dtype)
    if diffuse.shape != (n, 3):
        raise ValueError(f"face_diffuse shape {diffuse.shape} != ({n}, 3)")

    specular = np.asarray(
        params.get("specular", [0.04, 0.04, 0.04]),
        dtype=working_dtype,
    )
    roughness = np.asarray([params.get("roughness", 0.5)], dtype=working_dtype)
    normal_ndc = np.asarray([0.5, 0.5, 1.0], dtype=working_dtype)
    emissive = np.asarray(material.emissive, dtype=working_dtype)
    if emissive.shape != (3,):
        raise ValueError(f"material.emissive must be 3 floats, got shape {emissive.shape}")

    homo_part = np.concatenate([specular, roughness, normal_ndc, emissive])  # (10,)
    per_face = np.concatenate(
        [diffuse, np.broadcast_to(homo_part, (n, 10))], axis=1
    ).astype(working_dtype, copy=False)  # (N, 13)
    if legacy_texture_quantization:
        # The public V1 converter assembled these values in float64 and let
        # h5py cast the final tensor to float16. Quantize before expansion to
        # preserve those values without allocating a float64 (N,13,32,32).
        per_face = per_face.astype(np.float16).astype(np.float32)
    return np.broadcast_to(
        per_face[:, :, None, None],
        (n, V1_NUM_CHANNELS, V1_TEXTURE_SIZE, V1_TEXTURE_SIZE),
    ).astype(np.float32, copy=True)


def export_v1_h5(
    prepared: PreparedScene,
    profile: ExportProfile,
    output_path: str | Path,
    *,
    legacy_texture_quantization: bool = False,
) -> ExportResult:
    validate_scene_for_export(prepared.scene, profile)

    rr = prepared.render_results

    texture_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    vn_chunks: list[np.ndarray] = []
    for obj in prepared.objects:
        texture_chunks.append(
            _v1_texture_for_object(
                obj,
                legacy_texture_quantization=legacy_texture_quantization,
            )
        )
        triangle_chunks.append(obj.triangles)
        vn_chunks.append(obj.vn)
    texture = np.concatenate(texture_chunks, axis=0)
    mask = _rf1_lower_triangular_mask(V1_TEXTURE_SIZE)
    texture[:, :, ~mask] = 0.0

    datasets = {
        "triangles": np.concatenate(triangle_chunks, axis=0).astype(np.float32),
        "vn": np.concatenate(vn_chunks, axis=0).astype(np.float32),
        "texture": texture,
        "c2w": rr.c2w.astype(np.float32),
        "fov": rr.fov.astype(np.float32),
    }
    if rr.mvp is not None:
        datasets["mvp"] = rr.mvp.astype(np.float32)
    if rr.img is not None:
        datasets["img"] = rr.img.astype(np.float16)

    metadata = {
        "format": "v1",
        "schema_version": profile.schema_version,
        "scene_name": prepared.scene.scene_name,
        "texture_encoding": "rf_v1_13ch",
    }
    if legacy_texture_quantization:
        metadata["legacy_float16_value_quantization"] = True
    output_path = Path(output_path)
    atomic_write_h5(output_path, datasets=datasets, attrs={"export_metadata": metadata})
    return ExportResult(output_path=output_path, metadata=metadata)
