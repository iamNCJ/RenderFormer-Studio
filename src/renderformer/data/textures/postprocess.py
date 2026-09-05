from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from renderformer.data.envmap import prepare_raw_env_map
from renderformer.data.h5.io import atomic_write_h5
from renderformer.data.textures.encoders import TextureEncoder, build_texture_encoder


DEFAULT_DROP_KEYS = (
    "albedo_img",
    "diffuse_img",
    "glossy_img",
    "normal_img",
)


def _light_strength_from_raw_texture(raw_texture: np.ndarray) -> np.ndarray:
    if raw_texture.shape[1] >= 15:
        return np.clip(raw_texture[:, 12:15, 0, 0], 0.0, np.inf).astype(np.float16)
    return np.zeros((raw_texture.shape[0], 3), dtype=np.float16)


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return dict(json.loads(value))
    if isinstance(value, bytes):
        return dict(json.loads(value.decode("utf-8")))
    if isinstance(value, dict):
        return dict(value)
    return {}


def _rotate_env_map_to_chw(env_map: np.ndarray, rotation_degrees: np.ndarray, strength: float) -> np.ndarray:
    env_tensor = torch.from_numpy(np.asarray(env_map, dtype=np.float32))
    if env_tensor.ndim == 3 and env_tensor.shape[0] in (3, 4) and env_tensor.shape[-1] not in (3, 4):
        env_tensor = env_tensor[:3].permute(1, 2, 0)
    return prepare_raw_env_map(
        env_tensor,
        torch.from_numpy(np.asarray(rotation_degrees, dtype=np.float32)),
        strength,
    ).numpy().astype(np.float32)


def _postprocess_env_map(datasets: dict[str, np.ndarray]) -> None:
    if "env_map" not in datasets:
        return
    env_map = np.asarray(datasets["env_map"], dtype=np.float32)
    rotation = np.asarray(datasets.pop("env_map_rotation", np.zeros(3, dtype=np.float32)), dtype=np.float32)
    strength = float(np.asarray(datasets.pop("env_map_strength", np.ones(1, dtype=np.float32))).reshape(-1)[0])
    datasets["env_map"] = _rotate_env_map_to_chw(env_map, rotation, strength)


def _postprocess_volume(datasets: dict[str, np.ndarray]) -> None:
    volume_data = datasets.pop("volume_data", None)
    datasets.pop("volume_indices", None)
    datasets.pop("voxel_indices", None)
    if volume_data is None:
        return

    volume_data_np = np.asarray(volume_data, dtype=np.float32)
    if volume_data_np.size == 0 or volume_data_np.shape[0] == 0:
        datasets["volume_density"] = np.zeros((1, 64), dtype=np.float32)
        datasets["volume_scattering_scale"] = np.zeros((1, 3), dtype=np.float32)
        datasets["volume_absorption_scale"] = np.zeros((1, 3), dtype=np.float32)
        datasets["volume_position"] = np.zeros((1, 3), dtype=np.float32)
        datasets["volume_rotation"] = np.zeros((1, 3), dtype=np.float32)
        datasets["volume_scale"] = np.zeros((1, 3), dtype=np.float32)
        return

    if volume_data_np.ndim == 4 and volume_data_np.shape[1:] == (4, 4, 4):
        datasets["volume_density"] = volume_data_np.reshape(volume_data_np.shape[0], -1).astype(np.float32)
    elif volume_data_np.ndim == 2 and volume_data_np.shape[1] == 64:
        datasets["volume_density"] = volume_data_np.astype(np.float32)
    elif volume_data_np.ndim == 7 and volume_data_np.shape[1:] == (8, 4, 8, 4, 8, 4):
        raise ValueError(
            "volume_data has un-compacted micro-grid shape "
            f"{volume_data_np.shape}; regenerate it with renderformer.data.blender.scene_builder "
            "so occupied coarse voxels are exported as (N, 4, 4, 4)."
        )
    else:
        raise ValueError(
            "volume_data must have shape (N, 4, 4, 4) or (N, 64), "
            f"got {volume_data_np.shape}"
        )

    for key in (
        "volume_scattering_scale",
        "volume_absorption_scale",
        "volume_position",
        "volume_rotation",
        "volume_scale",
    ):
        if key in datasets:
            datasets[key] = np.asarray(datasets[key], dtype=np.float32)


def postprocess_v2_h5(
    input_path: str | Path,
    output_path: str | Path,
    texture_encoder_config: dict,
    keep_raw: bool,
    *,
    texture_encoder: TextureEncoder | None = None,
    drop_keys: Iterable[str] = (),
) -> None:
    with h5py.File(input_path, "r") as f:
        datasets = {key: f[key][:] for key in f.keys()}
        attrs = dict(f.attrs.items())

    raw_texture = np.asarray(datasets.pop("texture"), dtype=np.float32)
    datasets.pop("texture_raw", None)

    encoder = texture_encoder or build_texture_encoder(texture_encoder_config)
    triangle_batch_size = int(texture_encoder_config.get("triangle_batch_size", 256))
    datasets["texture"] = encoder.encode(
        raw_texture, triangle_batch_size=triangle_batch_size
    ).astype(np.float16)

    if keep_raw:
        datasets["texture_raw"] = raw_texture.astype(np.float16)
    if "light_strength" not in datasets:
        datasets["light_strength"] = _light_strength_from_raw_texture(raw_texture)

    _postprocess_env_map(datasets)
    _postprocess_volume(datasets)
    for key in drop_keys:
        datasets.pop(key, None)

    metadata = _decode_metadata(attrs.get("export_metadata"))
    metadata.update(
        {
            "format": "v2",
            "postprocessed": True,
            "texture_encoder": encoder.name,
        }
    )
    attrs["export_metadata"] = metadata

    atomic_write_h5(output_path, datasets=datasets, attrs=attrs)


def postprocess_v2_h5_batch(
    jobs: Iterable[tuple[str | Path, str | Path]],
    texture_encoder_config: dict,
    *,
    keep_raw: bool = False,
    skip_existing: bool = False,
    drop_keys: Iterable[str] = DEFAULT_DROP_KEYS,
    texture_encoder: TextureEncoder | None = None,
) -> list[Path]:
    """Postprocess multiple H5 files while reusing one texture encoder.

    Learned encoders own a large model and are intentionally instantiated once
    in the caller process. Only the ``raw`` encoder can use process-level
    parallelism; each learned-encoder worker would load another full model and
    can exhaust GPU memory.
    """

    materialized_jobs = [(Path(src), Path(dst)) for src, dst in jobs]
    if not materialized_jobs:
        raise ValueError("postprocess_v2_h5_batch requires at least one job")

    output_paths = [output_path.resolve() for _, output_path in materialized_jobs]
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("postprocess jobs must have unique output paths")

    for input_path, _ in materialized_jobs:
        if not input_path.is_file():
            raise FileNotFoundError(f"Input H5 file does not exist: {input_path}")

    pending_jobs = [
        (input_path, output_path)
        for input_path, output_path in materialized_jobs
        if not (skip_existing and output_path.exists())
    ]
    if not pending_jobs:
        return [output_path for _, output_path in materialized_jobs]

    encoder = texture_encoder or build_texture_encoder(texture_encoder_config)
    for input_path, output_path in pending_jobs:
        postprocess_v2_h5(
            input_path,
            output_path,
            texture_encoder_config,
            keep_raw,
            texture_encoder=encoder,
            drop_keys=drop_keys,
        )
    return [output_path for _, output_path in materialized_jobs]
