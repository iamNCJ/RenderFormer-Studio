"""Shared file discovery, H5 parsing, padding, and crop helpers."""

from __future__ import annotations

import glob
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from numpy.typing import DTypeLike
import torch
from natsort import natsorted

from .config import TriangleRenderH5DatasetConfig


def read_list_or_dir(path: str) -> List[str]:
    """Resolve a metadata list, or a directory for legacy CLI compatibility."""

    if os.path.isdir(path):
        return natsorted(glob.glob(os.path.join(path, "*.h5")))
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    raise ValueError(
        f"Invalid H5 source {path!r}; expected a directory or newline-delimited file list."
    )


def resolve_single_file_list(config: TriangleRenderH5DatasetConfig) -> List[str]:
    if not config.h5_folder_path:
        raise ValueError("h5_folder_path must be set for this loader")
    files = read_list_or_dir(config.h5_folder_path)
    if not files:
        raise ValueError(f"No H5 files found in {config.h5_folder_path!r}")
    return files


def resolve_weighted_file_lists(
    config: TriangleRenderH5DatasetConfig,
) -> Tuple[List[List[str]], List[float]]:
    sources = config.h5_lists
    if not sources:
        if not config.h5_folder_path:
            raise ValueError("Either h5_lists or h5_folder_path must be set")
        sources = [config.h5_folder_path]

    file_lists = [read_list_or_dir(source) for source in sources]
    for source, files in zip(sources, file_lists):
        if not files:
            raise ValueError(f"No H5 files found in {source!r}")

    if config.h5_list_weights:
        if len(config.h5_list_weights) != len(file_lists):
            raise ValueError(
                "h5_list_weights and h5_lists must have the same length "
                f"({len(config.h5_list_weights)} != {len(file_lists)})"
            )
        weights = [float(weight) for weight in config.h5_list_weights]
    else:
        weights = [1.0] * len(file_lists)
    if any(weight < 0 for weight in weights) or not any(weights):
        raise ValueError("h5_list_weights must be non-negative with at least one positive value")
    return file_lists, weights


class WeightedFileSampler:
    """Picklable sampler used by PyTorch and DALI worker processes."""

    def __init__(
        self,
        file_lists: Sequence[Sequence[str]],
        weights: Sequence[float],
        rng=None,
    ):
        if not file_lists or len(file_lists) != len(weights):
            raise ValueError("file_lists and weights must be non-empty and aligned")
        self.file_lists = [list(files) for files in file_lists]
        self.weights = list(weights)
        self._indices = list(range(len(self.file_lists)))
        self._rng = random if rng is None else rng

    def pick_file(self) -> str:
        list_index = self._rng.choices(self._indices, weights=self.weights, k=1)[0]
        return self._rng.choice(self.file_lists[list_index])

    def total_files(self) -> int:
        return sum(len(files) for files in self.file_lists)


def pad_first_axis(array: np.ndarray, target_length: int, name: str) -> np.ndarray:
    """Pad an array along axis 0, rejecting silently truncated samples."""

    length = array.shape[0]
    if length > target_length:
        raise ValueError(
            f"{name} has {length} entries, exceeding configured padding_length={target_length}"
        )
    if length == target_length:
        return array
    padding = np.zeros((target_length - length, *array.shape[1:]), dtype=array.dtype)
    return np.concatenate((array, padding), axis=0)


def sample_view_indices(num_views: int, limit: Optional[int]) -> np.ndarray:
    if limit is None or num_views <= limit:
        return np.arange(num_views)
    return np.random.choice(num_views, limit, replace=False)


def load_camera_arrays(
    h5_file: h5py.File,
    img_key: str,
    num_view_limit: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mvp = np.asarray(h5_file["mvp"], dtype=np.float32)
    img = np.asarray(h5_file[img_key], dtype=np.float16)
    c2w = np.asarray(h5_file["c2w"], dtype=np.float32)
    fov = np.asarray(h5_file["fov"], dtype=np.float32)

    if img.ndim == 3:
        img = img[None]
        mvp = mvp[None] if mvp.ndim == 2 else mvp
        c2w = c2w[None] if c2w.ndim == 2 else c2w
        fov = fov.reshape(1)

    indices = sample_view_indices(img.shape[0], num_view_limit)
    return mvp[indices], img[indices], c2w[indices], fov[indices], indices


def load_geometry_arrays(
    h5_file: h5py.File,
    padding_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    triangles = np.asarray(h5_file["triangles"], dtype=np.float32)
    num_triangles = triangles.shape[0]

    if "vn" in h5_file:
        normals = np.asarray(h5_file["vn"], dtype=np.float32)
    else:
        normals = np.zeros_like(triangles)

    triangles = pad_first_axis(triangles, padding_length, "triangles")
    normals = pad_first_axis(normals, padding_length, "vn")
    mask = np.zeros(padding_length, dtype=bool)
    mask[:num_triangles] = True
    return triangles, normals, mask, num_triangles


def load_texture_array(
    h5_file: h5py.File,
    padding_length: int,
    *,
    single_value_texture: bool = False,
    dtype: DTypeLike = np.float16,
) -> np.ndarray:
    texture = np.asarray(h5_file["texture"], dtype=dtype)
    texture = pad_first_axis(texture, padding_length, "texture")
    if single_value_texture:
        texture = texture[..., :1, :1]
    return texture


def load_volume_arrays(
    h5_file: h5py.File,
    volume_padding_length: int,
) -> Dict[str, np.ndarray]:
    if "volume_density" in h5_file:
        volume_density = np.asarray(h5_file["volume_density"], dtype=np.float16)
        arrays = {
            "volume_density": volume_density,
            "volume_position": np.asarray(h5_file["volume_position"], dtype=np.float32),
            "volume_rotation": np.asarray(h5_file["volume_rotation"], dtype=np.float32),
            "volume_scale": np.asarray(h5_file["volume_scale"], dtype=np.float32),
            "volume_scattering": np.asarray(
                h5_file["volume_scattering_scale"], dtype=np.float32
            ),
            "volume_absorption": np.asarray(
                h5_file["volume_absorption_scale"], dtype=np.float32
            ),
        }
        num_volumes = volume_density.shape[0]
    else:
        num_volumes = 0
        arrays = {
            "volume_density": np.zeros((0, 64), dtype=np.float16),
            "volume_position": np.zeros((0, 3), dtype=np.float32),
            "volume_rotation": np.zeros((0, 3), dtype=np.float32),
            "volume_scale": np.zeros((0, 3), dtype=np.float32),
            "volume_scattering": np.zeros((0, 3), dtype=np.float32),
            "volume_absorption": np.zeros((0, 3), dtype=np.float32),
        }

    for name, array in arrays.items():
        if array.shape[0] != num_volumes:
            raise ValueError(
                f"{name} has {array.shape[0]} entries but volume_density has {num_volumes}"
            )
        arrays[name] = pad_first_axis(array, volume_padding_length, name)
    volume_mask = np.zeros(volume_padding_length, dtype=bool)
    volume_mask[:num_volumes] = True
    arrays["volume_mask"] = volume_mask
    return arrays


def sample_aligned_crop_origin(
    source_res: int,
    crop_size: int,
    alignment: int,
    rng,
) -> int:
    max_origin = source_res - crop_size
    if max_origin < 0:
        raise ValueError(
            f"crop_size={crop_size} exceeds source resolution {source_res}"
        )
    if alignment <= 0:
        return rng.randint(0, max_origin)
    max_step = max_origin // alignment
    return rng.randint(0, max_step) * alignment


def crop_square_images(
    images: np.ndarray,
    crop_size: int,
    alignment: int,
    rng,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_views, height, width, channels = images.shape
    if height != width:
        raise ValueError(f"Only square training images are supported, got {height}x{width}")
    origins = np.zeros((num_views, 2), dtype=np.float32)
    cropped = np.empty(
        (num_views, crop_size, crop_size, channels), dtype=images.dtype
    )
    for view_index in range(num_views):
        y0 = sample_aligned_crop_origin(height, crop_size, alignment, rng)
        x0 = sample_aligned_crop_origin(width, crop_size, alignment, rng)
        origins[view_index] = (y0, x0)
        cropped[view_index] = images[
            view_index, y0 : y0 + crop_size, x0 : x0 + crop_size
        ]
    source_resolution = np.full((num_views,), height, dtype=np.float32)
    return cropped, origins, source_resolution


def numpy_sample_to_torch(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Match the historical PyTorch fallback: numeric payloads become float32."""

    result: Dict[str, torch.Tensor] = {}
    for name, value in sample.items():
        tensor = torch.from_numpy(value)
        result[name] = tensor if value.dtype == np.bool_ else tensor.float()
    return result
