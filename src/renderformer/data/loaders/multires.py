"""V2 weighted multi-resolution H5 loader with aligned image crops."""

from __future__ import annotations

import random
import traceback
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    WeightedFileSampler,
    crop_square_images,
    load_camera_arrays,
    load_geometry_arrays,
    load_texture_array,
    load_volume_arrays,
    numpy_sample_to_torch,
    pad_first_axis,
    read_list_or_dir,
    resolve_single_file_list,
    resolve_weighted_file_lists,
    sample_aligned_crop_origin,
)
from .config import TriangleRenderH5DatasetConfig
from .dali import DALIGenericIterator, dali, fn, require_dali, types


_DALI_OUTPUT_KEYS = (
    "mvp",
    "img",
    "triangles",
    "texture",
    "light_strength",
    "mask",
    "c2w",
    "fov",
    "vn",
    "env_map",
    "volume_density",
    "volume_position",
    "volume_rotation",
    "volume_scale",
    "volume_scattering",
    "volume_absorption",
    "volume_mask",
    "crop_origin",
    "source_resolution",
)


def load_v2_numpy_sample(
    file_path: str,
    config: TriangleRenderH5DatasetConfig,
    num_view_limit: Optional[int],
    single_value_texture: bool,
    rng,
    *,
    apply_crop: bool = True,
) -> Dict[str, np.ndarray]:
    """Read one V2 sample, optionally cropping views in source coordinates.

    ``apply_crop=False`` is the full-resolution compatibility path and omits
    ``crop_origin`` and ``source_resolution`` from the returned sample.
    """

    with h5py.File(file_path, "r") as h5_file:
        mvp, img, c2w, fov, _ = load_camera_arrays(
            h5_file, config.img_key, num_view_limit
        )
        if apply_crop:
            img, crop_origin, source_resolution = crop_square_images(
                img, config.crop_size, config.crop_alignment, rng
            )
        triangles, normals, mask, _ = load_geometry_arrays(
            h5_file, config.padding_length
        )
        texture = load_texture_array(
            h5_file,
            config.padding_length,
            single_value_texture=single_value_texture,
        )
        light_strength = pad_first_axis(
            np.asarray(h5_file["light_strength"], dtype=np.float16),
            config.padding_length,
            "light_strength",
        )
        env_map = (
            np.asarray(h5_file["env_map"], dtype=np.float32)
            if "env_map" in h5_file
            else np.zeros((3, 256, 512), dtype=np.float32)
        )
        volumes = load_volume_arrays(h5_file, config.volume_padding_length)

    sample = {
        "mvp": mvp,
        "img": img,
        "triangles": triangles,
        "texture": texture,
        "light_strength": light_strength,
        "mask": mask,
        "c2w": c2w,
        "fov": fov,
        "vn": normals,
        "env_map": env_map,
    }
    if apply_crop:
        sample["crop_origin"] = crop_origin
        sample["source_resolution"] = source_resolution
    sample.update(volumes)
    return sample


class TriangleRenderH5Dataset(Dataset[Dict[str, torch.Tensor]]):
    """Full-resolution V2 fallback for legacy inference/visualization scripts.

    It shares all parsing and padding code with the multires loader but does
    not crop, so migrating the old env/volume callers does not alter pixels.
    """

    def __init__(
        self,
        config: TriangleRenderH5DatasetConfig,
        direct_file_list: Optional[List[str]] = None,
        limit_num_views: Optional[int] = None,
    ):
        self.config = config
        self.file_list = (
            list(direct_file_list)
            if direct_file_list is not None
            else resolve_single_file_list(config)
        )
        if not self.file_list:
            raise ValueError("direct_file_list must not be empty")
        if config.shuffle_files:
            random.shuffle(self.file_list)
        self.limit_num_views = limit_num_views
        print(f"Found {len(self.file_list)} full-resolution V2 H5 files", flush=True)

    def __len__(self) -> int:
        return 100_000_000_000 if self.config.shuffle_files else len(self.file_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_path = (
            random.choice(self.file_list)
            if self.config.shuffle_files
            else self.file_list[index]
        )
        sample = load_v2_numpy_sample(
            file_path,
            self.config,
            self.limit_num_views,
            single_value_texture=False,
            rng=random,
            apply_crop=False,
        )
        return numpy_sample_to_torch(sample)


class TriangleRenderH5MultiresDataset(Dataset[Dict[str, torch.Tensor]]):
    """PyTorch V2 loader for validation and CPU fallback."""

    def __init__(
        self,
        config: TriangleRenderH5DatasetConfig,
        num_view_limit: Optional[int] = None,
        single_value_texture: bool = False,
        seed: Optional[int] = None,
        *,
        limit_num_views: Optional[int] = None,
    ):
        self.config = config
        if num_view_limit is not None and limit_num_views is not None:
            raise ValueError("Set only one of num_view_limit and limit_num_views")
        if limit_num_views is not None:
            num_view_limit = limit_num_views
        self.num_view_limit = 2 if num_view_limit is None else num_view_limit
        self.single_value_texture = single_value_texture
        file_lists, weights = resolve_weighted_file_lists(config)
        self.rng = random.Random(seed)
        self.sampler = WeightedFileSampler(file_lists, weights, rng=self.rng)
        self._ordered_files = [path for files in file_lists for path in files]
        print(
            f"[multires] {len(file_lists)} list(s), "
            f"{self.sampler.total_files()} H5 files, weights={weights}, "
            f"crop_size={config.crop_size}",
            flush=True,
        )

    def __len__(self) -> int:
        return (
            100_000_000_000
            if self.config.shuffle_files
            else self.sampler.total_files()
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_path = (
            self.sampler.pick_file()
            if self.config.shuffle_files
            else self._ordered_files[index]
        )
        sample = load_v2_numpy_sample(
            file_path,
            self.config,
            self.num_view_limit,
            self.single_value_texture,
            self.rng,
        )
        return numpy_sample_to_torch(sample)


class HDF5MultiresExternalSource:
    def __init__(
        self,
        sampler: WeightedFileSampler,
        config: TriangleRenderH5DatasetConfig,
        num_view_limit: int = 2,
        single_value_texture: bool = False,
    ):
        self.sampler = sampler
        self.config = config
        self.num_view_limit = num_view_limit
        self.single_value_texture = single_value_texture
        self._rng = random

    def __call__(self, sample_info):
        while True:
            file_path = self.sampler.pick_file()
            try:
                sample = load_v2_numpy_sample(
                    file_path,
                    self.config,
                    self.num_view_limit,
                    self.single_value_texture,
                    self._rng,
                )
                return tuple(sample[key] for key in _DALI_OUTPUT_KEYS)
            except (OSError, KeyError) as error:
                print(
                    f"Error processing {file_path}: {error}. Trying another sample.\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )


class TriangleRenderDALIDataset:
    """DALI V2 loader used by ``trainer.train_rf2``."""

    def __init__(
        self,
        config: TriangleRenderH5DatasetConfig,
        batch_size: int = 8,
        num_view_limit: int = 2,
        single_value_texture: bool = False,
        num_threads: int = 16,
        num_workers: int = 4,
        device_id: Optional[int] = None,
    ):
        require_dali()
        self.config = config
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.device_id = 0 if device_id is None else device_id

        file_lists, weights = resolve_weighted_file_lists(config)
        if config.shuffle_files:
            for files in file_lists:
                random.shuffle(files)
        self.sampler = WeightedFileSampler(file_lists, weights)
        print(
            f"[multires-DALI] {len(file_lists)} list(s), "
            f"{self.sampler.total_files()} H5 files, weights={weights}, "
            f"crop_size={config.crop_size}, alignment={config.crop_alignment}",
            flush=True,
        )

        self.external_source = HDF5MultiresExternalSource(
            self.sampler,
            config,
            num_view_limit,
            single_value_texture,
        )
        self.pipeline = self.create_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()
        self.iterator = DALIGenericIterator(self.pipeline, list(_DALI_OUTPUT_KEYS))
        self.samples_processed = 0

    def create_pipeline(self):
        @dali.pipeline_def(
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            py_num_workers=self.num_workers,
            device_id=self.device_id,
            py_start_method="spawn",
            set_affinity=True,
        )
        def triangle_render_pipeline():
            outputs = fn.external_source(
                source=self.external_source,
                num_outputs=len(_DALI_OUTPUT_KEYS),
                device="cpu",
                batch=False,
                prefetch_queue_depth=2,
                parallel=True,
                dtype=[
                    types.FLOAT,
                    types.FLOAT16,
                    types.FLOAT,
                    types.FLOAT16,
                    types.FLOAT16,
                    types.BOOL,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT16,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                    types.BOOL,
                    types.FLOAT,
                    types.FLOAT,
                ],
            )
            return tuple(output.gpu() for output in outputs)

        return triangle_render_pipeline()

    def __len__(self) -> int:
        return self.sampler.total_files()

    def __iter__(self):
        while True:
            for data in self.iterator:
                yield data[0]

    def _refresh_file_list(self) -> None:
        file_lists, weights = resolve_weighted_file_lists(self.config)
        if self.config.shuffle_files:
            for files in file_lists:
                random.shuffle(files)
        self.sampler = WeightedFileSampler(file_lists, weights)
        self.external_source.sampler = self.sampler
        self.samples_processed = 0
        print(
            f"[multires-DALI] refreshed: {self.sampler.total_files()} H5 files",
            flush=True,
        )


# Compatibility aliases for downstream notebooks and historical callers.
_MultiListSampler = WeightedFileSampler
_read_list_or_dir = read_list_or_dir
_resolve_file_lists = resolve_weighted_file_lists
_sample_aligned_crop_origin = sample_aligned_crop_origin


def _load_and_crop_sample(
    file_path: str,
    config: TriangleRenderH5DatasetConfig,
    num_view_limit: int,
    single_value_texture: bool,
    rng,
):
    sample = load_v2_numpy_sample(
        file_path,
        config,
        num_view_limit,
        single_value_texture,
        rng,
    )
    return tuple(sample[key] for key in _DALI_OUTPUT_KEYS)


__all__ = [
    "HDF5MultiresExternalSource",
    "TriangleRenderDALIDataset",
    "TriangleRenderH5DatasetConfig",
    "TriangleRenderH5Dataset",
    "TriangleRenderH5MultiresDataset",
    "load_v2_numpy_sample",
]
