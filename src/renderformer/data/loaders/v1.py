"""V1 RenderFormer H5 loaders used by the released RF1 training scripts."""

from __future__ import annotations

import random
import traceback
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    load_camera_arrays,
    load_geometry_arrays,
    load_texture_array,
    numpy_sample_to_torch,
    resolve_single_file_list,
)
from .config import TriangleRenderH5DatasetConfig
from .dali import DALIGenericIterator, dali, fn, require_dali, types


SampleLoader = Callable[..., Dict[str, np.ndarray]]

_DALI_OUTPUT_KEYS = (
    "mvp",
    "img",
    "triangles",
    "texture",
    "mask",
    "c2w",
    "fov",
    "vn",
)


def load_v1_numpy_sample(
    file_path: str,
    config: TriangleRenderH5DatasetConfig,
    num_view_limit: Optional[int],
    single_value_texture: bool,
) -> Dict[str, np.ndarray]:
    """Read and pad one V1 sample in the canonical training schema."""

    with h5py.File(file_path, "r") as h5_file:
        mvp, img, c2w, fov, _ = load_camera_arrays(
            h5_file, config.img_key, num_view_limit
        )
        triangles, normals, mask, _ = load_geometry_arrays(
            h5_file, config.padding_length
        )
        texture = load_texture_array(
            h5_file,
            config.padding_length,
            single_value_texture=single_value_texture,
            dtype=np.float32,
        )
    return {
        "mvp": mvp,
        "img": img,
        "triangles": triangles,
        "texture": texture,
        "mask": mask,
        "c2w": c2w,
        "fov": fov,
        "vn": normals,
    }


class TriangleRenderH5Dataset(Dataset[Dict[str, torch.Tensor]]):
    """Simple PyTorch V1 dataset for validation and CPU fallback."""

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
        print(f"Found {len(self.file_list)} V1 H5 files", flush=True)

    def __len__(self) -> int:
        return 100_000_000_000 if self.config.shuffle_files else len(self.file_list)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_path = (
            random.choice(self.file_list)
            if self.config.shuffle_files
            else self.file_list[index]
        )
        sample = load_v1_numpy_sample(
            file_path,
            self.config,
            self.limit_num_views,
            single_value_texture=False,
        )
        return numpy_sample_to_torch(sample)


class HDF5ExternalSource:
    """Retrying DALI external source; corrupt samples are replaced by another file."""

    def __init__(
        self,
        file_list: List[str],
        config: TriangleRenderH5DatasetConfig,
        num_view_limit: int = 2,
        single_value_texture: bool = False,
        sample_loader: SampleLoader = load_v1_numpy_sample,
    ):
        self.file_list = file_list
        self.config = config
        self.num_view_limit = num_view_limit
        self.single_value_texture = single_value_texture
        self.sample_loader = sample_loader

    def __call__(self, sample_info):
        while True:
            file_path = random.choice(self.file_list)
            try:
                sample = self.sample_loader(
                    file_path,
                    self.config,
                    self.num_view_limit,
                    self.single_value_texture,
                )
                # The historical DALI path deliberately used fp16 textures,
                # while the PyTorch/RF1 compatibility path retained fp32.
                sample["texture"] = sample["texture"].astype(
                    np.float16, copy=False
                )
                return tuple(sample[key] for key in _DALI_OUTPUT_KEYS)
            except (OSError, KeyError) as error:
                print(
                    f"Error processing {file_path}: {error}. Trying another sample.\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )


class TriangleRenderDALIDataset:
    """GPU-prefetching V1 dataset used by the published training entrypoints."""

    def __init__(
        self,
        config: TriangleRenderH5DatasetConfig,
        batch_size: int = 8,
        num_view_limit: int = 2,
        single_value_texture: bool = False,
        num_threads: int = 16,
        num_workers: int = 4,
        device_id: Optional[int] = None,
        *,
        _sample_loader: SampleLoader = load_v1_numpy_sample,
    ):
        require_dali()
        self.config = config
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.device_id = 0 if device_id is None else device_id
        self.file_list = resolve_single_file_list(config)
        if config.shuffle_files:
            random.shuffle(self.file_list)
        print(f"Found {len(self.file_list)} V1 H5 files", flush=True)

        self.external_source = HDF5ExternalSource(
            self.file_list,
            config,
            num_view_limit,
            single_value_texture,
            sample_loader=_sample_loader,
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
                    types.BOOL,
                    types.FLOAT,
                    types.FLOAT,
                    types.FLOAT,
                ],
            )
            return tuple(output.gpu() for output in outputs)

        return triangle_render_pipeline()

    def __len__(self) -> int:
        return len(self.file_list)

    def __iter__(self):
        while True:
            for data in self.iterator:
                yield data[0]

    def _refresh_file_list(self) -> None:
        self.file_list = resolve_single_file_list(self.config)
        if self.config.shuffle_files:
            random.shuffle(self.file_list)
        self.external_source.file_list = self.file_list
        self.samples_processed = 0
        print(f"Refreshed V1 file list: {len(self.file_list)} files", flush=True)


def create_dataloader(
    dataloader_type: str,
    config: TriangleRenderH5DatasetConfig,
    batch_size: int = 8,
    num_view_limit: int = 2,
    single_value_texture: bool = False,
    num_workers: int = 4,
    shuffle: bool = True,
    endless: bool = True,
    **kwargs,
):
    """Create one of the two supported V1 loaders: PyTorch or DALI."""

    del endless  # Retained for source compatibility with the retired Ray backend.
    kind = dataloader_type.lower()
    if kind == "pytorch":
        from torch.utils.data import DataLoader

        dataset = TriangleRenderH5Dataset(config, limit_num_views=num_view_limit)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle and not config.shuffle_files,
            **kwargs,
        )
    if kind == "dali":
        return TriangleRenderDALIDataset(
            config,
            batch_size=batch_size,
            num_view_limit=num_view_limit,
            single_value_texture=single_value_texture,
            num_workers=num_workers,
            **kwargs,
        )
    raise ValueError("dataloader_type must be 'pytorch' or 'dali'")


__all__ = [
    "HDF5ExternalSource",
    "TriangleRenderDALIDataset",
    "TriangleRenderH5Dataset",
    "TriangleRenderH5DatasetConfig",
    "create_dataloader",
    "load_v1_numpy_sample",
]
