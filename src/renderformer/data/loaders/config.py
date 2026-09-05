"""Shared configuration for the published V1 and V2 H5 loaders."""

from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class TriangleRenderH5DatasetConfig:
    """H5 input and padding settings.

    Field names are kept compatible with the historical training CLI and YAML
    files. V1 uses ``h5_folder_path``. V2 can additionally use weighted
    ``h5_lists`` and the crop/volume fields.
    """

    h5_folder_path: Optional[str] = None
    h5_lists: List[str] = dataclasses.field(default_factory=list)
    h5_list_weights: List[float] = dataclasses.field(default_factory=list)
    crop_size: int = 256
    crop_alignment: int = 8
    refresh_after_num_samples: int = 10_000
    shuffle_files: bool = True
    padding_length: int = 5_102
    volume_padding_length: int = 512
    img_key: str = "img"
