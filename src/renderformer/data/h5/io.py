"""Atomic H5 writing shared by RenderFormer data exporters."""

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def atomic_write_h5(
    path: str | Path,
    datasets: dict[str, np.ndarray],
    attrs: dict[str, Any],
    compression: str | None = "gzip",
    compression_opts: int | None = 9,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as f:
        for key, value in datasets.items():
            if compression is None:
                f.create_dataset(key, data=value)
            else:
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                )
        for key, value in attrs.items():
            if isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value, sort_keys=True)
            else:
                f.attrs[key] = value

    os.replace(tmp_path, path)

__all__ = ["atomic_write_h5"]
