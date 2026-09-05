"""Manifest-driven, fail-fast EXR data for material-autoencoder training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MATERIAL_DATASET_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class MaterialEXRRecord:
    """One explicitly cataloged material render."""

    sample_id: str
    path: Path
    split: str


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str):
    raise ValueError(f"strict JSON does not permit {value}")


def load_material_exr_manifest(
    manifest_path: str | Path,
) -> tuple[MaterialEXRRecord, ...]:
    """Load strict JSONL rows without discovering samples from the filesystem."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    records: list[MaterialEXRRecord] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line, parse_constant=_reject_json_constant)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number}: each row must be an object")

            required = {"id", "path", "split"}
            missing = sorted(required - set(payload))
            unknown = sorted(set(payload) - required)
            if missing or unknown:
                raise ValueError(
                    f"{path}:{line_number}: expected exactly id, path, split; "
                    f"missing={missing}, unknown={unknown}"
                )

            sample_id = payload["id"]
            source = payload["path"]
            split = payload["split"]
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id != sample_id.strip()
            ):
                raise ValueError(f"{path}:{line_number}: id must be a non-empty trimmed string")
            if sample_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
            if not isinstance(source, str) or not source:
                raise ValueError(f"{path}:{line_number}: path must be a non-empty string")
            if split not in MATERIAL_DATASET_SPLITS:
                raise ValueError(
                    f"{path}:{line_number}: split must be one of "
                    f"{MATERIAL_DATASET_SPLITS}, got {split!r}"
                )

            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                source_path = path.parent / source_path
            source_path = source_path.resolve()
            if source_path.suffix.lower() != ".exr":
                raise ValueError(f"{path}:{line_number}: expected an .exr path: {source}")
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{path}:{line_number}: EXR does not exist: {source_path}"
                )
            if source_path in seen_paths:
                raise ValueError(
                    f"{path}:{line_number}: duplicate EXR path: {source_path}"
                )

            records.append(MaterialEXRRecord(sample_id, source_path, split))
            seen_ids.add(sample_id)
            seen_paths.add(source_path)

    if not records:
        raise ValueError(f"material EXR manifest contains no records: {path}")
    if not any(record.split == "train" for record in records):
        raise ValueError(f"material EXR manifest contains no train records: {path}")
    return tuple(records)


def _read_material_exr(
    record: MaterialEXRRecord,
    *,
    image_channels: int,
    image_size: int,
    apply_alpha: bool,
) -> torch.Tensor:
    try:
        import simple_exr

        image = np.asarray(simple_exr.read_exr(str(record.path)))
    except Exception as error:
        raise RuntimeError(
            f"failed to read material EXR {record.sample_id!r}: {record.path}: {error}"
        ) from error

    if image.ndim != 3:
        raise ValueError(
            f"material EXR {record.sample_id!r} must be HWC, got {image.shape}: "
            f"{record.path}"
        )
    expected_spatial = (image_size, image_size)
    if image.shape[:2] != expected_spatial:
        raise ValueError(
            f"material EXR {record.sample_id!r} must be {expected_spatial}, got "
            f"{image.shape[:2]}: {record.path}"
        )
    if image.shape[2] not in (image_channels, image_channels + 1):
        raise ValueError(
            f"material EXR {record.sample_id!r} must have {image_channels} RGB "
            f"channels with optional alpha, got {image.shape[2]}: {record.path}"
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(
            f"material EXR {record.sample_id!r} must be floating point, got "
            f"{image.dtype}: {record.path}"
        )
    if not np.isfinite(image).all():
        raise ValueError(
            f"material EXR {record.sample_id!r} contains non-finite values: "
            f"{record.path}"
        )

    rgb = image[..., :image_channels]
    if image.shape[2] == image_channels + 1 and apply_alpha:
        alpha = image[..., image_channels : image_channels + 1]
        if not ((alpha >= 0) & (alpha <= 1)).all():
            raise ValueError(
                f"material EXR {record.sample_id!r} alpha must be in [0, 1]: "
                f"{record.path}"
            )
        rgb = rgb * alpha
    if not (rgb >= 0).all():
        raise ValueError(
            f"material EXR {record.sample_id!r} must contain nonnegative linear "
            f"RGB after alpha compositing: {record.path}"
        )

    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw)


class MaterialEXRDataset(Dataset):
    """Read only the manifest rows assigned to one explicit split.

    Read, shape, dtype, alpha, and finite-value failures are raised with the
    sample ID and path. A failed sample is never replaced with another row.
    """

    def __init__(
        self,
        source: str | Path | Sequence[MaterialEXRRecord],
        *,
        split: str,
        image_channels: int = 3,
        image_size: int = 256,
        apply_alpha: bool = True,
    ) -> None:
        if split not in MATERIAL_DATASET_SPLITS:
            raise ValueError(
                f"split must be one of {MATERIAL_DATASET_SPLITS}, got {split!r}"
            )
        if image_channels < 1:
            raise ValueError("image_channels must be positive")
        if image_size < 1:
            raise ValueError("image_size must be positive")

        if isinstance(source, (str, Path)):
            records = load_material_exr_manifest(source)
        else:
            records = tuple(source)
            if not all(isinstance(record, MaterialEXRRecord) for record in records):
                raise TypeError("source records must all be MaterialEXRRecord instances")
        selected = tuple(record for record in records if record.split == split)
        if not selected:
            raise ValueError(f"material EXR manifest contains no {split!r} records")

        self.records = selected
        self.split = split
        self.image_channels = image_channels
        self.image_size = image_size
        self.apply_alpha = apply_alpha

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image = _read_material_exr(
            record,
            image_channels=self.image_channels,
            image_size=self.image_size,
            apply_alpha=self.apply_alpha,
        )
        return {"id": record.sample_id, "image": image}


__all__ = [
    "MATERIAL_DATASET_SPLITS",
    "MaterialEXRDataset",
    "MaterialEXRRecord",
    "load_material_exr_manifest",
    "sha256_file",
]
