"""Deterministic preprocessing for external environment-map collections."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from multiprocessing import current_process, get_context
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Iterable
import unicodedata

import numpy as np


CHANNEL_PERMUTATIONS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("RGB", (0, 1, 2)),
    ("RBG", (0, 2, 1)),
    ("GRB", (1, 0, 2)),
    ("GBR", (1, 2, 0)),
    ("BRG", (2, 0, 1)),
    ("BGR", (2, 1, 0)),
)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 0x00000001
_RENAME_EXCL = 0x00000004


@dataclass(frozen=True)
class ChannelShuffleResult:
    output_root: Path
    output_list: Path
    manifest: Path
    source_count: int
    output_count: int


@dataclass(frozen=True)
class _SiblingLock:
    path: Path
    file_descriptor: int
    device: int
    inode: int


def channel_shuffle_variants(
    image: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the historical six RGB channel permutations as float32 arrays."""

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            "environment map must have shape (H,W,C) with at least three channels; "
            f"got {image.shape}"
        )
    rgb = image[..., :3]
    return tuple(
        (
            name,
            np.ascontiguousarray(rgb[..., list(indices)], dtype=np.float32),
        )
        for name, indices in CHANNEL_PERMUTATIONS
    )


def _parse_env_map_stems(text: str, *, list_path: Path) -> tuple[str, ...]:
    stems: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stem = raw_line.strip()
        if not stem:
            continue
        relative = Path(stem)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in stem
            or relative.suffix.lower() == ".exr"
        ):
            raise ValueError(
                f"{list_path}:{line_number}: expected a safe relative stem without "
                f"'.exr', got {stem!r}"
            )
        normalized = relative.as_posix()
        if normalized == ".":
            raise ValueError(
                f"{list_path}:{line_number}: environment-map stem must name a file"
            )
        if normalized in seen:
            raise ValueError(
                f"{list_path}:{line_number}: duplicate environment-map stem "
                f"{normalized!r}"
            )
        seen.add(normalized)
        stems.append(normalized)

    if not stems:
        raise ValueError(f"environment-map input list is empty: {list_path}")
    return tuple(stems)


def load_env_map_stems(path: str | Path) -> tuple[str, ...]:
    """Load a metadata list of safe, unique EXR stems without scanning a tree."""

    list_path = Path(path).expanduser().resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"environment-map input list is missing: {list_path}")
    return _parse_env_map_stems(
        list_path.read_text(encoding="utf-8"), list_path=list_path
    )


def _variant_stem(source_stem: str, permutation_name: str) -> str:
    source = Path(source_stem)
    return (source.parent / f"{source.name}_{permutation_name}").as_posix()


def _all_variant_stems(source_stems: Iterable[str]) -> tuple[str, ...]:
    # The recoverable historical non-blank list was globally sorted. This
    # produces BGR, BRG, GBR, GRB, RBG, RGB within a common source prefix.
    return tuple(
        sorted(
            _variant_stem(source_stem, name)
            for source_stem in source_stems
            for name, _ in CHANNEL_PERMUTATIONS
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _process_channel_shuffle_source(
    task: tuple[str, str, str],
) -> tuple[str, dict[str, object]]:
    source_stem, input_root_value, output_root_value = task
    input_root = Path(input_root_value)
    output_root = Path(output_root_value)

    import simple_exr

    source_path = input_root / f"{source_stem}.exr"
    source_sha256 = _sha256(source_path)
    image = np.asarray(simple_exr.read_exr(str(source_path)))
    if source_sha256 != _sha256(source_path):
        raise RuntimeError(f"environment map changed while being read: {source_path}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            "environment map must have shape (H,W,C) with at least three channels; "
            f"got {image.shape} from {source_path}"
        )
    if not np.isfinite(image[..., :3]).all():
        raise ValueError(f"environment map contains non-finite RGB values: {source_path}")

    output_records: list[dict[str, object]] = []
    for permutation_name, indices in CHANNEL_PERMUTATIONS:
        shuffled = np.ascontiguousarray(
            image[..., :3][..., list(indices)], dtype=np.float32
        )
        output_stem = _variant_stem(source_stem, permutation_name)
        output_path = output_root / f"{output_stem}.exr"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(
            f".{output_path.stem}.{os.getpid()}.tmp.exr"
        )
        try:
            simple_exr.write_exr(str(temporary), shuffled)
            decoded = np.asarray(simple_exr.read_exr(str(temporary)))
            if decoded.shape != shuffled.shape:
                raise RuntimeError(
                    "generated EXR has invalid shape "
                    f"{decoded.shape}; expected {shuffled.shape}: {temporary}"
                )
            if decoded.dtype != np.dtype(np.float32):
                raise RuntimeError(
                    "generated EXR has invalid dtype "
                    f"{decoded.dtype}; expected float32: {temporary}"
                )
            if not np.array_equal(decoded, shuffled):
                raise RuntimeError(
                    f"generated EXR failed exact RGB verification: {temporary}"
                )
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        output_records.append(
            {
                "stem": output_stem,
                "sha256": _sha256(output_path),
                "shape": list(decoded.shape),
                "dtype": str(decoded.dtype),
            }
        )
    return current_process().name, {
        "stem": source_stem,
        "sha256": source_sha256,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "outputs": output_records,
    }


def _safe_bundle_metadata_path(value: str, *, field: str) -> Path:
    path = Path(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.as_posix() in {"", ".", ".."}
    ):
        raise ValueError(f"{field} must be a filename inside output_root")
    return path


def _portable_path_key(value: str) -> str:
    return unicodedata.normalize(
        "NFC", unicodedata.normalize("NFC", value).casefold()
    )


def _validate_portable_output_paths(
    variant_stems: Iterable[str],
    *,
    output_list_relative: Path,
    manifest_relative: Path,
) -> None:
    planned_paths = [f"{stem}.exr" for stem in variant_stems]
    planned_paths.extend(
        (output_list_relative.as_posix(), manifest_relative.as_posix())
    )
    paths_by_portable_key: dict[str, str] = {}
    for relative_path in planned_paths:
        portable_key = _portable_path_key(relative_path)
        previous = paths_by_portable_key.get(portable_key)
        if previous is not None:
            raise ValueError(
                "portable output-path collision after Unicode normalization and "
                f"case folding: {previous!r} and {relative_path!r}"
            )
        paths_by_portable_key[portable_key] = relative_path


def _aggregate_records(records: Iterable[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _path_entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _acquire_sibling_lock(output_root: Path) -> _SiblingLock:
    lock_path = output_root.with_name(f".{output_root.name}.lock")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        file_descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            "another channel-shuffle publisher is active, or a stale lock must be "
            f"reviewed and removed: {lock_path}"
        ) from error

    try:
        try:
            lock_stat = os.fstat(file_descriptor)
        except OSError as fstat_error:
            try:
                lock_stat = os.stat(file_descriptor)
            except (OSError, TypeError, ValueError) as stat_error:
                raise RuntimeError(
                    "could not verify ownership of the newly created lock; it is "
                    f"retained for safe manual review at {lock_path}; fstat error: "
                    f"{fstat_error!r}; fd-stat error: {stat_error!r}"
                ) from fstat_error
    except BaseException:
        os.close(file_descriptor)
        raise
    try:
        os.write(
            file_descriptor,
            (f"pid={os.getpid()}\noutput_root={output_root}\n").encode("utf-8"),
        )
        os.fsync(file_descriptor)
    except BaseException:
        try:
            os.close(file_descriptor)
        finally:
            _unlink_lock_if_owned(
                lock_path, device=lock_stat.st_dev, inode=lock_stat.st_ino
            )
        raise
    return _SiblingLock(
        path=lock_path,
        file_descriptor=file_descriptor,
        device=lock_stat.st_dev,
        inode=lock_stat.st_ino,
    )


def _unlink_lock_if_owned(path: Path, *, device: int, inode: int) -> None:
    try:
        current_stat = os.lstat(path)
    except FileNotFoundError:
        return
    if (current_stat.st_dev, current_stat.st_ino) == (device, inode):
        path.unlink()


def _release_sibling_lock(lock: _SiblingLock) -> None:
    os.close(lock.file_descriptor)
    _unlink_lock_if_owned(lock.path, device=lock.device, inode=lock.inode)


def _validate_staging_bundle(
    staging_root: Path,
    *,
    output_list_relative: Path,
    manifest_relative: Path,
    expected_manifest: dict[str, object],
) -> None:
    resolved_staging_root = staging_root.resolve()
    output_list_path = staging_root / output_list_relative
    manifest_path = staging_root / manifest_relative
    for metadata_path in (output_list_path, manifest_path):
        if (
            not metadata_path.is_file()
            or metadata_path.is_symlink()
            or not metadata_path.resolve().is_relative_to(resolved_staging_root)
        ):
            raise RuntimeError(
                f"staged metadata is missing or not self-contained: {metadata_path}"
            )

    actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if actual_manifest != expected_manifest:
        raise RuntimeError(
            f"staged manifest changed before publication: {manifest_path}"
        )

    output_list_bytes = output_list_path.read_bytes()
    if hashlib.sha256(output_list_bytes).hexdigest() != expected_manifest[
        "output_list_sha256"
    ]:
        raise RuntimeError(f"staged output-list hash mismatch: {output_list_path}")

    source_records = expected_manifest["sources"]
    source_inventory = [
        {
            key: source_record[key]
            for key in ("stem", "sha256", "shape", "dtype")
        }
        for source_record in source_records
    ]
    if _aggregate_records(source_inventory) != expected_manifest[
        "source_aggregate_sha256"
    ]:
        raise RuntimeError("staged source inventory aggregate mismatch")

    output_records = sorted(
        (
            output
            for source_record in source_records
            for output in source_record["outputs"]
        ),
        key=lambda record: record["stem"],
    )
    expected_stems = [record["stem"] for record in output_records]
    expected_list_bytes = "".join(f"{stem}\n" for stem in expected_stems).encode(
        "utf-8"
    )
    if output_list_bytes != expected_list_bytes:
        raise RuntimeError(
            f"staged output list has the wrong inventory: {output_list_path}"
        )
    if len(output_records) != expected_manifest["output_count"]:
        raise RuntimeError("staged output count does not match its manifest")
    if _aggregate_records(output_records) != expected_manifest[
        "output_aggregate_sha256"
    ]:
        raise RuntimeError("staged output inventory aggregate mismatch")

    import simple_exr

    for output_record in output_records:
        output_path = staging_root / f"{output_record['stem']}.exr"
        if (
            not output_path.is_file()
            or output_path.is_symlink()
            or not output_path.resolve().is_relative_to(resolved_staging_root)
        ):
            raise RuntimeError(
                f"staged EXR is missing or not self-contained: {output_path}"
            )
        if _sha256(output_path) != output_record["sha256"]:
            raise RuntimeError(f"staged EXR hash mismatch: {output_path}")
        decoded = np.asarray(simple_exr.read_exr(str(output_path)))
        if list(decoded.shape) != output_record["shape"]:
            raise RuntimeError(f"staged EXR shape mismatch: {output_path}")
        if (
            decoded.shape[2] != 3
            or output_record["dtype"] != "float32"
            or decoded.dtype != np.dtype(np.float32)
        ):
            raise RuntimeError(
                f"staged EXR must decode as three-channel float32: {output_path}"
            )


def _raise_atomic_rename_error(
    error_number: int, *, staging_root: Path, output_root: Path
) -> None:
    message = (
        f"atomic no-clobber rename failed from {staging_root} to {output_root}: "
        f"{os.strerror(error_number)}"
    )
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, message, str(output_root))
    raise OSError(error_number, message, str(staging_root))


def _atomic_rename_noreplace(staging_root: Path, output_root: Path) -> None:
    source_bytes = os.fsencode(staging_root)
    destination_bytes = os.fsencode(output_root)

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-clobber publication requires libc renameat2 on Linux"
            ) from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
    elif sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renamex_np = libc.renamex_np
        except AttributeError as error:
            raise RuntimeError(
                "atomic no-clobber publication requires renamex_np on macOS"
            ) from error
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renamex_np(source_bytes, destination_bytes, _RENAME_EXCL)
    elif os.name == "nt":
        os.rename(staging_root, output_root)
        return
    else:
        raise RuntimeError(
            "atomic no-clobber publication is unavailable on this platform; "
            "supported platforms are Linux, macOS, and Windows"
        )

    if result != 0:
        _raise_atomic_rename_error(
            ctypes.get_errno(),
            staging_root=staging_root,
            output_root=output_root,
        )


def _publish_new_bundle(staging_root: Path, output_root: Path) -> None:
    if _path_entry_exists(output_root):
        raise FileExistsError(
            "channel-shuffle output root appeared during generation; refusing to "
            f"replace it: {output_root}"
        )
    _atomic_rename_noreplace(staging_root, output_root)


def generate_channel_shuffled_env_maps(
    *,
    input_list: str | Path,
    input_root: str | Path,
    output_root: str | Path,
    output_list_name: str = "envmaps-filtered-channel-shuffle.txt",
    manifest_name: str = "channel-shuffle-manifest.json",
    num_workers: int | None = None,
) -> ChannelShuffleResult:
    """Generate six RGB permutations for every explicitly listed EXR.

    The source list is the collection index; this function never discovers
    inputs by walking ``input_root``. Outputs use the same relative stem plus
    one of ``_RGB``, ``_RBG``, ``_GRB``, ``_GBR``, ``_BRG``, or ``_BGR``.
    """

    input_list_path = Path(input_list).expanduser().resolve()
    if not input_list_path.is_file():
        raise FileNotFoundError(
            f"environment-map input list is missing: {input_list_path}"
        )
    input_root_path = Path(input_root).expanduser().resolve()
    raw_output_root = Path(output_root).expanduser()
    if raw_output_root.name in {"", ".", ".."}:
        raise ValueError("output_root must name a dedicated collection directory")
    if not raw_output_root.is_absolute():
        raw_output_root = Path.cwd() / raw_output_root
    output_root_path = raw_output_root.parent.resolve() / raw_output_root.name
    output_list_relative = _safe_bundle_metadata_path(
        output_list_name, field="output_list_name"
    )
    manifest_relative = _safe_bundle_metadata_path(
        manifest_name, field="manifest_name"
    )
    if not input_root_path.is_dir():
        raise NotADirectoryError(
            f"environment-map input root is missing: {input_root_path}"
        )
    if _path_entry_exists(output_root_path):
        raise FileExistsError(
            f"channel-shuffle output root already exists: {output_root_path}"
        )
    protected_paths = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(__file__).resolve(),
    }
    resolved_output_root = output_root_path.resolve()
    if any(
        protected_path == resolved_output_root
        or protected_path.is_relative_to(resolved_output_root)
        for protected_path in protected_paths
    ):
        raise ValueError("output_root must name a dedicated collection directory")

    input_list_bytes = input_list_path.read_bytes()
    input_list_sha256 = hashlib.sha256(input_list_bytes).hexdigest()
    source_stems = _parse_env_map_stems(
        input_list_bytes.decode("utf-8"), list_path=input_list_path
    )
    worker_count = (
        min(os.cpu_count() or 1, 32, len(source_stems))
        if num_workers is None
        else num_workers
    )
    if worker_count < 1:
        raise ValueError(f"num_workers must be at least 1, got {worker_count}")
    variant_stems = _all_variant_stems(source_stems)
    _validate_portable_output_paths(
        variant_stems,
        output_list_relative=output_list_relative,
        manifest_relative=manifest_relative,
    )
    source_paths = [input_root_path / f"{stem}.exr" for stem in source_stems]
    resolved_source_paths = [path.resolve() for path in source_paths]
    escaped_source = next(
        (
            source_path
            for source_path, resolved_source_path in zip(
                source_paths, resolved_source_paths, strict=True
            )
            if not resolved_source_path.is_relative_to(input_root_path)
        ),
        None,
    )
    if escaped_source is not None:
        raise ValueError(
            f"listed environment map resolves outside input root: {escaped_source}"
        )
    missing_sources = [path for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "listed environment map is missing: " + str(missing_sources[0])
        )
    overlapping_source = next(
        (
            source_path
            for source_path, resolved_source_path in zip(
                source_paths, resolved_source_paths, strict=True
            )
            if resolved_source_path == resolved_output_root
            or resolved_source_path.is_relative_to(resolved_output_root)
            or resolved_output_root.is_relative_to(resolved_source_path)
        ),
        None,
    )
    if overlapping_source is not None:
        raise ValueError(
            "listed environment-map source overlaps output_root: "
            f"{overlapping_source}"
        )

    output_targets = {output_root_path / f"{stem}.exr" for stem in variant_stems}
    metadata_targets = {
        output_root_path / output_list_relative,
        output_root_path / manifest_relative,
    }
    final_targets = output_targets | metadata_targets
    escaped_output = next(
        (
            path
            for path in final_targets
            if not path.resolve().is_relative_to(output_root_path)
        ),
        None,
    )
    if escaped_output is not None:
        raise ValueError(
            f"channel-shuffle output resolves outside output root: {escaped_output}"
        )
    if output_targets & metadata_targets:
        raise ValueError("output list and manifest must not replace an EXR output")
    source_targets = set(resolved_source_paths) | {input_list_path}
    if {path.resolve() for path in final_targets} & source_targets:
        raise ValueError("channel-shuffle outputs must not overlap source inputs")
    if input_root_path.is_relative_to(
        resolved_output_root
    ) or input_list_path.is_relative_to(resolved_output_root):
        raise ValueError("output_root must not contain source inputs")

    output_root_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root_path.name}.staging-",
            dir=output_root_path.parent,
        )
    )
    published = False
    sibling_lock: _SiblingLock | None = None
    try:
        sibling_lock = _acquire_sibling_lock(output_root_path)
        if _path_entry_exists(output_root_path):
            raise FileExistsError(
                f"channel-shuffle output root already exists: {output_root_path}"
            )

        tasks = [
            (source_stem, str(input_root_path), str(staging_root))
            for source_stem in source_stems
        ]
        source_records: list[dict[str, object]] = []
        if worker_count == 1:
            completed = map(_process_channel_shuffle_source, tasks)
            for worker_name, source_record in completed:
                source_records.append(source_record)
                print(
                    f"[{worker_name}] generated {source_record['stem']}",
                    flush=True,
                )
        else:
            with get_context("spawn").Pool(processes=worker_count) as pool:
                for worker_name, source_record in pool.imap(
                    _process_channel_shuffle_source, tasks, chunksize=1
                ):
                    source_records.append(source_record)
                    print(
                        f"[{worker_name}] generated {source_record['stem']}",
                        flush=True,
                    )

        if input_list_sha256 != _sha256(input_list_path):
            raise RuntimeError(f"input list changed during generation: {input_list_path}")
        for source_path, source_record in zip(
            source_paths, source_records, strict=True
        ):
            if source_record["sha256"] != _sha256(source_path):
                raise RuntimeError(
                    f"environment map changed during generation: {source_path}"
                )

        output_records = sorted(
            (
                output
                for source_record in source_records
                for output in source_record["outputs"]
            ),
            key=lambda record: record["stem"],
        )
        if [record["stem"] for record in output_records] != list(variant_stems):
            raise RuntimeError("generated EXR inventory does not match planned stems")
        source_inventory = [
            {
                key: source_record[key]
                for key in ("stem", "sha256", "shape", "dtype")
            }
            for source_record in source_records
        ]

        output_list_text = "".join(f"{stem}\n" for stem in variant_stems)
        output_list_path = staging_root / output_list_relative
        manifest_path = staging_root / manifest_relative
        _atomic_write_text(output_list_path, output_list_text)
        manifest_payload = {
            "schema_version": 1,
            "operation": "rgb_channel_permutations",
            "input_list": str(input_list_path),
            "input_list_sha256": input_list_sha256,
            "input_root": str(input_root_path),
            "output_root": str(output_root_path),
            "output_list": output_list_relative.as_posix(),
            "output_list_order": "lexicographic_relative_stem",
            "output_list_sha256": hashlib.sha256(
                output_list_text.encode("utf-8")
            ).hexdigest(),
            "source_count": len(source_stems),
            "source_aggregate_sha256": _aggregate_records(source_inventory),
            "sources": source_records,
            "output_count": len(variant_stems),
            "output_aggregate_sha256": _aggregate_records(output_records),
            "permutations": [
                {"name": name, "indices": list(indices)}
                for name, indices in CHANNEL_PERMUTATIONS
            ],
            "output_dtype": "float32",
            "dependencies": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "simple-exr": _distribution_version("simple-exr"),
                "OpenEXR": _distribution_version("OpenEXR"),
            },
        }
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        )
        _validate_staging_bundle(
            staging_root,
            output_list_relative=output_list_relative,
            manifest_relative=manifest_relative,
            expected_manifest=manifest_payload,
        )
        _publish_new_bundle(staging_root, output_root_path)
        published = True
    finally:
        try:
            if not published and staging_root.exists():
                shutil.rmtree(staging_root)
        finally:
            if sibling_lock is not None:
                _release_sibling_lock(sibling_lock)

    return ChannelShuffleResult(
        output_root=output_root_path,
        output_list=output_root_path / output_list_relative,
        manifest=output_root_path / manifest_relative,
        source_count=len(source_stems),
        output_count=len(variant_stems),
    )


__all__ = [
    "CHANNEL_PERMUTATIONS",
    "ChannelShuffleResult",
    "channel_shuffle_variants",
    "generate_channel_shuffled_env_maps",
    "load_env_map_stems",
]
