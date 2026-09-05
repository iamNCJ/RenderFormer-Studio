#!/usr/bin/env python3
"""Build deterministic RenderFormer Blender extension archives with PyPI bpy."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile


ADDON_ROOT = Path(__file__).resolve().parent
DEPENDENCY_LOCK = ADDON_ROOT / "release_dependencies.toml"
PACKAGE_NAMES = ("renderformer_v1_tools", "renderformer_v2_tools")
REQUIRED_BPY_VERSION = (4, 5, 10)
REQUIRED_BPY_DISTRIBUTION = "4.5.10"
DEFAULT_SOURCE_DATE_EPOCH = 315619200  # 1980-01-02 UTC, safely inside ZIP's range.
_PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[A-Za-z0-9_.+!-]+)$")


def _normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_dependency_lock(path: Path = DEPENDENCY_LOCK) -> tuple[str, dict[str, tuple[str, ...]]]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported dependency-lock schema in {path}")
    python_version = payload.get("python_version")
    if python_version != "3.11":
        raise ValueError(f"release dependency lock must target Python 3.11, got {python_version!r}")
    package_table = payload.get("packages", {})
    if set(package_table) != set(PACKAGE_NAMES):
        raise ValueError(
            f"dependency lock packages must be {PACKAGE_NAMES}, got {tuple(package_table)}"
        )

    locked: dict[str, tuple[str, ...]] = {}
    for package_name in PACKAGE_NAMES:
        requirements = tuple(package_table[package_name].get("requirements", ()))
        if not requirements or any(_PIN.fullmatch(item) is None for item in requirements):
            raise ValueError(f"{package_name} dependencies must use exact name==version pins")
        normalized = [
            _normalized_distribution(_PIN.fullmatch(item).group("name"))  # type: ignore[union-attr]
            for item in requirements
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{package_name} dependency lock contains duplicate distributions")
        locked[package_name] = requirements
    return python_version, locked


def local_blender_platform(system: str | None = None, machine: str | None = None) -> str:
    system_name = (system or platform.system()).lower()
    machine_name = (machine or platform.machine()).lower()
    system_name = {"darwin": "macos"}.get(system_name, system_name)
    machine_name = {
        "amd64": "x64",
        "x86_64": "x64",
        "aarch64": "arm64",
        "aarch32": "arm32",
    }.get(machine_name, machine_name)
    result = f"{system_name}-{machine_name}"
    supported = {
        "linux-x64",
        "macos-x64",
        "macos-arm64",
        "windows-x64",
        "windows-arm64",
    }
    if result not in supported:
        raise ValueError(f"unsupported Blender extension build platform: {result}")
    return result


def _wheel_identity(path: Path) -> tuple[str, str]:
    if path.suffix != ".whl":
        raise ValueError(f"not a wheel file: {path}")
    parts = path.name[:-4].split("-")
    if len(parts) < 5:
        raise ValueError(f"invalid wheel filename: {path.name}")
    return _normalized_distribution(parts[0]), parts[1]


def select_locked_wheels(
    wheel_dir: Path, requirements: tuple[str, ...]
) -> tuple[Path, ...]:
    wheels = sorted(wheel_dir.glob("*.whl"), key=lambda path: path.name.lower())
    selected: list[Path] = []
    for requirement in requirements:
        match = _PIN.fullmatch(requirement)
        if match is None:
            raise ValueError(f"dependency is not exactly pinned: {requirement}")
        expected = (
            _normalized_distribution(match.group("name")),
            match.group("version").replace("_", "-"),
        )
        candidates = [
            wheel
            for wheel in wheels
            if (_wheel_identity(wheel)[0], _wheel_identity(wheel)[1].replace("_", "-"))
            == expected
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected exactly one compatible wheel for {requirement}, found "
                f"{[candidate.name for candidate in candidates]}"
            )
        selected.append(candidates[0])
    unexpected = sorted(set(wheels) - set(selected), key=lambda path: path.name.lower())
    if unexpected:
        raise ValueError(
            f"wheel download produced unexpected files: {[p.name for p in unexpected]}"
        )
    return tuple(selected)


def download_locked_wheels(
    requirements: tuple[str, ...], destination: Path, wheelhouse: Path | None = None
) -> tuple[Path, ...]:
    destination.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--no-deps",
        "--dest",
        str(destination),
    ]
    if wheelhouse is not None:
        command.extend(("--no-index", "--find-links", str(wheelhouse.resolve())))
    command.extend(requirements)
    subprocess.run(command, check=True)
    return select_locked_wheels(destination, requirements)


def _release_manifest(source: str, wheel_names: tuple[str, ...], platform_tag: str) -> str:
    if re.search(r"(?m)^\s*(?:platforms|wheels)\s*=", source):
        raise ValueError("source manifest must not contain release-only platforms or wheels")
    table = re.search(r"(?m)^\[", source)
    if table is None:
        raise ValueError("manifest contains no TOML table insertion point")
    wheel_lines = "\n".join(f'  "./wheels/{name}",' for name in wheel_names)
    release_fields = (
        f'platforms = ["{platform_tag}"]\n'
        "wheels = [\n"
        f"{wheel_lines}\n"
        "]\n\n"
    )
    return source[: table.start()] + release_fields + source[table.start() :]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_mtimes(root: Path, epoch: int) -> None:
    if epoch < DEFAULT_SOURCE_DATE_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH cannot predate 1980-01-01 for ZIP output")
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        os.utime(path, (epoch, epoch), follow_symlinks=False)
    os.utime(root, (epoch, epoch), follow_symlinks=False)


def stage_release_package(
    source_dir: Path,
    staging_dir: Path,
    wheels: tuple[Path, ...],
    requirements: tuple[str, ...],
    platform_tag: str,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> Path:
    _manifest_from_directory(source_dir)
    selected = select_locked_wheels(wheels[0].parent, requirements)
    if tuple(wheels) != selected:
        raise ValueError("wheel paths must be the complete sorted locked wheel set")
    if staging_dir.exists():
        raise FileExistsError(f"release staging directory already exists: {staging_dir}")

    shutil.copytree(
        source_dir,
        staging_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.whl", "*.zip"),
    )
    staged_wheel_dir = staging_dir / "wheels"
    staged_wheel_dir.mkdir()
    staged_wheels = []
    for wheel in wheels:
        target = staged_wheel_dir / wheel.name
        shutil.copy2(wheel, target)
        staged_wheels.append(target)

    manifest_path = staging_dir / "blender_manifest.toml"
    manifest_path.write_text(
        _release_manifest(
            manifest_path.read_text(encoding="utf-8"),
            tuple(path.name for path in staged_wheels),
            platform_tag,
        ),
        encoding="utf-8",
    )
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = [f"./wheels/{path.name}" for path in staged_wheels]
    if manifest.get("platforms") != [platform_tag] or manifest.get("wheels") != expected_paths:
        raise RuntimeError("release manifest injection did not round-trip")

    inventory = {
        "schema_version": 1,
        "python_version": "3.11",
        "platform": platform_tag,
        "requirements": list(requirements),
        "wheels": [
            {"filename": path.name, "sha256": _sha256(path)} for path in staged_wheels
        ],
    }
    (staging_dir / "WHEEL_INVENTORY.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _normalize_mtimes(staging_dir, source_date_epoch)
    return staging_dir


def _bpy_version() -> tuple[int, int, int]:
    bpy = importlib.import_module("bpy")
    version = tuple(bpy.app.version)
    if len(version) != 3 or any(not isinstance(part, int) for part in version):
        raise RuntimeError(f"bpy.app.version must be a three-integer tuple, got {version!r}")
    return version  # type: ignore[return-value]


def _bpy_distribution_version() -> str:
    return importlib.metadata.version("bpy")


def require_build_environment(python_version: str) -> None:
    running_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if running_python != python_version:
        raise RuntimeError(
            f"extension builds require Python {python_version}; running {running_python}"
        )
    distribution_version = _bpy_distribution_version()
    if distribution_version != REQUIRED_BPY_DISTRIBUTION:
        raise RuntimeError(
            "extension builds require bpy==4.5.10 from PyPI exactly; installed "
            f"bpy distribution is {distribution_version}"
        )
    bpy_version = _bpy_version()
    if bpy_version != REQUIRED_BPY_VERSION:
        raise RuntimeError(
            "extension builds require PyPI bpy==4.5.10 exactly; bpy.app.version is "
            + ".".join(str(part) for part in bpy_version)
        )


def _manifest_from_directory(package_dir: Path) -> dict[str, object]:
    manifest = tomllib.loads(
        (package_dir / "blender_manifest.toml").read_text(encoding="utf-8")
    )
    package_name = manifest.get("id")
    if package_name not in PACKAGE_NAMES or package_dir.name != package_name:
        raise ValueError(
            f"manifest/package ID mismatch for {package_dir}: {package_name!r}"
        )
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError(f"unsupported Blender manifest schema for {package_name!r}")
    if manifest.get("blender_version_min") != "4.5.10":
        raise ValueError(f"{package_name} must require Blender 4.5.10 or newer")
    if manifest.get("blender_version_max") != "4.5.11":
        raise ValueError(f"{package_name} must support only Blender 4.5.10")
    return manifest


def stage_source_package(
    source_dir: Path,
    staging_dir: Path,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> Path:
    _manifest_from_directory(source_dir)
    if staging_dir.exists():
        raise FileExistsError(f"source staging directory already exists: {staging_dir}")
    shutil.copytree(
        source_dir,
        staging_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.whl", "*.zip"),
    )
    _normalize_mtimes(staging_dir, source_date_epoch)
    return staging_dir


def _zip_datetime(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    if source_date_epoch < DEFAULT_SOURCE_DATE_EPOCH:
        raise ValueError("SOURCE_DATE_EPOCH cannot predate 1980-01-01 for ZIP output")
    return time.gmtime(source_date_epoch)[:6]


def write_deterministic_archive(
    package_dir: Path, archive: Path, source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH
) -> Path:
    archive.parent.mkdir(parents=True, exist_ok=True)
    timestamp = _zip_datetime(source_date_epoch)
    files = sorted(
        (path for path in package_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(package_dir).as_posix(),
    )
    if not files:
        raise ValueError(f"extension package is empty: {package_dir}")
    # Wheels are already ZIP-compressed. Storing every member also avoids making
    # archive bytes depend on the host's zlib implementation.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as handle:
        for path in files:
            if path.is_symlink():
                raise ValueError(f"extension archives cannot contain symlinks: {path}")
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=timestamp)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())
    return archive


def _read_archive_manifest(
    handle: zipfile.ZipFile, archive: Path
) -> tuple[dict[str, object], set[str]]:
    archive_names = handle.namelist()
    names = set(archive_names)
    if len(names) != len(archive_names):
        raise RuntimeError(f"extension archive contains duplicate paths: {archive}")
    if archive_names != sorted(archive_names):
        raise RuntimeError(f"extension archive paths are not deterministic: {archive}")
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "__pycache__" in path.parts:
            raise RuntimeError(f"extension archive contains an unsafe path {name!r}: {archive}")
        if name.endswith((".pyc", ".zip")):
            raise RuntimeError(f"extension archive contains a generated file {name!r}: {archive}")
    required = {"__init__.py", "blender_manifest.toml", "LICENSE", "THIRD_PARTY_NOTICES.md"}
    missing = required - names
    if missing:
        raise RuntimeError(f"extension archive is missing {sorted(missing)}: {archive}")
    manifest = tomllib.loads(handle.read("blender_manifest.toml").decode("utf-8"))
    package_name = manifest.get("id")
    if package_name not in PACKAGE_NAMES:
        raise RuntimeError(f"extension archive has an unknown package ID: {package_name!r}")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("blender_version_min") != "4.5.10"
        or manifest.get("blender_version_max") != "4.5.11"
    ):
        raise RuntimeError(f"extension archive manifest is not release-pinned: {archive}")
    return manifest, names


def verify_source_archive(archive: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        manifest, names = _read_archive_manifest(handle, archive)
        if "platforms" in manifest or "wheels" in manifest:
            raise RuntimeError(f"source-only archive has release-only manifest fields: {archive}")
        if "WHEEL_INVENTORY.json" in names or any(
            name.startswith("wheels/") or name.endswith(".whl") for name in names
        ):
            raise RuntimeError(f"source-only archive contains bundled wheels: {archive}")


def verify_release_archive(archive: Path, platform_tag: str) -> None:
    with zipfile.ZipFile(archive) as handle:
        manifest, names = _read_archive_manifest(handle, archive)
        wheel_paths = manifest.get("wheels", [])
        package_name = manifest.get("id")
        python_version, dependency_lock = load_dependency_lock()
        if (
            manifest.get("platforms") != [platform_tag]
            or not isinstance(wheel_paths, list)
            or not wheel_paths
            or not all(
                isinstance(path, str)
                and re.fullmatch(r"\./wheels/[A-Za-z0-9_.+-]+\.whl", path)
                for path in wheel_paths
            )
            or len(wheel_paths) != len(set(wheel_paths))
        ):
            raise RuntimeError(f"release archive metadata is incomplete: {archive}")
        normalized_wheel_paths = [path.removeprefix("./") for path in wheel_paths]
        archived_wheels = {
            name for name in names if name.startswith("wheels/") and name.endswith(".whl")
        }
        if archived_wheels != set(normalized_wheel_paths):
            raise RuntimeError(
                "release archive wheel contents do not match its manifest: "
                f"{archive}"
            )
        if "WHEEL_INVENTORY.json" not in names:
            raise RuntimeError(f"release archive is missing WHEEL_INVENTORY.json: {archive}")
        if "THIRD_PARTY_NOTICES.md" not in names:
            raise RuntimeError(f"release archive is missing third-party notices: {archive}")
        inventory = json.loads(handle.read("WHEEL_INVENTORY.json").decode("utf-8"))
        expected_requirements = list(dependency_lock[package_name])
        if (
            inventory.get("schema_version") != 1
            or inventory.get("python_version") != python_version
            or inventory.get("platform") != platform_tag
            or inventory.get("requirements") != expected_requirements
            or not isinstance(inventory.get("wheels"), list)
        ):
            raise RuntimeError(f"release archive wheel inventory is inconsistent: {archive}")
        inventory_wheels = inventory["wheels"]
        expected_filenames = [Path(path).name for path in normalized_wheel_paths]
        inventory_filenames = [
            item.get("filename") if isinstance(item, dict) else None
            for item in inventory_wheels
        ]
        if inventory_filenames != expected_filenames:
            raise RuntimeError(
                f"release archive wheel inventory does not match its manifest: {archive}"
            )
        for wheel_path, item in zip(
            normalized_wheel_paths, inventory_wheels, strict=True
        ):
            expected_sha256 = item.get("sha256")
            actual_sha256 = hashlib.sha256(handle.read(wheel_path)).hexdigest()
            if (
                not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
                or actual_sha256 != expected_sha256
            ):
                raise RuntimeError(
                    f"release archive wheel hash mismatch for {wheel_path}: {archive}"
                )


def build_release_archive(
    staging_dir: Path,
    output_dir: Path,
    platform_tag: str,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> Path:
    manifest = _manifest_from_directory(staging_dir)
    archive = output_dir / f"{manifest['id']}-{manifest['version']}-{platform_tag}.zip"
    write_deterministic_archive(staging_dir, archive, source_date_epoch)
    verify_release_archive(archive, platform_tag)
    return archive


def build_source_archive(
    staging_dir: Path,
    output_dir: Path,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
) -> Path:
    manifest = _manifest_from_directory(staging_dir)
    archive = output_dir / f"{manifest['id']}-source-only.zip"
    write_deterministic_archive(staging_dir, archive, source_date_epoch)
    verify_source_archive(archive)
    return archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source-only", "release"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Offline wheel directory; omit to download the exact locked versions",
    )
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_SOURCE_DATE_EPOCH)),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    python_version, dependency_lock = load_dependency_lock()
    require_build_environment(python_version)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="renderformer-blender-build-") as temporary:
        temporary_root = Path(temporary)
        for package_name in PACKAGE_NAMES:
            staging_dir = temporary_root / "staging" / package_name
            if args.mode == "source-only":
                stage_source_package(
                    ADDON_ROOT / package_name,
                    staging_dir,
                    args.source_date_epoch,
                )
                archive = build_source_archive(
                    staging_dir,
                    args.output_dir,
                    args.source_date_epoch,
                )
                print(f"built {archive} sha256={_sha256(archive)}", flush=True)
                continue

            platform_tag = local_blender_platform()
            requirements = dependency_lock[package_name]
            wheel_dir = temporary_root / "downloads" / package_name
            wheels = download_locked_wheels(requirements, wheel_dir, args.wheelhouse)
            stage_release_package(
                ADDON_ROOT / package_name,
                staging_dir,
                wheels,
                requirements,
                platform_tag,
                args.source_date_epoch,
            )
            archive = build_release_archive(
                staging_dir,
                args.output_dir,
                platform_tag,
                args.source_date_epoch,
            )
            print(f"built {archive} sha256={_sha256(archive)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
