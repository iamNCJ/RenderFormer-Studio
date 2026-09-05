"""Validated H5 inputs shared by V1 and V2 inference."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import torch

from renderformer.config import CheckpointVersion


@dataclasses.dataclass
class SceneData:
    path: Path
    triangles: torch.Tensor
    texture: torch.Tensor
    vn: torch.Tensor
    c2w: torch.Tensor
    fov: torch.Tensor
    mvp: torch.Tensor
    image: torch.Tensor | None = None
    light_strength: torch.Tensor | None = None
    env_map: torch.Tensor | None = None
    volume: dict[str, torch.Tensor] | None = None
    input_metadata: dict[str, object] = dataclasses.field(default_factory=dict)

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def num_views(self) -> int:
        return int(self.c2w.shape[0])


def _require_keys(h5_file: h5py.File, keys: set[str], path: Path) -> None:
    missing = sorted(keys - set(h5_file.keys()))
    if missing:
        raise ValueError(f"{path} is missing required H5 keys: {missing}")


def _require_float32(h5_file: h5py.File, keys: set[str], path: Path) -> None:
    wrong = {
        key: str(h5_file[key].dtype)
        for key in sorted(keys)
        if h5_file[key].dtype != np.dtype(np.float32)
    }
    if wrong:
        raise ValueError(f"RF1 arrays must be float32 in {path}; got {wrong}")


_RF1_DTYPE_KEYS = ("triangles", "texture", "vn", "c2w", "fov")
_RF1_STRICT_DTYPE_SIGNATURE = (
    "float32",
    "float32",
    "float32",
    "float32",
    "float32",
)
_RF1_LEGACY_VIDEO_DTYPE_SIGNATURES = {
    ("float64", "float16", "float64", "float32", "float32"),
    ("float32", "float16", "float32", "float32", "float32"),
}


def _rf1_source_dtypes(h5_file: h5py.File) -> dict[str, str]:
    return {key: str(h5_file[key].dtype) for key in _RF1_DTYPE_KEYS}


def _validate_rf1_dtype_signature(
    h5_file: h5py.File,
    path: Path,
    *,
    allow_legacy_rf1_dtypes: bool,
) -> tuple[dict[str, str], list[str], str]:
    source_dtypes = _rf1_source_dtypes(h5_file)
    signature = tuple(source_dtypes[key] for key in _RF1_DTYPE_KEYS)
    if signature == _RF1_STRICT_DTYPE_SIGNATURE:
        return source_dtypes, [], "strict_rf1_contract"
    if not allow_legacy_rf1_dtypes:
        _require_float32(h5_file, set(_RF1_DTYPE_KEYS), path)
        raise AssertionError("_require_float32 must reject a non-float32 signature")
    if signature not in _RF1_LEGACY_VIDEO_DTYPE_SIGNATURES:
        rendered_signature = {
            key: source_dtypes[key] for key in _RF1_DTYPE_KEYS
        }
        raise ValueError(
            "--allow-legacy-rf1-dtypes accepts only the two dtype signatures "
            "published in renderformer/renderformer-video-data; "
            f"got {rendered_signature} in {path}"
        )
    normalized_keys = [
        key for key in _RF1_DTYPE_KEYS if source_dtypes[key] != "float32"
    ]
    return source_dtypes, normalized_keys, "historical_video_bundle"


def _load_v1(
    h5_file: h5py.File,
    path: Path,
    *,
    allow_legacy_rf1_dtypes: bool,
) -> SceneData:
    required = set(_RF1_DTYPE_KEYS)
    _require_keys(h5_file, required, path)
    source_dtypes, normalized_keys, dtype_profile = _validate_rf1_dtype_signature(
        h5_file,
        path,
        allow_legacy_rf1_dtypes=allow_legacy_rf1_dtypes,
    )

    triangles = np.asarray(h5_file["triangles"], dtype=np.float32)
    texture = np.asarray(h5_file["texture"], dtype=np.float32)
    vn = np.asarray(h5_file["vn"], dtype=np.float32)
    c2w = np.asarray(h5_file["c2w"], dtype=np.float32)
    fov = np.asarray(h5_file["fov"], dtype=np.float32)
    num_triangles = triangles.shape[0]

    expected_shapes = {
        "triangles": (num_triangles, 3, 3),
        "texture": (num_triangles, 13, 32, 32),
        "vn": (num_triangles, 3, 3),
    }
    actual_shapes = {
        "triangles": triangles.shape,
        "texture": texture.shape,
        "vn": vn.shape,
    }
    wrong_shapes = {
        key: (actual_shapes[key], expected)
        for key, expected in expected_shapes.items()
        if actual_shapes[key] != expected
    }
    if wrong_shapes:
        raise ValueError(f"invalid RF1 geometry shapes in {path}: {wrong_shapes}")
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must have shape (views, 4, 4) in {path}; got {c2w.shape}")
    if fov.shape != (c2w.shape[0],):
        raise ValueError(f"fov must have shape ({c2w.shape[0]},) in {path}; got {fov.shape}")
    if num_triangles == 0 or c2w.shape[0] == 0:
        raise ValueError(f"RF1 input must contain triangles and views: {path}")

    mvp = np.repeat(np.eye(4, dtype=np.float32)[None], c2w.shape[0], axis=0)
    return SceneData(
        path=path,
        triangles=torch.from_numpy(triangles),
        texture=torch.from_numpy(texture),
        vn=torch.from_numpy(vn),
        c2w=torch.from_numpy(c2w),
        fov=torch.from_numpy(fov),
        mvp=torch.from_numpy(mvp),
        input_metadata={
            "texture": "rf1_raw",
            "light_strength": "texture",
            "environment": "unsupported",
            "rf1_dtype_normalization": {
                "profile": dtype_profile,
                "applied": bool(normalized_keys),
                "source_dtypes": source_dtypes,
                "normalized_keys": normalized_keys,
                "target_dtype": "float32",
            },
        },
    )


def _optional_array(h5_file: h5py.File, key: str, dtype=np.float32):
    if key not in h5_file or h5_file[key].size == 0:
        return None
    return np.asarray(h5_file[key], dtype=dtype)


def _load_v2(h5_file: h5py.File, path: Path, image_key: str) -> SceneData:
    required = {"triangles", "texture", "c2w", "fov"}
    _require_keys(h5_file, required, path)

    triangles = np.asarray(h5_file["triangles"], dtype=np.float32)
    texture = np.asarray(h5_file["texture"], dtype=np.float32)
    vn = (
        np.asarray(h5_file["vn"], dtype=np.float32)
        if "vn" in h5_file
        else np.zeros_like(triangles)
    )
    c2w = np.asarray(h5_file["c2w"], dtype=np.float32)
    fov = np.asarray(h5_file["fov"], dtype=np.float32)
    if fov.ndim == 2 and fov.shape[-1] == 1:
        fov = fov[:, 0]
    num_triangles = int(triangles.shape[0])
    if triangles.shape != (num_triangles, 3, 3):
        raise ValueError(f"triangles must be (N, 3, 3) in {path}; got {triangles.shape}")
    if vn.shape != triangles.shape:
        raise ValueError(f"vn must match triangles in {path}; got {vn.shape}")
    if texture.ndim not in (2, 4) or texture.shape[0] != num_triangles:
        raise ValueError(f"texture must be (N,C) or (N,C,H,W) in {path}; got {texture.shape}")
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (views,4,4) in {path}; got {c2w.shape}")
    if fov.shape != (c2w.shape[0],):
        raise ValueError(f"fov must have one value per view in {path}; got {fov.shape}")
    if num_triangles == 0 or c2w.shape[0] == 0:
        raise ValueError(f"V2 input must contain triangles and views: {path}")

    if "mvp" in h5_file:
        mvp = np.asarray(h5_file["mvp"], dtype=np.float32)
        if mvp.ndim == 2:
            mvp = mvp[None]
    else:
        mvp = np.repeat(np.eye(4, dtype=np.float32)[None], c2w.shape[0], axis=0)
    if mvp.shape[0] != c2w.shape[0] or mvp.shape[1:] != (4, 4):
        raise ValueError(f"mvp must be (views,4,4) in {path}; got {mvp.shape}")

    image_array = _optional_array(h5_file, image_key)
    if image_array is not None and image_array.ndim == 3:
        image_array = image_array[None]
    if image_array is not None and image_array.shape[0] != c2w.shape[0]:
        raise ValueError(f"{image_key} must have one image per view in {path}")

    light_array = _optional_array(h5_file, "light_strength")
    light_source = "h5"
    if light_array is None:
        if texture.ndim == 4 and texture.shape[1] in (15, 16):
            light_array = np.clip(texture[:, 12:15, 0, 0], 0.0, np.inf)
            light_source = "texture"
        else:
            light_array = np.zeros((num_triangles, 3), dtype=np.float32)
            light_source = "missing"
    if light_array.shape != (num_triangles, 3):
        raise ValueError(
            f"light_strength must have shape ({num_triangles},3) in {path}; "
            f"got {light_array.shape}"
        )

    env_array = _optional_array(h5_file, "env_map")
    environment_source = "missing"
    if env_array is not None:
        if env_array.ndim != 3:
            raise ValueError(f"env_map must be CHW or HWC in {path}; got {env_array.shape}")
        if env_array.shape[-1] in (3, 4):
            from renderformer.data.envmap import prepare_raw_env_map

            rotation = _optional_array(h5_file, "env_map_rotation")
            if rotation is None:
                rotation = np.zeros((3,), dtype=np.float32)
            strength_array = _optional_array(h5_file, "env_map_strength")
            strength = (
                1.0
                if strength_array is None
                else float(np.asarray(strength_array).reshape(-1)[0])
            )
            env_array = prepare_raw_env_map(
                torch.from_numpy(env_array),
                torch.from_numpy(np.asarray(rotation, dtype=np.float32)),
                strength,
            ).numpy()
            environment_source = "raw_transformed"
        elif env_array.shape[0] in (3, 4):
            env_array = env_array[:3]
            environment_source = "processed"
        else:
            raise ValueError(f"env_map must contain 3 color channels in {path}")

    volume = None
    if "volume_data" in h5_file and h5_file["volume_data"].size:
        raise ValueError(
            f"{path} contains raw volume_data; run renderformer data convert first"
        )
    if "volume_density" in h5_file and h5_file["volume_density"].size:
        volume_keys = {
            "density": "volume_density",
            "position": "volume_position",
            "rotation": "volume_rotation",
            "scale": "volume_scale",
            "scattering": "volume_scattering_scale",
            "absorption": "volume_absorption_scale",
        }
        _require_keys(h5_file, set(volume_keys.values()), path)
        volume = {
            name: torch.from_numpy(np.asarray(h5_file[key], dtype=np.float32))
            for name, key in volume_keys.items()
        }
        count = volume["density"].shape[0]
        if volume["density"].shape != (count, 64):
            raise ValueError(f"volume_density must be (N,64) in {path}")
        if any(value.shape[0] != count for value in volume.values()):
            raise ValueError(f"all volume arrays must have the same length in {path}")

    return SceneData(
        path=path,
        triangles=torch.from_numpy(triangles),
        texture=torch.from_numpy(texture),
        vn=torch.from_numpy(vn),
        c2w=torch.from_numpy(c2w),
        fov=torch.from_numpy(fov),
        mvp=torch.from_numpy(mvp),
        image=torch.from_numpy(image_array) if image_array is not None else None,
        light_strength=torch.from_numpy(light_array),
        env_map=torch.from_numpy(env_array) if env_array is not None else None,
        volume=volume,
        input_metadata={
            "texture": (
                "raw"
                if texture.ndim == 4 and 12 <= texture.shape[1] <= 16
                else "preencoded"
            ),
            "light_strength": light_source,
            "environment": environment_source,
        },
    )


def load_h5_scene(
    path: str | Path,
    version: CheckpointVersion,
    *,
    image_key: str = "img",
    allow_legacy_rf1_dtypes: bool = False,
) -> SceneData:
    if not isinstance(allow_legacy_rf1_dtypes, bool):
        raise TypeError("allow_legacy_rf1_dtypes must be a bool")
    path = Path(path)
    with h5py.File(path, "r") as h5_file:
        if version is CheckpointVersion.V1:
            return _load_v1(
                h5_file,
                path,
                allow_legacy_rf1_dtypes=allow_legacy_rf1_dtypes,
            )
        if allow_legacy_rf1_dtypes:
            raise ValueError(
                "allow_legacy_rf1_dtypes is only valid for RF1/V1 inputs"
            )
        return _load_v2(h5_file, path, image_key)


def collect_h5_inputs(path: str | Path) -> list[Path]:
    """Resolve one H5, a newline manifest, or one non-recursive directory."""

    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".h5":
        return [path]
    if path.is_file() and path.suffix.lower() in {".txt", ".manifest"}:
        entries = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            item = Path(line)
            entries.append(item if item.is_absolute() else path.parent / item)
        missing = [entry for entry in entries if not entry.is_file()]
        if missing:
            raise FileNotFoundError(f"manifest contains missing inputs: {missing[:5]}")
        return entries
    if path.is_dir():
        entries = sorted(path.glob("*.h5"))
        if entries:
            return entries
        raise FileNotFoundError(f"no .h5 files found directly under {path}")
    raise FileNotFoundError(f"input must be an H5, manifest, or directory: {path}")
