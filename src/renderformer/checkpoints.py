"""Checkpoint resolution and strict shared-model loading."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from renderformer.config import InferenceConfig, load_inference_config


DEFAULT_V2_REPOSITORY = "RenderFormer/renderformer-v2"
DEFAULT_V2_TRANSFORMER_SUBFOLDERS = {
    512: "transformer_512",
    2048: "transformer_2048",
}
DEFAULT_V2_MATERIAL_AUTOENCODER_SUBFOLDER = "material_autoencoder"
DEFAULT_V2_MAPPER_SUBFOLDERS = {
    "diffuse_specular": "diffspec_mapper",
    "metallic_roughness": "metallic_mapper",
    "metallic_roughness_transmission": "metallic_transmission_mapper",
}


def default_v2_transformer_subfolder(resolution: int) -> str:
    """Return the native V2 transformer component for an output resolution."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    native_resolution = 2048 if resolution >= 2048 else 512
    return DEFAULT_V2_TRANSFORMER_SUBFOLDERS[native_resolution]


def select_v2_transformer_subfolder(
    source: str | Path,
    *,
    resolution: int,
    subfolder: str | None = None,
) -> str | None:
    """Select a bundled transformer while leaving standalone checkpoints flat."""

    if subfolder is not None:
        return normalize_checkpoint_subfolder(subfolder)
    local_path = Path(source).expanduser()
    is_bundle = str(source) == DEFAULT_V2_REPOSITORY or (
        local_path.is_dir() and (local_path / "model_index.json").is_file()
    )
    if not is_bundle:
        return None
    return default_v2_transformer_subfolder(resolution)


def normalize_checkpoint_subfolder(subfolder: str | None) -> str | None:
    """Validate a component path so it cannot escape a checkpoint bundle."""

    if subfolder is None:
        return None
    raw = str(subfolder).replace("\\", "/")
    parts = raw.split("/")
    if (
        not raw
        or raw.startswith("/")
        or (parts and parts[0].endswith(":"))
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"invalid checkpoint subfolder: {subfolder!r}")
    return "/".join(parts)


def resolve_checkpoint(
    source: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    token: str | bool | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> Path:
    """Resolve a local checkpoint directory or Hugging Face snapshot."""

    subfolder = normalize_checkpoint_subfolder(subfolder)
    local_path = Path(source).expanduser()
    if local_path.exists():
        return local_path.parent if local_path.is_file() else local_path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "loading a Hugging Face model ID requires the inference extra"
        ) from exc

    allow_patterns = [
        "model_index.json",
        "config.json",
        "config.yaml",
        "config.yml",
        "model_config.json",
        "model.safetensors",
    ]
    if subfolder is not None:
        allow_patterns.extend(
            f"{subfolder}/{filename}"
            for filename in (
                "config.json",
                "config.yaml",
                "config.yml",
                "model_config.json",
                "model.safetensors",
            )
        )
    snapshot = snapshot_download(
        repo_id=str(source),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=allow_patterns,
    )
    return Path(snapshot)


def checkpoint_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_shared_model(
    checkpoint_dir: str | Path,
    *,
    device: Any = "cpu",
    apply_fused_kernels: bool = True,
    subfolder: str | None = None,
):
    """Instantiate the one training/inference model and strictly load weights."""

    from renderformer.modeling import RenderFormerModel

    checkpoint_dir = Path(checkpoint_dir)
    subfolder = normalize_checkpoint_subfolder(subfolder)
    component_dir = checkpoint_dir if subfolder is None else checkpoint_dir / subfolder
    config = load_inference_config(component_dir)
    model = RenderFormerModel.from_pretrained(
        checkpoint_dir,
        config=config,
        subfolder=subfolder,
        strict=True,
        device="cpu",
    )

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if apply_fused_kernels and device_type == "cuda":
        from renderformer.ops.apply_kernels import apply_kernels

        model = apply_kernels(model, verbose=False)
    model.eval().requires_grad_(False).to(device)
    return model, config


def checkpoint_metadata(
    checkpoint_dir: str | Path,
    *,
    subfolder: str | None = None,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    subfolder = normalize_checkpoint_subfolder(subfolder)
    component_dir = checkpoint_dir if subfolder is None else checkpoint_dir / subfolder
    config = load_inference_config(component_dir)
    weights_path = component_dir / "model.safetensors"
    return {
        "version": config.runtime.version.value,
        "subfolder": subfolder,
        "config_file": config.source_file.name,
        "config_sha256": checkpoint_sha256(config.source_file),
        "weights_file": weights_path.name,
        "weights_size": weights_path.stat().st_size,
        "weights_sha256": checkpoint_sha256(weights_path),
    }
