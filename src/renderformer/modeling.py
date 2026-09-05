"""Public model component shared by training and inference.

The pipeline classes assemble preprocessing and auxiliary encoders around this
component.  Training code loads :class:`RenderFormerModel` directly, so it does
not need to construct an inference pipeline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch

from renderformer.config import InferenceConfig, load_inference_config
from renderformer.models.config import RenderTransformerConfig
from renderformer.models.triangle_radiosity_transformer import (
    TriangleRadiosityTransformer,
)


def _coerce_model_config(
    config: RenderTransformerConfig | InferenceConfig | dict[str, Any] | None,
    checkpoint_dir: Path,
    subfolder: str | None,
) -> RenderTransformerConfig:
    if config is None:
        component_dir = checkpoint_dir if subfolder is None else checkpoint_dir / subfolder
        component_config = component_dir / "model_config.json"
        if component_config.is_file():
            with component_config.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise TypeError(f"model component config must be an object: {component_config}")
            if payload.get("format") != "renderformer-model" or payload.get("format_version") != 1:
                raise ValueError(f"unsupported model component config: {component_config}")
            model_values = payload.get("model_config")
            if not isinstance(model_values, dict):
                raise ValueError(
                    f"model component config is missing model_config: {component_config}"
                )
            return RenderTransformerConfig(**model_values)
        config_dir = (
            component_dir
            if any(
                (component_dir / name).is_file()
                for name in ("config.json", "config.yaml", "config.yml")
            )
            else checkpoint_dir
        )
        return load_inference_config(config_dir).model_config
    if isinstance(config, InferenceConfig):
        return config.model_config
    if isinstance(config, RenderTransformerConfig):
        return config
    if isinstance(config, dict):
        return RenderTransformerConfig(**config)
    raise TypeError(
        "config must be RenderTransformerConfig, InferenceConfig, a mapping, or None"
    )


def _resolve_local_weight_file(
    source: str | Path,
    checkpoint_dir: Path,
    subfolder: str | None,
) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_file():
        if subfolder is not None:
            raise ValueError("subfolder cannot be used when source is a weight file")
        return source_path
    component_dir = checkpoint_dir if subfolder is None else checkpoint_dir / subfolder
    weights_path = component_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights_path}")
    return weights_path


class RenderFormerModel(TriangleRadiosityTransformer):
    """The trainable RenderFormer transformer component.

    ``from_pretrained`` intentionally leaves the module trainable and in its
    default training mode.  Pipelines switch it to frozen evaluation mode;
    training entrypoints can configure freezing and optimizers themselves.
    """

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        *,
        config: RenderTransformerConfig | InferenceConfig | dict[str, Any] | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        token: str | bool | None = None,
        local_files_only: bool = False,
        subfolder: str | None = None,
        strict: bool = True,
        ignore_mismatched_sizes: bool | tuple[str, ...] | list[str] = False,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        output_loading_info: bool = False,
    ):
        """Load only the transformer component from a local or Hub checkpoint.

        A direct ``.safetensors`` path is supported when ``config`` is supplied
        (or a sibling config file exists).  A directory/Hub source may also use
        ``subfolder`` for a component-style checkpoint layout.
        """

        for name, value in (
            ("force_download", force_download),
            ("local_files_only", local_files_only),
            ("strict", strict),
            ("output_loading_info", output_loading_info),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if isinstance(ignore_mismatched_sizes, bool):
            ignored_prefixes = None if ignore_mismatched_sizes else ()
        elif isinstance(ignore_mismatched_sizes, (tuple, list)):
            if not all(
                isinstance(prefix, str) and bool(prefix)
                for prefix in ignore_mismatched_sizes
            ):
                raise TypeError(
                    "ignore_mismatched_sizes must contain nonempty string prefixes"
                )
            ignored_prefixes = tuple(ignore_mismatched_sizes)
        else:
            raise TypeError(
                "ignore_mismatched_sizes must be a bool or a tuple/list of "
                "nonempty string prefixes"
            )
        if torch_dtype is not None:
            if not isinstance(torch_dtype, torch.dtype):
                raise TypeError(
                    f"torch_dtype must be a torch.dtype, got {type(torch_dtype).__name__}"
                )
            if not torch_dtype.is_floating_point:
                raise ValueError(f"torch_dtype must be floating point, got {torch_dtype}")

        try:
            import safetensors.torch
        except ImportError as exc:
            raise RuntimeError("checkpoint loading requires safetensors and torch") from exc

        from renderformer.checkpoints import (
            normalize_checkpoint_subfolder,
            resolve_checkpoint,
        )

        subfolder = normalize_checkpoint_subfolder(subfolder)
        checkpoint_dir = resolve_checkpoint(
            source,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        model_config = _coerce_model_config(config, checkpoint_dir, subfolder)
        weights_path = _resolve_local_weight_file(source, checkpoint_dir, subfolder)
        state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")

        model = cls(model_config)
        expected = model.state_dict()
        mismatched = {
            name: (tuple(tensor.shape), tuple(expected[name].shape))
            for name, tensor in state_dict.items()
            if name in expected and tensor.shape != expected[name].shape
        }
        unignored_mismatches = {
            name: shapes
            for name, shapes in mismatched.items()
            if ignored_prefixes is not None
            and not any(name.startswith(prefix) for prefix in ignored_prefixes)
        }
        if unignored_mismatches:
            preview = dict(list(unignored_mismatches.items())[:8])
            raise RuntimeError(f"checkpoint contains mismatched tensor shapes: {preview}")
        if mismatched and ignored_prefixes == ():
            preview = dict(list(mismatched.items())[:8])
            raise RuntimeError(f"checkpoint contains mismatched tensor shapes: {preview}")
        if mismatched:
            state_dict = {
                name: tensor
                for name, tensor in state_dict.items()
                if name not in mismatched
            }

        incompatible = model.load_state_dict(state_dict, strict=False)
        del state_dict
        ignored_mismatch_keys = set(mismatched)
        missing_keys = [
            name
            for name in incompatible.missing_keys
            if name not in ignored_mismatch_keys
        ]
        unexpected_keys = list(incompatible.unexpected_keys)
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                "checkpoint state dict is incompatible: "
                f"missing_keys={missing_keys[:8]}, unexpected_keys={unexpected_keys[:8]}"
            )
        if torch_dtype is None:
            model.to(device=device)
        else:
            model.to(device=device, dtype=torch_dtype)
        loading_info = {
            "checkpoint_dir": checkpoint_dir,
            "weights_path": weights_path,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched,
        }
        model.loading_info = loading_info
        if output_loading_info:
            return model, loading_info
        return model

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save this trainable transformer as a standalone model component.

        The bundle intentionally contains only ``model.safetensors`` and the
        architecture-only ``model_config.json``. Optimizer, scheduler, runtime
        preprocessing, and auxiliary V2 encoders remain owned by training or
        inference orchestration.
        """

        try:
            import safetensors.torch
        except ImportError as exc:
            raise RuntimeError("checkpoint saving requires safetensors and torch") from exc

        save_directory = Path(save_directory).expanduser()
        if save_directory.exists() and not save_directory.is_dir():
            raise NotADirectoryError(
                f"model component destination is not a directory: {save_directory}"
            )
        save_directory.mkdir(parents=True, exist_ok=True)
        weights_path = save_directory / "model.safetensors"
        state_dict = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.state_dict().items()
        }
        safetensors.torch.save_file(
            state_dict,
            str(weights_path),
            metadata={"format": "pt", "component": "RenderFormerModel"},
        )
        payload = {
            "format": "renderformer-model",
            "format_version": 1,
            "model_config": dataclasses.asdict(self.config),
        }
        (save_directory / "model_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return save_directory
