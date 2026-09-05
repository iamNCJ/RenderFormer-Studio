"""Hugging Face mixin adapter for models stored in component subfolders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import PyTorchModelHubMixin, snapshot_download

from renderformer.checkpoints import normalize_checkpoint_subfolder


class ComponentModelHubMixin(PyTorchModelHubMixin):
    """Add Diffusers-style ``subfolder`` loading to PyTorchModelHubMixin."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        force_download: bool = False,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **model_kwargs: Any,
    ):
        if subfolder is None:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                force_download=force_download,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **model_kwargs,
            )

        subfolder = normalize_checkpoint_subfolder(subfolder)
        source_path = Path(pretrained_model_name_or_path).expanduser()
        if source_path.is_file():
            raise ValueError("subfolder cannot be used when source is a weight file")
        if source_path.is_dir():
            component_dir = source_path / subfolder
        else:
            snapshot = snapshot_download(
                repo_id=str(pretrained_model_name_or_path),
                revision=revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                force_download=force_download,
                token=token,
                local_files_only=local_files_only,
                allow_patterns=[
                    f"{subfolder}/config.json",
                    f"{subfolder}/model.safetensors",
                    f"{subfolder}/preprocessor_config.json",
                ],
            )
            component_dir = Path(snapshot) / subfolder
        if not component_dir.is_dir():
            raise FileNotFoundError(
                f"model component subfolder not found: {component_dir}"
            )
        return super().from_pretrained(
            component_dir,
            local_files_only=True,
            **model_kwargs,
        )


__all__ = ["ComponentModelHubMixin"]
