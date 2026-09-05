"""Training-only construction and pretrained model-component loading."""

from __future__ import annotations

from pathlib import Path

from renderformer.models.config import RenderTransformerConfig
from renderformer.modeling import RenderFormerModel


def load_training_model(
    model_config: RenderTransformerConfig,
    checkpoint: str | Path | None,
    *,
    strict: bool = True,
    ignore_mismatched_sizes: bool | tuple[str, ...] = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    subfolder: str | None = None,
) -> RenderFormerModel:
    """Build a scratch model or load only its trainable checkpoint component.

    This helper intentionally does not import or instantiate an inference
    pipeline. Full optimizer/scheduler resume remains the separate explicit
    ``trainer_config.load_ckpt`` or ``trainer_config.auto_resume`` path handled
    by Accelerate.
    """

    if checkpoint is None:
        return RenderFormerModel(model_config)
    if strict and ignore_mismatched_sizes:
        raise ValueError(
            "strict model loading cannot ignore mismatched tensor sizes"
        )

    print(f"Loading model component from {checkpoint}", flush=True)
    model, loading_info = RenderFormerModel.from_pretrained(
        checkpoint,
        config=model_config,
        strict=strict,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        subfolder=subfolder,
        output_loading_info=True,
    )
    print(
        "Model load summary: "
        f"missing={len(loading_info['missing_keys'])}, "
        f"unexpected={len(loading_info['unexpected_keys'])}, "
        f"mismatched={len(loading_info['mismatched_keys'])}",
        flush=True,
    )
    if not strict:
        for label in ("missing_keys", "unexpected_keys", "mismatched_keys"):
            keys = list(loading_info[label])
            if keys:
                print(f"  {label}: {keys}", flush=True)
    return model
