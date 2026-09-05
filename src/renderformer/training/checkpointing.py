"""Lightweight training checkpoint save/resume semantics."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Optional

import yaml


def checkpoint_step(checkpoint_dir: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "training_state.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        step = metadata.get("global_step") if isinstance(metadata, dict) else None
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"invalid training step metadata: {metadata_path}")
        return step
    match = re.fullmatch(r"ckpt_(\d+)", checkpoint_dir.name)
    if match is None:
        raise ValueError(
            "checkpoint directory must contain training_state.json or be named "
            f"ckpt_<step>: {checkpoint_dir}"
        )
    return int(match.group(1))


def auto_resume(
    accelerator,
    output_dir: Path,
    load_ckpt: Optional[str] = None,
    *,
    enabled: bool = False,
) -> int:
    """Load an explicit state, or scan the run directory when enabled."""

    if load_ckpt is not None:
        step = checkpoint_step(load_ckpt)
        print(f"Loading checkpoint from {load_ckpt} at step {step}", flush=True)
        accelerator.load_state(load_ckpt)
        return step
    if not enabled:
        return 0
    checkpoint_root = Path(output_dir, "checkpoints")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(
            f"auto-resume checkpoint root not found: {checkpoint_root}"
        )
    latest_path = checkpoint_root / "latest_checkpoint.json"
    checkpoint_dirs = []
    if latest_path.is_file():
        with latest_path.open("r", encoding="utf-8") as handle:
            latest = json.load(handle)
        checkpoint_name = latest.get("checkpoint") if isinstance(latest, dict) else None
        if not isinstance(checkpoint_name, str) or re.fullmatch(
            r"ckpt_\d+", checkpoint_name
        ) is None:
            raise ValueError(f"invalid latest checkpoint metadata: {latest_path}")
        checkpoint_dirs.append(checkpoint_root / checkpoint_name)
    # Compatibility fallback for runs created before latest_checkpoint.json.
    checkpoint_dirs.extend(
        path
        for path in checkpoint_root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"ckpt_\d+", path.name)
        and path not in checkpoint_dirs
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"auto-resume found no ckpt_<step> directories under {checkpoint_root}"
        )
    load_errors = []
    for checkpoint_dir in sorted(
        checkpoint_dirs,
        key=lambda path: int(path.name.removeprefix("ckpt_")),
        reverse=True,
    ):
        print(f"Loading checkpoint from {checkpoint_dir}", flush=True)
        try:
            step = checkpoint_step(checkpoint_dir)
            accelerator.load_state(str(checkpoint_dir))
            return step
        except Exception as exc:
            load_errors.append((checkpoint_dir, exc))
            print(
                f"Error loading checkpoint {checkpoint_dir}: {exc}, try next one...",
                flush=True,
            )
    attempted = ", ".join(str(path) for path, _ in load_errors)
    raise RuntimeError(
        f"auto-resume failed to load every candidate checkpoint: {attempted}"
    ) from load_errors[-1][1]


def maybe_save_ckpt(accelerator, output_dir: Path, config, global_step: int) -> None:
    """Save full Accelerator state, config, and explicit step metadata."""

    if (global_step + 1) % config.trainer_config.save_interval != 0:
        return
    checkpoint_dir = Path(output_dir, "checkpoints", f"ckpt_{global_step + 1}")
    accelerator.save_state(str(checkpoint_dir))
    if accelerator.is_main_process:
        with (checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle)
        with (checkpoint_dir / "training_state.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump({"global_step": global_step + 1}, handle, indent=2)
            handle.write("\n")
        with (checkpoint_dir.parent / "latest_checkpoint.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "checkpoint": checkpoint_dir.name,
                    "global_step": global_step + 1,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
