"""Standalone material-autoencoder training with explicit HDR preprocessing."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from renderformer.data.material_exr import (
    MaterialEXRDataset,
    load_material_exr_manifest,
    sha256_file,
)
from renderformer.data.material_preprocessing import (
    MATERIAL_PREPROCESSING_MODES,
    MaterialPreprocessor,
)
from renderformer.models.material import MaterialAutoencoder


CHECKPOINT_FORMAT = "renderformer-material-training-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="renderformer train material",
        description=(
            "Train the checkpoint-compatible 3x256x256 material autoencoder "
            "from an explicit EXR JSONL manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Strict material EXR JSONL manifest.")
    parser.add_argument("--model-config", required=True, help="Material architecture YAML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--preprocessing-mode",
        required=True,
        choices=MATERIAL_PREPROCESSING_MODES,
        help="Explicit physical-HDR to network-space transform.",
    )
    parser.add_argument(
        "--no-apply-alpha",
        dest="apply_alpha",
        action="store_false",
        help="Drop EXR alpha without multiplying it into RGB.",
    )
    parser.set_defaults(apply_alpha=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for historical ||D(z)-D(clamp(z+xi))||_1 regularization; "
            "zero disables it."
        ),
    )
    parser.add_argument(
        "--smoothness-noise-std",
        type=float,
        default=2.0 / 255.0,
    )
    parser.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N.")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--resume",
        help="Resume a full checkpoint produced by this runner.",
    )
    initialization.add_argument(
        "--load-model",
        help=(
            "Load model weights only from a legacy .pt checkpoint, safetensors "
            "file, local pretrained directory, or Hugging Face model ID."
        ),
    )
    parser.add_argument("--load-model-revision")
    parser.add_argument(
        "--load-model-subfolder",
        help="Component subfolder inside a local or Hugging Face model bundle.",
    )
    parser.add_argument(
        "--allow-missing-preprocessing-metadata",
        action="store_true",
        help=(
            "Acknowledge that a model-only source has no preprocessing metadata. "
            "Use this for an original standalone legacy .pt, whose log10_1p "
            "contract was recovered from training history but not embedded in "
            "the file."
        ),
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use torchrun-provided RANK/WORLD_SIZE/LOCAL_RANK.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable lazy W&B logging.")
    parser.add_argument("--wandb-project", default="renderformer-material-autoencoder")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read and preprocess every manifest row without constructing the model.",
    )
    return parser


def _positive_int(value: int, *, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "epochs", "log_interval", "save_every_epochs"):
        _positive_int(getattr(args, name), name=name.replace("_", "-"))
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be non-negative and finite")
    if not math.isfinite(args.smoothness_weight) or args.smoothness_weight < 0:
        raise ValueError("smoothness-weight must be non-negative and finite")
    if not math.isfinite(args.smoothness_noise_std) or args.smoothness_noise_std <= 0:
        raise ValueError("smoothness-noise-std must be positive and finite")
    if not math.isfinite(args.lr_factor) or not 0 < args.lr_factor < 1:
        raise ValueError("lr-factor must be finite and in (0, 1)")
    if args.lr_patience < 0:
        raise ValueError("lr-patience must be non-negative")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.load_model_revision is not None and args.load_model is None:
        raise ValueError("load-model-revision requires --load-model")
    if args.load_model_subfolder is not None and args.load_model is None:
        raise ValueError("load-model-subfolder requires --load-model")
    if args.allow_missing_preprocessing_metadata and args.load_model is None:
        raise ValueError(
            "allow-missing-preprocessing-metadata requires --load-model"
        )
    if args.validate_only and args.ddp:
        raise ValueError("validate-only is a single-process operation; omit --ddp")


def _load_model_config(path: str | Path) -> dict[str, int | str]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"material model config must be a YAML mapping: {config_path}")
    required = {"architecture", "latent_dim", "image_channels", "image_size"}
    missing = sorted(required - set(raw))
    unknown = sorted(set(raw) - required)
    if missing or unknown:
        raise ValueError(
            f"material model config must contain exactly {sorted(required)}; "
            f"missing={missing}, unknown={unknown}: {config_path}"
        )
    if raw["architecture"] != "MaterialAutoencoder":
        raise ValueError(
            "material model architecture must be 'MaterialAutoencoder', got "
            f"{raw['architecture']!r}"
        )
    for field in ("latent_dim", "image_channels", "image_size"):
        value = raw[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"material model {field} must be an integer")
    expected_dimensions = {
        "latent_dim": 9,
        "image_channels": 3,
        "image_size": 256,
    }
    actual_dimensions = {field: raw[field] for field in expected_dimensions}
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            "the checkpoint-compatible material architecture requires "
            f"{expected_dimensions}, got {actual_dimensions}"
        )
    return {
        "architecture": str(raw["architecture"]),
        "latent_dim": int(raw["latent_dim"]),
        "image_channels": int(raw["image_channels"]),
        "image_size": int(raw["image_size"]),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_contract(
    *,
    model_config: Mapping[str, Any],
    preprocessor: MaterialPreprocessor,
    manifest_path: Path,
    train_count: int,
    validation_count: int,
    apply_alpha: bool,
    smoothness_weight: float,
    smoothness_noise_std: float,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "model": dict(model_config),
        "preprocessing": preprocessor.to_dict(),
        "dataset": {
            "manifest_sha256": sha256_file(manifest_path),
            "train_samples": train_count,
            "validation_samples": validation_count,
            "apply_alpha": apply_alpha,
        },
        "objective": {
            "reconstruction": "l1",
            "smoothness_weight": smoothness_weight,
            "smoothness_noise_std": smoothness_noise_std,
            "smoothness_latent_clamp": [-1.0, 1.0],
        },
    }


def _source_preprocessing_from_mapping(
    payload: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    contract = payload.get("material_training_contract")
    if isinstance(contract, Mapping):
        preprocessing = contract.get("preprocessing")
        if isinstance(preprocessing, Mapping):
            return preprocessing
    return None


def _validate_source_preprocessing(
    source_preprocessing: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    *,
    allow_missing_metadata: bool,
    source: str,
) -> None:
    if source_preprocessing is None:
        if not allow_missing_metadata:
            raise ValueError(
                f"model source {source!r} has no preprocessing metadata. The "
                "source files predating explicit preprocessing metadata require "
                "--allow-missing-preprocessing-metadata."
            )
        print(
            f"warning: {source!r} has no embedded preprocessing metadata; "
            f"loading under the explicitly selected contract {dict(current)}",
            file=sys.stderr,
            flush=True,
        )
        return
    if dict(source_preprocessing) != dict(current):
        raise ValueError(
            f"model source preprocessing does not match this run: "
            f"source={dict(source_preprocessing)}, current={dict(current)}"
        )


def _load_model_only(
    model: MaterialAutoencoder,
    source: str,
    *,
    revision: str | None,
    subfolder: str | None,
    current_preprocessing: Mapping[str, Any],
    allow_missing_preprocessing_metadata: bool,
) -> None:
    def load_preprocessing_metadata(path: Path) -> Mapping[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise TypeError(f"preprocessor config must be an object: {path}")
        return metadata

    from renderformer.checkpoints import normalize_checkpoint_subfolder

    subfolder = normalize_checkpoint_subfolder(subfolder)
    source_path = Path(source).expanduser()
    if revision is not None and source_path.exists():
        raise ValueError("load-model-revision is valid only for a Hugging Face model ID")
    source_preprocessing: Mapping[str, Any] | None = None
    if source_path.is_file():
        if subfolder is not None:
            raise ValueError("load-model-subfolder cannot be used with a weight file")
        if source_path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(source_path), device="cpu")
            metadata_path = source_path.parent / "preprocessor_config.json"
            if metadata_path.is_file():
                source_preprocessing = load_preprocessing_metadata(metadata_path)
        else:
            loaded = torch.load(source_path, map_location="cpu", weights_only=True)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"model checkpoint must deserialize to a mapping: {source_path}")
            state_dict = loaded.get("model_state_dict")
            if not isinstance(state_dict, Mapping):
                raise KeyError(f"model checkpoint is missing model_state_dict: {source_path}")
            source_preprocessing = _source_preprocessing_from_mapping(loaded)
        model.load_state_dict(state_dict, strict=True)
    elif source_path.is_dir():
        component_dir = source_path if subfolder is None else source_path / subfolder
        loaded_model = MaterialAutoencoder.from_pretrained(
            str(component_dir),
            strict=True,
        )
        model.load_state_dict(loaded_model.state_dict(), strict=True)
        metadata_path = component_dir / "preprocessor_config.json"
        if metadata_path.is_file():
            source_preprocessing = load_preprocessing_metadata(metadata_path)
    else:
        kwargs: dict[str, Any] = {"strict": True}
        if revision is not None:
            kwargs["revision"] = revision
        loaded_model = MaterialAutoencoder.from_pretrained(
            source,
            subfolder=subfolder,
            **kwargs,
        )
        model.load_state_dict(loaded_model.state_dict(), strict=True)
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            metadata_path = Path(
                hf_hub_download(
                    repo_id=source,
                    filename="preprocessor_config.json",
                    subfolder=subfolder,
                    revision=revision,
                )
            )
        except EntryNotFoundError:
            pass
        else:
            source_preprocessing = load_preprocessing_metadata(metadata_path)

    _validate_source_preprocessing(
        source_preprocessing,
        current_preprocessing,
        allow_missing_metadata=allow_missing_preprocessing_metadata,
        source=source,
    )


def _load_resume_checkpoint(
    path: str | Path,
    *,
    model: MaterialAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    contract: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"training checkpoint must deserialize to a mapping: {checkpoint_path}")
    if loaded.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"--resume requires a {CHECKPOINT_FORMAT} full checkpoint; use "
            "--load-model for historical model-only initialization"
        )
    stored_contract = loaded.get("material_training_contract")
    if stored_contract != contract:
        raise ValueError(
            "resume checkpoint contract differs from the requested run; use "
            "--load-model to start a new run when changing data or preprocessing"
        )
    state_dict = loaded.get("model_state_dict")
    optimizer_state = loaded.get("optimizer_state_dict")
    scheduler_state = loaded.get("scheduler_state_dict")
    if not isinstance(state_dict, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError(f"resume checkpoint is missing model/optimizer state: {checkpoint_path}")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError(f"resume checkpoint is missing scheduler state: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    next_epoch = loaded.get("next_epoch")
    global_step = loaded.get("global_step")
    best_validation_loss = loaded.get("best_validation_loss")
    if not isinstance(next_epoch, int) or next_epoch < 0:
        raise ValueError(f"invalid next_epoch in {checkpoint_path}")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError(f"invalid global_step in {checkpoint_path}")
    if not isinstance(best_validation_loss, (int, float)):
        raise ValueError(f"invalid best_validation_loss in {checkpoint_path}")
    return next_epoch, global_step, float(best_validation_loss)


def _save_checkpoint(
    path: Path,
    *,
    model: MaterialAutoencoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    contract: Mapping[str, Any],
    run_config: Mapping[str, Any],
    next_epoch: int,
    global_step: int,
    best_validation_loss: float,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "next_epoch": next_epoch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "material_training_contract": dict(contract),
            "run_config": dict(run_config),
        },
    )


def _save_pretrained_bundle(
    path: Path,
    *,
    model: MaterialAutoencoder,
    preprocessor: MaterialPreprocessor,
) -> None:
    """Export model-only weights plus the physical-input contract."""

    model.save_pretrained(path)
    _atomic_json(path / "preprocessor_config.json", preprocessor.to_dict())


def _setup_distributed(args: argparse.Namespace) -> tuple[int, int, int, torch.device, Any]:
    if not args.ddp:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return 0, 1, 0, device, None

    missing = [name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ]
    if missing:
        raise RuntimeError(f"--ddp requires torchrun environment variables: {missing}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError(
            f"invalid distributed ranks: rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}"
        )
    if args.device == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP CUDA was requested but CUDA is unavailable")
        if local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} exceeds {torch.cuda.device_count()} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    import torch.distributed as distributed

    distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, local_rank, device, distributed


def _make_loaders(
    *,
    train_dataset: MaterialEXRDataset,
    validation_dataset: MaterialEXRDataset | None,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, Any, Any]:
    train_sampler = None
    validation_sampler = None
    generator = torch.Generator().manual_seed(args.seed)
    if args.ddp:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if validation_dataset is not None:
            validation_sampler = DistributedSampler(
                validation_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = DataLoader(
            validation_dataset,
            shuffle=False,
            sampler=validation_sampler,
            **loader_kwargs,
        )
    return train_loader, validation_loader, train_sampler, validation_sampler


def _smoothness_loss(
    model: MaterialAutoencoder,
    latent: torch.Tensor,
    *,
    noise_std: float,
) -> torch.Tensor:
    perturbation = torch.randn_like(latent) * noise_std
    perturbed_latent = (latent + perturbation).clamp(-1.0, 1.0)
    return F.l1_loss(model.decode(latent), model.decode(perturbed_latent))


def _reduce_epoch_metrics(
    values: tuple[float, float, float, int],
    *,
    device: torch.device,
    distributed: Any,
) -> tuple[float, float, float]:
    totals = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed is not None:
        distributed.all_reduce(totals)
    sample_count = int(totals[3].item())
    if sample_count < 1:
        raise RuntimeError("epoch produced no samples")
    return tuple(float(totals[index].item() / sample_count) for index in range(3))


def _run_epoch(
    *,
    wrapped_model: torch.nn.Module,
    model: MaterialAutoencoder,
    loader: DataLoader,
    preprocessor: MaterialPreprocessor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    smoothness_weight: float,
    smoothness_noise_std: float,
    distributed: Any,
    rank: int,
    global_step: int,
    log_interval: int,
    wandb_run: Any,
) -> tuple[tuple[float, float, float], int]:
    training = optimizer is not None
    wrapped_model.train(training)
    total_loss = 0.0
    total_reconstruction = 0.0
    total_smoothness = 0.0
    sample_count = 0

    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for batch_index, batch in enumerate(loader, start=1):
            physical_images = batch["image"].to(device, non_blocking=device.type == "cuda")
            network_images = preprocessor(physical_images)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            reconstructed, latent = wrapped_model(network_images)
            reconstruction = F.l1_loss(reconstructed, network_images)
            if smoothness_weight > 0:
                smoothness = _smoothness_loss(
                    model,
                    latent,
                    noise_std=smoothness_noise_std,
                )
            else:
                smoothness = reconstruction.new_zeros(())
            loss = reconstruction + smoothness_weight * smoothness
            if not bool(torch.isfinite(loss)):
                sample_ids = ", ".join(str(value) for value in batch["id"])
                raise FloatingPointError(
                    f"non-finite material loss for manifest samples: {sample_ids}"
                )
            if optimizer is not None:
                loss.backward()
                optimizer.step()
                global_step += 1

            batch_size = int(network_images.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_reconstruction += float(reconstruction.detach()) * batch_size
            total_smoothness += float(smoothness.detach()) * batch_size
            sample_count += batch_size

            if training and rank == 0 and global_step % log_interval == 0:
                message = (
                    f"step={global_step} batch={batch_index}/{len(loader)} "
                    f"loss={float(loss.detach()):.6f} "
                    f"reconstruction={float(reconstruction.detach()):.6f} "
                    f"smoothness={float(smoothness.detach()):.6f}"
                )
                print(message, flush=True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(loss.detach()),
                            "train/reconstruction_l1": float(reconstruction.detach()),
                            "train/smoothness_l1": float(smoothness.detach()),
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

    metrics = _reduce_epoch_metrics(
        (total_loss, total_reconstruction, total_smoothness, sample_count),
        device=device,
        distributed=distributed,
    )
    return metrics, global_step


def _validate_dataset(
    datasets: Sequence[MaterialEXRDataset],
    preprocessor: MaterialPreprocessor,
) -> None:
    total = 0
    maximum_roundtrip_error = 0.0
    for dataset in datasets:
        for index in range(len(dataset)):
            image = dataset[index]["image"]
            network = preprocessor(image)
            restored = preprocessor.inverse(network)
            maximum_roundtrip_error = max(
                maximum_roundtrip_error,
                float(torch.max(torch.abs(restored - image))),
            )
            total += 1
    print(
        f"valid: {total} EXR samples; preprocessing={preprocessor.mode}; "
        f"max_roundtrip_abs_error={maximum_roundtrip_error:.6g}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        model_config = _load_model_config(args.model_config)
        manifest_path = Path(args.manifest).expanduser().resolve()
        records = load_material_exr_manifest(manifest_path)
        preprocessor = MaterialPreprocessor(args.preprocessing_mode)
        train_dataset = MaterialEXRDataset(
            records,
            split="train",
            image_channels=int(model_config["image_channels"]),
            image_size=int(model_config["image_size"]),
            apply_alpha=args.apply_alpha,
        )
        validation_dataset = None
        if any(record.split == "validation" for record in records):
            validation_dataset = MaterialEXRDataset(
                records,
                split="validation",
                image_channels=int(model_config["image_channels"]),
                image_size=int(model_config["image_size"]),
                apply_alpha=args.apply_alpha,
            )
        if args.validate_only:
            datasets = [train_dataset]
            if validation_dataset is not None:
                datasets.append(validation_dataset)
            _validate_dataset(datasets, preprocessor)
            return 0
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    rank, world_size, local_rank, device, distributed = _setup_distributed(args)
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + rank)

        output_dir = Path(args.output_dir).expanduser().resolve()
        checkpoint_dir = output_dir / "checkpoints"
        pointer_path = checkpoint_dir / "latest_checkpoint.json"
        protected_outputs = (
            output_dir / "material_training_contract.json",
            output_dir / "preprocessor_config.json",
            checkpoint_dir,
            output_dir / "best_model",
            output_dir / "final_model",
        )
        existing_outputs = [path for path in protected_outputs if path.exists()]
        if existing_outputs and args.resume is None:
            raise FileExistsError(
                "refusing to overwrite material training artifacts in "
                f"{output_dir}: {existing_outputs}; use --resume or choose "
                "another output directory"
            )
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed is not None:
            distributed.barrier()

        contract = _build_contract(
            model_config=model_config,
            preprocessor=preprocessor,
            manifest_path=manifest_path,
            train_count=len(train_dataset),
            validation_count=0 if validation_dataset is None else len(validation_dataset),
            apply_alpha=args.apply_alpha,
            smoothness_weight=args.smoothness_weight,
            smoothness_noise_std=args.smoothness_noise_std,
        )
        run_config = {
            "batch_size_per_rank": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "num_workers_per_rank": args.num_workers,
            "seed": args.seed,
            "world_size": world_size,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": args.lr_factor,
                "patience": args.lr_patience,
            },
        }
        train_loader, validation_loader, train_sampler, _ = _make_loaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            args=args,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        model = MaterialAutoencoder(
            latent_dim=int(model_config["latent_dim"]),
            image_channels=int(model_config["image_channels"]),
            image_size=int(model_config["image_size"]),
        ).to(device)
        if args.load_model is not None:
            _load_model_only(
                model,
                args.load_model,
                revision=args.load_model_revision,
                subfolder=args.load_model_subfolder,
                current_preprocessing=preprocessor.to_dict(),
                allow_missing_preprocessing_metadata=args.allow_missing_preprocessing_metadata,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )
        start_epoch = 0
        global_step = 0
        best_validation_loss = math.inf
        if args.resume is not None:
            start_epoch, global_step, best_validation_loss = _load_resume_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                contract=contract,
                device=device,
            )
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint starts at epoch {start_epoch}, but --epochs={args.epochs}; "
                "increase --epochs to continue"
            )
        if rank == 0:
            _atomic_json(output_dir / "material_training_contract.json", contract)
            _atomic_json(
                output_dir / "preprocessor_config.json",
                preprocessor.to_dict(),
            )
        if distributed is not None:
            distributed.barrier()

        wrapped_model: torch.nn.Module = model
        if args.ddp:
            from torch.nn.parallel import DistributedDataParallel

            wrapped_model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )

        wandb_run = None
        if args.wandb and rank == 0:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={"contract": contract, "run": run_config},
            )

        if rank == 0:
            print(
                f"training MaterialAutoencoder ({sum(parameter.numel() for parameter in model.parameters()):,} parameters) "
                f"on {len(train_dataset)} train / "
                f"{0 if validation_dataset is None else len(validation_dataset)} validation samples; "
                f"preprocessing={preprocessor.mode}; world_size={world_size}",
                flush=True,
            )

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics, global_step = _run_epoch(
                wrapped_model=wrapped_model,
                model=model,
                loader=train_loader,
                preprocessor=preprocessor,
                device=device,
                optimizer=optimizer,
                smoothness_weight=args.smoothness_weight,
                smoothness_noise_std=args.smoothness_noise_std,
                distributed=distributed,
                rank=rank,
                global_step=global_step,
                log_interval=args.log_interval,
                wandb_run=wandb_run,
            )
            if validation_loader is not None:
                validation_metrics, _ = _run_epoch(
                    wrapped_model=wrapped_model,
                    model=model,
                    loader=validation_loader,
                    preprocessor=preprocessor,
                    device=device,
                    optimizer=None,
                    smoothness_weight=args.smoothness_weight,
                    smoothness_noise_std=args.smoothness_noise_std,
                    distributed=distributed,
                    rank=rank,
                    global_step=global_step,
                    log_interval=args.log_interval,
                    wandb_run=None,
                )
                scheduler_metric = validation_metrics[0]
            else:
                validation_metrics = None
                scheduler_metric = train_metrics[0]
            scheduler.step(scheduler_metric)

            if rank == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} train_loss={train_metrics[0]:.6f} "
                    + (
                        ""
                        if validation_metrics is None
                        else f"validation_loss={validation_metrics[0]:.6f} "
                    )
                    + f"lr={optimizer.param_groups[0]['lr']:.3g}",
                    flush=True,
                )
                if wandb_run is not None:
                    metrics = {
                        "epoch": epoch + 1,
                        "train/loss_epoch": train_metrics[0],
                        "train/reconstruction_l1_epoch": train_metrics[1],
                        "train/smoothness_l1_epoch": train_metrics[2],
                        "learning_rate_epoch": optimizer.param_groups[0]["lr"],
                    }
                    if validation_metrics is not None:
                        metrics.update(
                            {
                                "validation/loss": validation_metrics[0],
                                "validation/reconstruction_l1": validation_metrics[1],
                                "validation/smoothness_l1": validation_metrics[2],
                            }
                        )
                    wandb_run.log(metrics, step=global_step)

                improved = scheduler_metric < best_validation_loss
                if improved:
                    best_validation_loss = scheduler_metric
                    _save_checkpoint(
                        checkpoint_dir / "best_model.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        contract=contract,
                        run_config=run_config,
                        next_epoch=epoch + 1,
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                    )
                    _save_pretrained_bundle(
                        output_dir / "best_model",
                        model=model,
                        preprocessor=preprocessor,
                    )
                if (epoch + 1) % args.save_every_epochs == 0:
                    epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
                    _save_checkpoint(
                        epoch_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        contract=contract,
                        run_config=run_config,
                        next_epoch=epoch + 1,
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                    )
                    _atomic_json(
                        pointer_path,
                        {
                            "checkpoint": epoch_path.name,
                            "next_epoch": epoch + 1,
                            "global_step": global_step,
                        },
                    )
            if distributed is not None:
                distributed.barrier()

        if rank == 0:
            _save_checkpoint(
                checkpoint_dir / "final_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                contract=contract,
                run_config=run_config,
                next_epoch=args.epochs,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
            )
            _save_pretrained_bundle(
                output_dir / "final_model",
                model=model,
                preprocessor=preprocessor,
            )
            _atomic_json(
                pointer_path,
                {
                    "checkpoint": "final_model.pt",
                    "next_epoch": args.epochs,
                    "global_step": global_step,
                },
            )
            if wandb_run is not None:
                wandb_run.finish()
        if distributed is not None:
            distributed.barrier()
        return 0
    finally:
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
