"""Command-line entrypoint for V1 and V2 H5 inference."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
from typing import TYPE_CHECKING
import unicodedata

from renderformer.config import CheckpointVersion

if TYPE_CHECKING:
    import numpy as np
    import torch


def _device_from_arg(value: str) -> torch.device:
    import torch

    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _precision_from_arg(value: str, device: torch.device) -> str:
    if value != "auto":
        return value
    return "fp16" if device.type == "cuda" else "fp32"


def _tone_map(hdr: np.ndarray, name: str) -> np.ndarray:
    import numpy as np

    if name == "none":
        return np.clip(hdr, 0.0, 1.0)
    try:
        from simple_ocio import ToneMapper
    except ImportError as exc:
        raise RuntimeError("tone mapping requires the inference extra") from exc
    ocio_name = "Khronos PBR Neutral" if name == "pbr_neutral" else name
    return ToneMapper(ocio_name).hdr_to_ldr(hdr)


def _output_dir(input_arg: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    if input_arg.is_file():
        return input_arg.parent / "renderformer_outputs"
    return input_arg / "renderformer_outputs"


def _manifest_input_path(path: Path) -> str:
    """Record an unambiguous absolute H5 path in an inference receipt."""

    return os.fspath(path.expanduser().resolve())


def _load_scene_worker(arguments):
    from renderformer.config import CheckpointVersion
    from renderformer.data import load_h5_scene

    path, version, image_key, allow_legacy_rf1_dtypes = arguments
    return load_h5_scene(
        path,
        CheckpointVersion(version),
        image_key=image_key,
        allow_legacy_rf1_dtypes=allow_legacy_rf1_dtypes,
    )


@contextmanager
def _scene_loader_pool(num_workers: int, window_size: int, input_count: int):
    """Reuse one bounded spawn pool and shut it down cleanly."""

    worker_count = min(num_workers, window_size, input_count)
    if worker_count == 0:
        yield None
        return
    pool = multiprocessing.get_context("spawn").Pool(worker_count)
    try:
        yield pool
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()


def _validate_unique_output_stems(inputs: list[Path]) -> None:
    """Reject inputs that would map to the same flat output filename."""

    by_stem: dict[str, list[Path]] = {}
    for path in inputs:
        output_key = unicodedata.normalize("NFC", path.stem).casefold()
        by_stem.setdefault(output_key, []).append(path)
    collisions = [paths for paths in by_stem.values() if len(paths) > 1]
    if collisions:
        details = "; ".join(
            ", ".join(os.fspath(path) for path in paths)
            for paths in collisions
        )
        raise ValueError(
            "input H5 files must have unique filename stems because outputs use a "
            f"flat directory; conflicting paths: {details}"
        )


def _preflight_h5_inputs(inputs: list[Path]) -> None:
    """Open every H5 header before model loading or partial output writes."""

    import h5py

    for path in inputs:
        with h5py.File(path, "r"):
            pass


def _validate_padding_length(
    version: CheckpointVersion, padding_length: int | None
) -> None:
    if version is CheckpointVersion.V2 and padding_length is not None:
        raise ValueError("--padding-length is only valid for V1 checkpoints")


def _validate_view_batch_size(
    version: CheckpointVersion, view_batch_size: int | None
) -> None:
    if version is CheckpointVersion.V1 and view_batch_size is not None:
        raise ValueError("--view-batch-size is only valid for V2 checkpoints")


def _execution_profile_selection(
    version: CheckpointVersion,
    requested: str | None,
    *,
    view_transformer_tf32: bool | None = None,
    use_packed_sequence: bool | None = None,
) -> dict[str, str | dict[str, bool | None] | None]:
    """Resolve a variant-aware execution profile and its explicit overrides."""

    default = "checkpoint" if version is CheckpointVersion.V1 else "release"
    resolved = default if requested is None else requested
    if version is CheckpointVersion.V1 and (
        resolved != "checkpoint"
        or view_transformer_tf32 is not None
        or use_packed_sequence is not None
    ):
        raise ValueError(
            "V2 execution profile/overrides cannot be used with a V1 checkpoint"
        )
    return {
        "requested": requested,
        "default": default,
        "resolved": resolved,
        "overrides": {
            "view_transformer_tf32": view_transformer_tf32,
            "use_packed_sequence": use_packed_sequence,
        },
    }


def _fused_kernel_status(pipeline) -> dict[str, bool]:
    """Return the requested and effective fused-kernel state for a receipt."""

    return {
        "requested": pipeline.fused_kernels_requested,
        "applied": pipeline.fused_kernels_applied,
    }


def _default_checkpoint(variant: str, resolution: int) -> tuple[str, str | None]:
    """Choose an official model only when the requested generation is known."""

    if variant == "v2":
        from renderformer.checkpoints import (
            DEFAULT_V2_REPOSITORY,
            default_v2_transformer_subfolder,
        )

        return DEFAULT_V2_REPOSITORY, default_v2_transformer_subfolder(resolution)
    return "microsoft/renderformer-v1.1-swin-large", None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="renderformer infer",
        description="Render RF1/V1 or V2 H5 scenes with the shared training model"
    )
    parser.add_argument(
        "--input",
        "--h5-file",
        "--h5_file",
        "--h5-folder",
        "--h5_folder",
        dest="input_path",
        type=Path,
        required=True,
        help="One .h5 file, a newline manifest, or a non-recursive H5 directory",
    )
    parser.add_argument(
        "--checkpoint",
        "--model-id",
        "--model_id",
        "--model-path",
        "--model_path",
        dest="checkpoint",
        default=None,
        help=(
            "Local checkpoint directory or Hugging Face model ID; defaults to "
            "V1 Large, or the matching native V2 model with --variant v2"
        ),
    )
    parser.add_argument(
        "--revision",
        help=(
            "Optional Hugging Face branch, tag, or commit; defaults to the "
            "repository default revision"
        ),
    )
    parser.add_argument(
        "--checkpoint-subfolder",
        "--transformer-subfolder",
        dest="checkpoint_subfolder",
        help=(
            "Transformer component inside a bundled checkpoint repository; "
            "the official V2 bundle selects transformer_512 or "
            "transformer_2048 from --resolution"
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload Hub checkpoint files instead of reusing the local cache",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", "--output_dir", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, or cuda:N")
    parser.add_argument(
        "--precision",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--padding-length", "--padding_length", type=int)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--first-view-only", action="store_true")
    parser.add_argument(
        "--view-batch-size",
        "--view_batch_size",
        type=int,
        help=(
            "Maximum camera views per V2 model call; auto keeps the existing "
            "unchunked path below 2048 and uses one view per call at 2048+"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=["auto", "v1", "v2"],
        default="auto",
        help="Optional schema assertion; auto detects from checkpoint config",
    )
    parser.add_argument(
        "--allow-legacy-rf1-dtypes",
        action="store_true",
        help=(
            "Accept and normalize only the two historical dtype signatures in "
            "renderformer/renderformer-video-data; RF1 remains strict by default"
        ),
    )
    parser.add_argument(
        "--execution-profile",
        choices=["release", "checkpoint", "legacy_v2", "legacy_template_v2"],
        default=None,
        help=(
            "V2 execution semantics (default: release for V2, checkpoint for V1)"
        ),
    )
    parser.add_argument(
        "--view-transformer-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the V2 checkpoint's view-transformer execution mode",
    )
    parser.add_argument(
        "--packed-sequence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the V2 checkpoint's packed-sequence execution mode",
    )
    parser.add_argument(
        "--tone-mapper",
        "--tone_mapper",
        choices=["none", "agx", "filmic", "pbr_neutral"],
        default="none",
    )
    parser.add_argument(
        "--no-fused-kernels",
        action="store_true",
        help="Disable the repository's local CUDA kernel replacements",
    )
    parser.add_argument(
        "--optimize",
        choices=["none", "compile"],
        default="none",
        help="Optional torch.compile fusion; includes warm-up on first inputs",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Open every H5 and enforce checkpoint-specific input constraints "
            "without loading auxiliary encoders or running the model"
        ),
    )
    parser.add_argument(
        "--save-video",
        "--save_video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write all PNG outputs in input order to video.mp4",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resolution <= 0 or args.resolution % 8:
        raise ValueError("resolution must be a positive multiple of 8")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.padding_length is not None and args.padding_length <= 0:
        raise ValueError("padding_length must be positive")
    if args.view_batch_size is not None and args.view_batch_size <= 0:
        raise ValueError("view_batch_size must be positive")
    if args.checkpoint is None:
        args.checkpoint, default_subfolder = _default_checkpoint(
            args.variant,
            args.resolution,
        )
        if args.checkpoint_subfolder is None:
            args.checkpoint_subfolder = default_subfolder
    elif args.checkpoint_subfolder is None:
        from renderformer.checkpoints import select_v2_transformer_subfolder

        args.checkpoint_subfolder = select_v2_transformer_subfolder(
            args.checkpoint,
            resolution=args.resolution,
        )

    try:
        import imageio.v3 as iio
        import numpy as np
        import torch

        from renderformer.checkpoints import resolve_checkpoint
        from renderformer.config import load_inference_config
        from renderformer.data import collect_h5_inputs
        from renderformer.image_io import write_hdr_exr
        from renderformer.pipelines import (
            RenderFormerPipeline,
            RenderFormerV2Pipeline,
        )
    except ModuleNotFoundError as exc:
        parser.error(
            "inference dependencies are not installed "
            f"(missing {exc.name!r}); install this project with its inference "
            'extra, for example: python -m pip install -e ".[inference]"'
        )

    device = _device_from_arg(args.device)
    precision = _precision_from_arg(args.precision, device)
    revision = args.revision
    inputs = collect_h5_inputs(args.input_path)
    _validate_unique_output_stems(inputs)
    _preflight_h5_inputs(inputs)

    checkpoint_dir = resolve_checkpoint(
        args.checkpoint,
        revision=revision,
        cache_dir=args.cache_dir,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
        subfolder=args.checkpoint_subfolder,
    )
    config_dir = (
        checkpoint_dir
        if args.checkpoint_subfolder is None
        else checkpoint_dir / args.checkpoint_subfolder
    )
    detected_config = load_inference_config(config_dir)
    execution_profile_selection = _execution_profile_selection(
        detected_config.runtime.version,
        args.execution_profile,
        view_transformer_tf32=args.view_transformer_tf32,
        use_packed_sequence=args.packed_sequence,
    )
    execution_profile = execution_profile_selection["resolved"]
    assert isinstance(execution_profile, str)
    _validate_padding_length(detected_config.runtime.version, args.padding_length)
    _validate_view_batch_size(
        detected_config.runtime.version,
        args.view_batch_size,
    )
    if args.variant != "auto" and args.variant != detected_config.runtime.version.value:
        raise ValueError(
            f"--variant {args.variant} conflicts with detected "
            f"{detected_config.runtime.version.value} checkpoint schema"
        )
    if (
        args.allow_legacy_rf1_dtypes
        and detected_config.runtime.version is not CheckpointVersion.V1
    ):
        raise ValueError(
            "--allow-legacy-rf1-dtypes is only valid with an RF1/V1 checkpoint"
        )
    pipeline_class = (
        RenderFormerPipeline
        if detected_config.runtime.version is CheckpointVersion.V1
        else RenderFormerV2Pipeline
    )
    component_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]
    pipeline_kwargs = {
        "cache_dir": args.cache_dir,
        "force_download": args.force_download,
        "local_files_only": args.local_files_only,
        "transformer_subfolder": args.checkpoint_subfolder,
        "device": device,
        "apply_fused_kernels": not args.no_fused_kernels,
    }
    if pipeline_class is RenderFormerV2Pipeline:
        environment_encoder_dtype = (
            torch.float32
            if execution_profile == "legacy_v2"
            else component_dtype
        )
        pipeline_kwargs.update(
            resolution=args.resolution,
            environment_encoder_dtype=environment_encoder_dtype,
            texture_encoder_dtype=torch.float32,
            load_auxiliary_models=not args.validate_only,
        )
    pipeline = pipeline_class.from_pretrained(checkpoint_dir, **pipeline_kwargs)
    pipeline.optimize(args.optimize)
    print(
        f"checkpoint={args.checkpoint} subfolder={args.checkpoint_subfolder} "
        f"version={pipeline.config.runtime.version.value} "
        f"device={device} precision={precision} scenes={len(inputs)}",
        flush=True,
    )

    def validate_scene_batch(scenes) -> None:
        for scene in scenes:
            validation = {}
            if isinstance(pipeline, RenderFormerV2Pipeline):
                validation = pipeline.validate_scene(
                    scene,
                    execution_profile=execution_profile,
                    view_transformer_tf32=args.view_transformer_tf32,
                    use_packed_sequence=args.packed_sequence,
                )
            elif (
                args.padding_length is not None
                and scene.num_triangles > args.padding_length
            ):
                raise ValueError(
                    f"{scene.path} has {scene.num_triangles} triangles, exceeding "
                    f"padding_length={args.padding_length}"
                )
            print(
                f"validated {scene.path}: triangles={scene.num_triangles} "
                f"views={scene.num_views} constraints={validation}",
                flush=True,
            )

    output_dir = None
    records = []
    video_frames = []

    if not args.validate_only:
        output_dir = _output_dir(args.input_path, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def save_result(scene, result):
        assert output_dir is not None
        hdr_batch = result.hdr[0].detach().cpu().to(torch.float32).numpy()
        outputs = []
        for view_index, hdr in enumerate(hdr_batch):
            hdr = hdr.astype(np.float32, copy=False)
            ldr = (_tone_map(hdr, args.tone_mapper) * 255.0).clip(0, 255).astype(np.uint8)
            stem = f"{scene.path.stem}_view_{view_index}"
            exr_path = output_dir / f"{stem}.exr"
            png_path = output_dir / f"{stem}.png"
            write_hdr_exr(exr_path, hdr)
            iio.imwrite(png_path, ldr)
            if args.save_video:
                video_frames.append(ldr)
            outputs.append({"view": view_index, "exr": exr_path.name, "png": png_path.name})
            print(f"saved {exr_path} and {png_path}", flush=True)
        record = {
            "input": _manifest_input_path(scene.path),
            "num_triangles": scene.num_triangles,
            "num_views": len(outputs),
            "input_preprocessing": dict(scene.input_metadata),
            "outputs": outputs,
        }
        if isinstance(pipeline, RenderFormerV2Pipeline):
            record["effective_view_batch_size"] = pipeline.resolve_view_batch_size(
                resolution=args.resolution,
                num_views=scene.num_views,
                first_view_only=args.first_view_only,
                view_batch_size=args.view_batch_size,
            )
        records.append(record)

    def render_scene_batch(scenes) -> None:
        if pipeline.config.runtime.version is CheckpointVersion.V1:
            view_counts = {
                1 if args.first_view_only else scene.num_views for scene in scenes
            }
            if len(view_counts) == 1:
                results = pipeline.render_v1_scenes(
                    scenes,
                    resolution=args.resolution,
                    precision=precision,
                    padding_length=args.padding_length,
                    first_view_only=args.first_view_only,
                )
            else:
                results = [
                    pipeline.render_scene(
                        scene,
                        resolution=args.resolution,
                        precision=precision,
                        first_view_only=args.first_view_only,
                    )
                    for scene in scenes
                ]
        else:
            # V2 env maps and volume counts are intentionally processed per scene.
            results = [
                pipeline.render_scene(
                    scene,
                    resolution=args.resolution,
                    precision=precision,
                    first_view_only=args.first_view_only,
                    view_batch_size=args.view_batch_size,
                    execution_profile=execution_profile,
                    view_transformer_tf32=args.view_transformer_tf32,
                    use_packed_sequence=args.packed_sequence,
                )
                for scene in scenes
            ]
        for scene, result in zip(scenes, results):
            save_result(scene, result)

    with _scene_loader_pool(args.num_workers, args.batch_size, len(inputs)) as pool:
        for start in range(0, len(inputs), args.batch_size):
            window = inputs[start : start + args.batch_size]
            load_arguments = [
                (
                    path,
                    pipeline.config.runtime.version.value,
                    pipeline.config.runtime.image_key,
                    args.allow_legacy_rf1_dtypes,
                )
                for path in window
            ]
            scenes = (
                pool.map(_load_scene_worker, load_arguments)
                if pool is not None
                else [_load_scene_worker(item) for item in load_arguments]
            )
            try:
                if args.validate_only:
                    validate_scene_batch(scenes)
                else:
                    render_scene_batch(scenes)
            finally:
                # Scene tensors can dwarf the model. Drop the completed window
                # before opening the next H5 files instead of retaining an
                # entire manifest in host memory.
                scenes.clear()

    if args.validate_only:
        return 0

    assert output_dir is not None

    if args.save_video:
        video_path = output_dir / "video.mp4"
        iio.imwrite(video_path, np.stack(video_frames), fps=24, quality=9)
        print(f"saved {video_path}", flush=True)

    execution_settings = (
        pipeline.execution_settings(
            execution_profile,
            view_transformer_tf32=args.view_transformer_tf32,
            use_packed_sequence=args.packed_sequence,
        )
        if isinstance(pipeline, RenderFormerV2Pipeline)
        else {
            "view_transformer_tf32": precision in {"fp16", "bf16"},
            "use_packed_sequence": False,
            "missing_environment": "unsupported",
            "missing_volume": "unsupported",
            "cuda_matmul_allow_tf32": (
                device.type == "cuda" and precision in {"fp16", "bf16"}
            ),
            "cudnn_allow_tf32": (
                device.type == "cuda" and precision in {"fp16", "bf16"}
            ),
        }
    )
    manifest = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "checkpoint_subfolder": args.checkpoint_subfolder,
        "revision": revision,
        "model_version": pipeline.config.runtime.version.value,
        "precision": precision,
        "resolution": args.resolution,
        "tone_mapper": args.tone_mapper,
        "batch_size": args.batch_size,
        "view_batch_size": args.view_batch_size,
        "view_batching": {
            "requested": args.view_batch_size,
            "policy": (
                "not_applicable"
                if pipeline.config.runtime.version is CheckpointVersion.V1
                else "explicit"
                if args.view_batch_size is not None
                else "auto_single_view_at_2048_plus"
            ),
            "auto_single_view_min_resolution": (
                RenderFormerV2Pipeline.AUTO_SINGLE_VIEW_MIN_RESOLUTION
                if isinstance(pipeline, RenderFormerV2Pipeline)
                and args.view_batch_size is None
                else None
            ),
        },
        "padding_length": args.padding_length,
        "optimize": args.optimize,
        "execution_profile": execution_profile,
        "execution_profile_selection": execution_profile_selection,
        "execution_settings": execution_settings,
        "fused_kernels": _fused_kernel_status(pipeline),
        "components": getattr(pipeline, "component_info", {}),
        "video": "video.mp4" if args.save_video else None,
        "scenes": records,
    }
    if pipeline.config.runtime.version is CheckpointVersion.V1:
        manifest["allow_legacy_rf1_dtypes"] = args.allow_legacy_rf1_dtypes
    manifest_path = output_dir / "inference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
