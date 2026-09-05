#!/usr/bin/env python3
"""Postprocess one or more V2 H5 files with explicit job metadata."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, current_process
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from renderformer.data.textures.postprocess import (
    DEFAULT_DROP_KEYS,
    postprocess_v2_h5,
    postprocess_v2_h5_batch,
)
from renderformer.data.textures.encoders import TextureEncoder, build_texture_encoder


LEARNED_ENCODERS = {"qwen", "qwen_vae", "qwen_texture_vae", "qwen-vae", "dcae", "dc_ae", "dc-ae"}
_CPU_ENCODER: TextureEncoder | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Encode V2 H5 textures and normalize env-map/volume fields. "
            "Batch mode consumes a JSON manifest instead of scanning directories."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Input H5 for a single job")
    source.add_argument(
        "--manifest",
        type=Path,
        help='JSON file shaped as {"jobs": [{"input": "...", "output": "..."}]}',
    )
    parser.add_argument("--output", type=Path, help="Output H5 for --input mode")
    parser.add_argument(
        "--texture-encoder",
        default="qwen_vae",
        choices=("qwen_vae", "dc_ae", "raw"),
    )
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--triangle-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--drop-key",
        action="append",
        default=None,
        help="Dataset key to omit; repeat as needed (defaults to rendered auxiliary passes)",
    )
    return parser


def _resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def load_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.input is not None:
        if args.output is None:
            raise ValueError("--output is required with --input")
        jobs = [(args.input, args.output)]
    else:
        if args.output is not None:
            raise ValueError("--output is only valid with --input")
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_jobs = manifest.get("jobs")
        if not isinstance(raw_jobs, list) or not raw_jobs:
            raise ValueError("Manifest must contain a non-empty 'jobs' list")
        jobs = []
        for index, job in enumerate(raw_jobs):
            if not isinstance(job, dict) or set(job) != {"input", "output"}:
                raise ValueError(
                    f"jobs[{index}] must contain exactly 'input' and 'output'"
                )
            jobs.append(
                (
                    _resolve_manifest_path(str(job["input"]), manifest_path.parent),
                    _resolve_manifest_path(str(job["output"]), manifest_path.parent),
                )
            )

    if len({dst.resolve() for _, dst in jobs}) != len(jobs):
        raise ValueError("Every postprocess job must have a unique output path")
    for src, _ in jobs:
        if not src.is_file():
            raise FileNotFoundError(f"Input H5 file does not exist: {src}")
    return jobs


def build_encoder_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "type": args.texture_encoder,
        "triangle_batch_size": args.triangle_batch_size,
        "torch_dtype": args.torch_dtype,
    }
    if args.model_id is not None:
        config["model_id"] = args.model_id
    if args.revision is not None:
        config["revision"] = args.revision
    if args.cache_dir is not None:
        config["cache_dir"] = args.cache_dir
    config["local_files_only"] = args.local_files_only
    if args.device is not None:
        config["device"] = args.device
    return config


def _initialize_cpu_worker(encoder_config: dict[str, Any]) -> None:
    global _CPU_ENCODER
    _CPU_ENCODER = build_texture_encoder(encoder_config)
    print(f"[{current_process().name}] encoder ready: {_CPU_ENCODER.name}", flush=True)


def _process_cpu_job(
    task: tuple[Path, Path, dict[str, Any], bool, bool, tuple[str, ...]],
) -> str:
    src, dst, encoder_config, keep_raw, skip_existing, drop_keys = task
    if _CPU_ENCODER is None:
        raise RuntimeError("Texture encoder worker was not initialized")
    worker = current_process().name
    if skip_existing and dst.exists():
        print(f"[{worker}] skip {dst}", flush=True)
        return str(dst)
    print(f"[{worker}] {src} -> {dst}", flush=True)
    postprocess_v2_h5(
        src,
        dst,
        encoder_config,
        keep_raw,
        texture_encoder=_CPU_ENCODER,
        drop_keys=drop_keys,
    )
    return str(dst)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("--num-workers must be at least 1")

    jobs = load_jobs(args)
    encoder_config = build_encoder_config(args)
    drop_keys = tuple(args.drop_key) if args.drop_key is not None else DEFAULT_DROP_KEYS

    if str(args.texture_encoder).lower() in LEARNED_ENCODERS and args.num_workers != 1:
        raise ValueError(
            "Learned texture encoders require --num-workers 1 so one process owns "
            "one preloaded GPU model"
        )

    if args.num_workers == 1:
        postprocess_v2_h5_batch(
            jobs,
            encoder_config,
            keep_raw=args.keep_raw,
            skip_existing=args.skip_existing,
            drop_keys=drop_keys,
        )
    else:
        tasks = [
            (src, dst, encoder_config, args.keep_raw, args.skip_existing, drop_keys)
            for src, dst in jobs
        ]
        with Pool(
            processes=args.num_workers,
            initializer=_initialize_cpu_worker,
            initargs=(encoder_config,),
        ) as pool:
            pool.map(_process_cpu_job, tasks)

    print(f"Processed {len(jobs)} H5 job(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
