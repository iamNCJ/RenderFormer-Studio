#!/usr/bin/env python3
"""Local remesh and UV-unwrapping CLI.

For batches, pass a JSON manifest containing ``{"jobs": [{"input": ...,
"output": ...}]}``. Each job runs in a separate process; no directory crawling,
cloud queue, or internal mount convention is involved.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool, current_process
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from renderformer.data.geometry import remesh_file, unwrap_uv


def _load_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.manifest is None:
        if args.input is None or args.output is None:
            raise ValueError("input and output are required without --manifest")
        return [{"input": args.input, "output": args.output}]
    if args.input is not None or args.output is not None:
        raise ValueError("positional input/output cannot be combined with --manifest")

    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("manifest must contain a non-empty 'jobs' list")
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or set(job) != {"input", "output"}:
            raise ValueError(f"manifest job {index} must contain only input and output")
        if not all(isinstance(job[key], str) and job[key] for key in ("input", "output")):
            raise ValueError(f"manifest job {index} paths must be non-empty strings")
    return jobs


def _process_job(payload: tuple[int, dict[str, str], dict[str, Any]]) -> str:
    index, job, options = payload
    worker = current_process().name
    print(f"[{worker}] job {index}: {job['input']}", flush=True)
    if options["operation"] == "remesh":
        result = remesh_file(
            job["input"],
            job["output"],
            target_faces=options["target_faces"],
            backend=options["backend"],
            simplify_backend=options["simplify_backend"],
            normalize_radius=options["normalize_radius"],
            voxel_resolution=options["voxel_resolution"],
            sdf_resolution=options["sdf_resolution"],
            sdf_scale=options["sdf_scale"],
            device=options["device"],
        )
    else:
        result = unwrap_uv(
            job["input"],
            job["output"],
            method=options["uv_method"],
            normalize=options["uv_normalize"],
        )
    print(f"[{worker}] job {index}: wrote {result}", flush=True)
    return str(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("remesh", "unwrap-uv"))
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--manifest")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument(
        "--backend",
        choices=("simplify", "voxel", "cuda-sdf"),
        default="simplify",
    )
    parser.add_argument("--simplify-backend", choices=("auto", "igl", "trimesh"), default="auto")
    parser.add_argument("--normalize-radius", type=float, default=0.45)
    parser.add_argument("--voxel-resolution", type=int, default=256)
    parser.add_argument("--sdf-resolution", type=int, default=256)
    parser.add_argument("--sdf-scale", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--uv-method", default="cube_project")
    parser.add_argument("--uv-normalize", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    jobs = _load_jobs(args)
    options = vars(args).copy()
    payloads = [(index, job, options) for index, job in enumerate(jobs)]
    if args.num_workers == 1:
        for payload in payloads:
            _process_job(payload)
    else:
        with Pool(processes=args.num_workers) as pool:
            pool.map(_process_job, payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
