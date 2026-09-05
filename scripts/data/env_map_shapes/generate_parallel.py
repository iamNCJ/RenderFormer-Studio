"""Generate PyPI-bpy shapes in isolated Python subprocesses and merge metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time


def stream_output(process: subprocess.Popen[str], worker_id: int) -> None:
    """Forward one subprocess stream with a stable worker prefix."""
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[W{worker_id}] {line.rstrip()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shapes", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-types", default="all")
    parser.add_argument("--subdivisions", type=int, default=2)
    parser.add_argument("--mesh-resolution", type=int, default=32)
    return parser


def _merge_worker_metadata(worker_dirs: list[Path], output_dir: Path) -> list[dict]:
    records = []
    for worker_dir in worker_dirs:
        metadata_path = worker_dir / "all_shapes.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        with metadata_path.open("r", encoding="utf-8") as handle:
            worker_records = json.load(handle)
        if not isinstance(worker_records, list):
            raise ValueError(f"{metadata_path} must contain a list")
        for record in worker_records:
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError(f"invalid shape record in {metadata_path}")
            source_dir = worker_dir / record["id"]
            destination_dir = output_dir / record["id"]
            if destination_dir.exists():
                raise FileExistsError(destination_dir)
            shutil.move(str(source_dir), destination_dir)
            merged_record = dict(record)
            merged_record["index"] = len(records)
            merged_record["obj_path"] = str(destination_dir / "original.obj")
            records.append(merged_record)
    return records


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shapes < 1:
        raise ValueError("num_shapes must be at least 1")
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "all_shapes.json"
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {metadata_path}")

    generator = Path(__file__).with_name("create_convex_shapes.py")
    worker_count = min(args.num_workers, args.num_shapes)
    base_size, remainder = divmod(args.num_shapes, worker_count)
    started_at = time.monotonic()

    with TemporaryDirectory(prefix=".shape-parts-", dir=output_dir) as temporary:
        temporary_root = Path(temporary)
        processes = []
        streams = []
        worker_dirs = []
        for worker_id in range(worker_count):
            worker_size = base_size + int(worker_id < remainder)
            worker_dir = temporary_root / f"worker-{worker_id}"
            worker_dirs.append(worker_dir)
            command = [
                sys.executable,
                "-I",
                str(generator),
                "--output_dir",
                str(worker_dir),
                "--num_shapes",
                str(worker_size),
                "--seed",
                str(args.seed + worker_id * 10_000),
                "--shape_types",
                args.shape_types,
                "--subdivisions",
                str(args.subdivisions),
                "--mesh_resolution",
                str(args.mesh_resolution),
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append((worker_id, process))
            stream = threading.Thread(
                target=stream_output,
                args=(process, worker_id),
                daemon=True,
            )
            stream.start()
            streams.append(stream)

        failures = []
        for worker_id, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((worker_id, return_code))
        for stream in streams:
            stream.join()
        if failures:
            raise RuntimeError(f"shape generation workers failed: {failures}")

        records = _merge_worker_metadata(worker_dirs, output_dir)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
    elapsed = time.monotonic() - started_at
    print(
        f"generated {len(records)} shapes in {elapsed:.1f}s; metadata: {metadata_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
