"""Generate, remesh, and UV-unwrap local environment-reflection shapes."""

from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import random
import shutil
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from renderformer.data.geometry import load_mesh, remesh_file, unwrap_uv


def run_bpy_generation(
    output_dir: Path,
    num_shapes: int,
    seed: int,
    shape_types: str = "all",
) -> None:
    """Run the sibling PyPI-bpy generator with the active Python executable."""
    script_path = Path(__file__).with_name("create_convex_shapes.py")
    command = [
        sys.executable,
        "-I",
        str(script_path),
        "--output_dir",
        str(output_dir),
        "--num_shapes",
        str(num_shapes),
        "--seed",
        str(seed),
        "--shape_types",
        shape_types,
    ]
    subprocess.run(command, check=True)


def load_shape_directories(input_dir: Path) -> list[Path]:
    """Resolve generated shapes from ``all_shapes.json`` without tree walking."""
    metadata_path = input_dir / "all_shapes.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} is required; generate shapes first or provide its metadata"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{metadata_path} must contain a non-empty list")

    shape_directories = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise ValueError(f"invalid shape record {index} in {metadata_path}")
        shape_dir = input_dir / record["id"]
        if not (shape_dir / "original.obj").is_file():
            raise FileNotFoundError(shape_dir / "original.obj")
        shape_directories.append(shape_dir)
    return shape_directories


def process_single_shape(
    shape_dir: Path,
    *,
    output_dir: Path,
    remesh_backend: str,
    remesh_size: int,
    remesh_scale: float,
    target_faces: int,
    target_faces_variation: float,
    uv_method: str,
    seed: int,
) -> dict[str, str | int]:
    """Prepare one shape using the shared local geometry utilities."""
    shape_id = shape_dir.name
    original_obj = shape_dir / "original.obj"
    output_shape_dir = output_dir / shape_id
    output_shape_dir.mkdir(parents=True, exist_ok=True)

    original_copy = output_shape_dir / "original.obj"
    remeshed_path = output_shape_dir / "remeshed.obj"
    final_path = output_shape_dir / "final.obj"
    shutil.copyfile(original_obj, original_copy)

    metadata_src = shape_dir / "metadata.json"
    if metadata_src.is_file():
        shutil.copyfile(metadata_src, output_shape_dir / "metadata.json")

    generator = random.Random(f"{seed}:{shape_id}")
    variation = generator.uniform(
        1.0 - target_faces_variation,
        1.0 + target_faces_variation,
    )
    randomized_target_faces = max(500, int(target_faces * variation))
    remesh_file(
        original_obj,
        remeshed_path,
        target_faces=randomized_target_faces,
        backend=remesh_backend,
        sdf_resolution=remesh_size,
        sdf_scale=remesh_scale,
    )
    unwrap_uv(remeshed_path, final_path, method=uv_method)
    return {
        "id": shape_id,
        "original": str(original_copy),
        "remeshed": str(remeshed_path),
        "final": str(final_path),
        "face_count": len(load_mesh(final_path).faces),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate source meshes first with PyPI bpy in the active Python environment",
    )
    parser.add_argument("--num-shapes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape-types", default="all")
    parser.add_argument(
        "--remesh-backend",
        choices=("simplify", "cuda-sdf"),
        default="simplify",
        help="simplify is local/default; cuda-sdf requires torchcumesh2sdf and diso",
    )
    parser.add_argument("--remesh-size", type=int, default=256)
    parser.add_argument("--remesh-scale", type=float, default=0.8)
    parser.add_argument("--target-faces", type=int, default=4000)
    parser.add_argument("--target-faces-variation", type=float, default=0.3)
    parser.add_argument(
        "--uv-method",
        choices=("cube_project", "smart_project", "sphere_project"),
        default="cube_project",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if args.target_faces_variation < 0 or args.target_faces_variation >= 1:
        raise ValueError("target_faces_variation must be in [0, 1)")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if args.generate:
        input_dir.mkdir(parents=True, exist_ok=True)
        run_bpy_generation(
            input_dir,
            args.num_shapes,
            args.seed,
            args.shape_types,
        )
    shape_directories = load_shape_directories(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process = partial(
        process_single_shape,
        output_dir=output_dir,
        remesh_backend=args.remesh_backend,
        remesh_size=args.remesh_size,
        remesh_scale=args.remesh_scale,
        target_faces=args.target_faces,
        target_faces_variation=args.target_faces_variation,
        uv_method=args.uv_method,
        seed=args.seed,
    )
    if args.num_workers == 1:
        results = []
        for index, shape_dir in enumerate(shape_directories):
            print(
                f"[W0] {index + 1}/{len(shape_directories)}: {shape_dir.name}",
                flush=True,
            )
            results.append(process(shape_dir))
    else:
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(process, shape_directories)

    summary_path = output_dir / "processed_shapes.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"processed {len(results)} shapes; metadata: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
