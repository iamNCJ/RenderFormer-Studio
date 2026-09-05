"""Metadata-driven Objaverse mesh preparation and collection receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
from importlib import metadata
from importlib.util import find_spec
import json
import math
from multiprocessing import current_process, get_context
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
import uuid


TARGET_FACE_COUNTS = (1188, 3968)
OBJECT_LISTS = {
    "objects-1188.txt": 1188,
    "objects-3968.txt": 3968,
}
SPECIAL_OBJECT_LIST = "objects-special.txt"
PIPELINE_ID = "renderformer-objaverse-watertight-qslim-v1"
# Bump PIPELINE_ID before changing any operation that can alter mesh bytes.
_SAFE_OBJECT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CANONICAL_DISTRIBUTIONS = {
    "bpy": "4.5.10",
    "bpy-helper": "0.0.13",
    "libigl": "2.5.1",
    "numpy": "1.26.4",
    "scikit-image": "0.25.2",
    "scipy": "1.15.3",
    "trimesh": "4.7.1",
}
_BLENDER_IMPORT_PROBE = """
from importlib import metadata
from pathlib import Path

distribution_roots = {
    name: Path(metadata.distribution(name).locate_file("")).resolve()
    for name in ("bpy", "bpy-helper")
}
import bpy_helper.mesh as helper_mesh
import bpy

if tuple(bpy.app.version[:3]) != (4, 5, 10):
    raise RuntimeError(f"unexpected bpy runtime: {bpy.app.version}")
if not callable(getattr(helper_mesh, "uv_unwrap", None)):
    raise RuntimeError("bpy_helper.mesh has no callable uv_unwrap")
for distribution_name, module in (("bpy", bpy), ("bpy-helper", helper_mesh)):
    distribution_root = distribution_roots[distribution_name]
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(distribution_root):
        raise RuntimeError(
            f"{distribution_name} resolved outside its environment distribution: "
            f"{module_path} (expected under {distribution_root})"
        )
"""


@dataclass(frozen=True)
class ObjaverseSource:
    """One explicit source row from the input JSONL catalog."""

    object_id: str
    source_path: Path
    manifest_path: str
    special: bool


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"strict JSON does not permit {constant}")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, _canonical_json(value) + b"\n")


def load_objaverse_sources(
    input_manifest: str | Path,
    *,
    allow_empty_special: bool = False,
) -> tuple[ObjaverseSource, ...]:
    """Load a strict JSONL catalog without discovering files on disk."""

    manifest_path = Path(input_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    sources: list[ObjaverseSource] = []
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            value = json.loads(line, parse_constant=_reject_json_constant)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{manifest_path}:{line_number}: each row must be a JSON object"
                )
            unknown = set(value) - {"id", "path", "special"}
            missing = {"id", "path"} - set(value)
            if missing or unknown:
                raise ValueError(
                    f"{manifest_path}:{line_number}: expected id, path, and "
                    f"optional special; missing={sorted(missing)}, "
                    f"unknown={sorted(unknown)}"
                )

            object_id = value["id"]
            declared_path = value["path"]
            special = value.get("special", False)
            if not isinstance(object_id, str) or not _SAFE_OBJECT_ID.fullmatch(object_id):
                raise ValueError(
                    f"{manifest_path}:{line_number}: id must match "
                    "[A-Za-z0-9][A-Za-z0-9._-]*"
                )
            if object_id in seen_ids:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate id {object_id!r}"
                )
            if not isinstance(declared_path, str) or not declared_path:
                raise ValueError(
                    f"{manifest_path}:{line_number}: path must be a non-empty string"
                )
            if not isinstance(special, bool):
                raise ValueError(
                    f"{manifest_path}:{line_number}: special must be true or false"
                )

            source_path = Path(declared_path).expanduser()
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path}:{line_number}: source mesh is missing: {source_path}"
                )
            seen_ids.add(object_id)
            sources.append(
                ObjaverseSource(
                    object_id=object_id,
                    source_path=source_path,
                    manifest_path=declared_path,
                    special=special,
                )
            )

    if not sources:
        raise ValueError(f"Objaverse source manifest is empty: {manifest_path}")
    if not allow_empty_special and not any(source.special for source in sources):
        raise ValueError(
            "at least one source row must set special=true to create "
            f"{SPECIAL_OBJECT_LIST}; pass --allow-empty-special only when stage 6 "
            "will not be generated"
        )
    return tuple(sources)


def _require_dependencies(watertight_backend: str) -> None:
    required_modules = {
        "igl": "libigl",
        "bpy": "bpy",
        "bpy_helper": "bpy-helper",
    }
    if watertight_backend == "voxel":
        required_modules["skimage"] = "scikit-image"
    elif watertight_backend == "cuda-sdf":
        required_modules["torchcumesh2sdf"] = "torchcumesh2sdf"
        required_modules["diso"] = "diso"
    else:
        raise ValueError(f"unsupported watertight backend: {watertight_backend}")

    missing = [
        distribution
        for module, distribution in required_modules.items()
        if find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Objaverse preparation is missing required packages: " + ", ".join(missing)
        )

    mismatches = []
    for distribution, expected in _CANONICAL_DISTRIBUTIONS.items():
        if distribution == "scikit-image" and watertight_backend != "voxel":
            continue
        try:
            actual = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed (expected {expected})")
            continue
        if actual != expected:
            mismatches.append(f"{distribution}=={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError(
            "Objaverse preparation requires the canonical datagen versions: "
            + "; ".join(mismatches)
        )

    for module in sorted(set(required_modules) - {"bpy", "bpy_helper"}):
        import_module(module)

    probe = subprocess.run(
        [sys.executable, "-I", "-c", _BLENDER_IMPORT_PROBE],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "the isolated PyPI bpy 4.5.10 / bpy-helper 0.0.13 import probe "
            f"failed:\n{probe.stderr.strip()}"
        )


def _dependency_versions(watertight_backend: str) -> dict[str, str | None]:
    names = ["numpy", "scipy", "trimesh", "libigl", "bpy", "bpy-helper"]
    if watertight_backend == "voxel":
        names.append("scikit-image")
    else:
        names.extend(("torch", "torchcumesh2sdf", "diso"))

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _has_complete_obj_uv(path: Path) -> bool:
    texture_coordinate_count = 0
    has_face = False
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            tokens = line.split()
            if not tokens or tokens[0].startswith("#"):
                continue
            if tokens[0] == "vt":
                if len(tokens) < 3:
                    return False
                try:
                    uv = (float(tokens[1]), float(tokens[2]))
                except ValueError:
                    return False
                if not all(math.isfinite(component) for component in uv):
                    return False
                texture_coordinate_count += 1
            elif tokens[0] == "f":
                has_face = True
                corners = tokens[1:]
                if len(corners) < 3:
                    return False
                for corner in corners:
                    fields = corner.split("/")
                    if len(fields) < 2 or not fields[1]:
                        return False
                    try:
                        texture_index = int(fields[1])
                    except ValueError:
                        return False
                    if texture_index == 0:
                        return False
                    if not (
                        -texture_coordinate_count <= texture_index <= -1
                        or 1 <= texture_index <= texture_coordinate_count
                    ):
                        return False

    return texture_coordinate_count > 0 and has_face


def _topology_is_watertight(mesh: Any) -> bool:
    if mesh.is_watertight:
        return True
    merged = mesh.copy()
    # OBJ exporters may split one geometric vertex at UV seams or hard normals.
    # Ignore those attributes only for this topology check; the delivered OBJ
    # keeps its original indexed UVs and normals.
    merged.merge_vertices(merge_tex=True, merge_norm=True)
    return bool(merged.is_watertight)


def _validate_prepared_mesh(path: Path, target_faces: int) -> tuple[int, bool]:
    from renderformer.data.geometry import load_mesh

    mesh = load_mesh(path)
    face_count = int(len(mesh.faces))
    if face_count > target_faces:
        raise ValueError(
            f"prepared mesh exceeds its {target_faces}-face cap: "
            f"{path} has {face_count} faces"
        )
    watertight = _topology_is_watertight(mesh)
    if not watertight:
        raise ValueError(f"prepared mesh is not watertight: {path}")
    if not _has_complete_obj_uv(path):
        raise ValueError(f"prepared mesh has no indexed OBJ UV coordinates: {path}")
    return face_count, watertight


def _object_record_path(output_dir: Path, object_id: str) -> Path:
    return output_dir / "records" / f"{object_id}.json"


def _embedded_record_path(output_dir: Path, object_id: str) -> Path:
    return output_dir / "objects" / object_id / ".renderformer-record.json"


def _validate_completed_record(
    source: ObjaverseSource,
    *,
    output_dir: Path,
    plan_sha256: str,
) -> dict[str, Any]:
    record_path = _object_record_path(output_dir, source.object_id)
    embedded_path = _embedded_record_path(output_dir, source.object_id)
    if record_path.is_file():
        selected_path = record_path
    elif embedded_path.is_file():
        selected_path = embedded_path
    else:
        raise FileNotFoundError(
            f"no completion record for prepared object {source.object_id!r}"
        )
    value = json.loads(selected_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("id") != source.object_id:
        raise ValueError(f"invalid completed-object record: {selected_path}")
    if value.get("plan_sha256") != plan_sha256:
        raise ValueError(
            f"completed-object record belongs to another plan: {selected_path}"
        )
    if value.get("source", {}).get("manifest_path") != source.manifest_path:
        raise ValueError(f"source path changed for completed object: {source.object_id}")
    if value.get("source", {}).get("sha256") != _sha256(source.source_path):
        raise ValueError(f"source bytes changed for completed object: {source.object_id}")
    if value.get("special") is not source.special:
        raise ValueError(f"special flag changed for completed object: {source.object_id}")

    outputs = value.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"completed-object record has no outputs: {selected_path}")
    for target_faces in TARGET_FACE_COUNTS:
        key = str(target_faces)
        item = outputs.get(key)
        if not isinstance(item, dict):
            raise ValueError(
                f"completed-object record has no {key} output: {selected_path}"
            )
        relative_path = Path(item.get("path", ""))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe output path in completed-object record: {selected_path}"
            )
        expected_path = (
            Path("objects")
            / source.object_id
            / f"mesh_{target_faces}.obj"
        )
        if relative_path != expected_path:
            raise ValueError(
                f"unexpected output path in completed-object record: {selected_path}"
            )
        output_path = output_dir / relative_path
        if not output_path.is_file():
            raise FileNotFoundError(output_path)
        if item.get("sha256") != _sha256(output_path):
            raise ValueError(f"prepared output bytes changed: {output_path}")
        face_count, watertight = _validate_prepared_mesh(output_path, target_faces)
        if item.get("face_count") != face_count or item.get("watertight") is not watertight:
            raise ValueError(f"prepared output metadata changed: {output_path}")
    if selected_path == embedded_path:
        _atomic_write_json(record_path, value)
    return value


def _prepare_source(payload: dict[str, Any]) -> dict[str, Any]:
    from renderformer.data.geometry import (
        export_mesh,
        load_mesh,
        sdf_remesh_cuda,
        simplify_mesh,
        unwrap_uv,
        voxel_remesh,
    )

    source = ObjaverseSource(
        object_id=payload["id"],
        source_path=Path(payload["source_path"]),
        manifest_path=payload["manifest_path"],
        special=payload["special"],
    )
    output_dir = Path(payload["output_dir"])
    plan_sha256 = payload["plan_sha256"]
    record_path = _object_record_path(output_dir, source.object_id)
    final_object_dir = output_dir / "objects" / source.object_id
    if record_path.is_file() or final_object_dir.is_dir():
        worker = current_process().name
        print(f"[{worker}] {source.object_id}: validating completed record", flush=True)
        return _validate_completed_record(
            source,
            output_dir=output_dir,
            plan_sha256=plan_sha256,
        )

    if final_object_dir.exists():
        raise FileExistsError(
            f"prepared object exists without a valid record: {final_object_dir}"
        )

    worker = current_process().name
    print(f"[{worker}] {source.object_id}: watertight reconstruction", flush=True)
    source_sha256 = _sha256(source.source_path)
    source_mesh = load_mesh(source.source_path)
    if payload["watertight_backend"] == "voxel":
        watertight_mesh = voxel_remesh(
            source_mesh,
            resolution=payload["voxel_resolution"],
            radius=payload["normalize_radius"],
        )
    else:
        watertight_mesh = sdf_remesh_cuda(
            source_mesh,
            resolution=payload["sdf_resolution"],
            scale=payload["sdf_scale"],
            device=payload["device"],
        )
    if not watertight_mesh.is_watertight:
        raise ValueError(
            f"watertight reconstruction failed for source {source.object_id!r}"
        )

    staging_dir = output_dir / "objects" / (
        f".{source.object_id}.{uuid.uuid4().hex}.tmp"
    )
    staging_dir.mkdir(parents=True)
    outputs: dict[str, dict[str, Any]] = {}
    try:
        for target_faces in TARGET_FACE_COUNTS:
            print(f"[{worker}] {source.object_id}: QSlim {target_faces}", flush=True)
            simplified = simplify_mesh(watertight_mesh, target_faces, backend="igl")
            if not simplified.is_watertight:
                raise ValueError(
                    f"QSlim broke watertightness for {source.object_id!r} at {target_faces}"
                )
            pre_uv_path = staging_dir / f"mesh_{target_faces}.pre-uv.obj"
            output_path = staging_dir / f"mesh_{target_faces}.obj"
            export_mesh(simplified, pre_uv_path)
            unwrap_uv(pre_uv_path, output_path, method="cube_project", normalize=False)
            pre_uv_path.unlink()
            face_count, watertight = _validate_prepared_mesh(output_path, target_faces)
            relative_path = Path("objects") / source.object_id / output_path.name
            outputs[str(target_faces)] = {
                "path": relative_path.as_posix(),
                "face_count": face_count,
                "watertight": watertight,
                "sha256": _sha256(output_path),
            }

        if _sha256(source.source_path) != source_sha256:
            raise RuntimeError(
                f"source mesh changed during processing: {source.source_path}"
            )
        record = {
            "schema_version": 1,
            "id": source.object_id,
            "special": source.special,
            "plan_sha256": plan_sha256,
            "source": {
                "manifest_path": source.manifest_path,
                "sha256": source_sha256,
            },
            "outputs": outputs,
        }
        _atomic_write_json(staging_dir / ".renderformer-record.json", record)
        final_object_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(final_object_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    _atomic_write_json(record_path, record)
    print(f"[{worker}] {source.object_id}: complete", flush=True)
    return record


def _write_collection_outputs(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    plan_sha256: str,
) -> Path:
    def list_entry(record: dict[str, Any], target_faces: int) -> str:
        collection_path = Path(record["outputs"][str(target_faces)]["path"])
        try:
            return collection_path.relative_to("objects").as_posix()
        except ValueError as exc:
            raise ValueError(
                f"prepared output is outside the object root: {collection_path}"
            ) from exc

    records.sort(key=lambda item: item["id"])
    records_content = b"".join(_canonical_json(record) + b"\n" for record in records)
    records_path = output_dir / "processed-objects.jsonl"
    _atomic_write(records_path, records_content)

    lists_dir = output_dir / "lists"
    list_metadata: dict[str, dict[str, Any]] = {}
    for list_name, target_faces in OBJECT_LISTS.items():
        entries = [list_entry(record, target_faces) for record in records]
        content = "".join(f"{entry}\n" for entry in entries).encode("utf-8")
        list_path = lists_dir / list_name
        _atomic_write(list_path, content)
        list_metadata[list_name] = {
            "target_face_count": target_faces,
            "count": len(entries),
            "sha256": _sha256(list_path),
        }

    special_entries = [
        list_entry(record, 3968) for record in records if record["special"]
    ]
    special_content = "".join(f"{entry}\n" for entry in special_entries).encode(
        "utf-8"
    )
    special_path = lists_dir / SPECIAL_OBJECT_LIST
    _atomic_write(special_path, special_content)
    list_metadata[SPECIAL_OBJECT_LIST] = {
        "target_face_count": 3968,
        "selection": "input rows with special=true",
        "count": len(special_entries),
        "sha256": _sha256(special_path),
    }

    receipt = {
        "schema_version": 1,
        "pipeline": PIPELINE_ID,
        "plan_sha256": plan_sha256,
        "source_count": len(records),
        "special_source_count": len(special_entries),
        "processed_objects": {
            "path": records_path.name,
            "sha256": _sha256(records_path),
        },
        "lists": list_metadata,
        "parameters": plan["parameters"],
        "dependencies": plan["dependencies"],
        "historical_equivalence": False,
        "compatibility_note": (
            "The face caps and UV-mesh list contract are recovered, but the old "
            "random remesh variants, split seed, and special-object selection are "
            "not. Special objects come only from source rows marked special=true."
        ),
    }
    receipt_path = output_dir / "objaverse-receipt.json"
    _atomic_write_json(receipt_path, receipt)
    return receipt_path


def prepare_objaverse_collection(
    input_manifest: str | Path,
    output_dir: str | Path,
    *,
    num_workers: int = 1,
    watertight_backend: str = "voxel",
    voxel_resolution: int = 256,
    normalize_radius: float = 0.45,
    sdf_resolution: int = 256,
    sdf_scale: float = 0.8,
    device: str = "cuda",
    allow_empty_special: bool = False,
) -> Path:
    """Prepare an explicit Objaverse catalog and emit its complete data receipt."""

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if voxel_resolution < 32 or sdf_resolution < 32:
        raise ValueError("voxel and SDF resolutions must be at least 32")
    if not 0 < sdf_scale <= 1:
        raise ValueError("sdf_scale must be in (0, 1]")
    if normalize_radius <= 0:
        raise ValueError("normalize_radius must be positive")

    _require_dependencies(watertight_backend)
    manifest_path = Path(input_manifest).expanduser().resolve()
    sources = load_objaverse_sources(
        manifest_path,
        allow_empty_special=allow_empty_special,
    )
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if manifest_path == resolved_output_dir or manifest_path.is_relative_to(
        resolved_output_dir
    ):
        raise ValueError("input manifest cannot live inside the output directory")
    for source in sources:
        if source.source_path.is_relative_to(resolved_output_dir):
            raise ValueError(
                f"source mesh cannot live inside the output directory: {source.source_path}"
            )

    parameters = {
        "target_face_counts": list(TARGET_FACE_COUNTS),
        "watertight_backend": watertight_backend,
        "voxel_resolution": voxel_resolution,
        "normalize_radius": normalize_radius,
        "sdf_resolution": sdf_resolution,
        "sdf_scale": sdf_scale,
        "device": device,
        "simplify_backend": "igl.qslim",
        "uv_method": "cube_project",
        "allow_empty_special": allow_empty_special,
    }
    plan = {
        "schema_version": 1,
        "pipeline": PIPELINE_ID,
        "input_manifest": {
            "sha256": _sha256(manifest_path),
            "source_count": len(sources),
            "special_source_count": sum(source.special for source in sources),
        },
        "parameters": parameters,
        "dependencies": _dependency_versions(watertight_backend),
    }
    plan_bytes = _canonical_json(plan) + b"\n"
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = resolved_output_dir / "objaverse-plan.json"
    if plan_path.exists():
        if plan_path.read_bytes() != plan_bytes:
            raise ValueError(
                f"existing output belongs to another Objaverse plan: {plan_path}"
            )
    else:
        if resolved_output_dir.exists() and any(resolved_output_dir.iterdir()):
            raise FileExistsError(
                f"non-empty output has no Objaverse plan: {resolved_output_dir}"
            )
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(plan_path, plan_bytes)

    payloads = [
        {
            "id": source.object_id,
            "source_path": str(source.source_path),
            "manifest_path": source.manifest_path,
            "special": source.special,
            "output_dir": str(resolved_output_dir),
            "plan_sha256": plan_sha256,
            "watertight_backend": watertight_backend,
            "voxel_resolution": voxel_resolution,
            "normalize_radius": normalize_radius,
            "sdf_resolution": sdf_resolution,
            "sdf_scale": sdf_scale,
            "device": device,
        }
        for source in sources
    ]
    if num_workers == 1:
        records = [_prepare_source(payload) for payload in payloads]
    else:
        with get_context("spawn").Pool(processes=num_workers) as pool:
            records = list(pool.imap_unordered(_prepare_source, payloads))

    return _write_collection_outputs(
        resolved_output_dir,
        records=records,
        plan=plan,
        plan_sha256=plan_sha256,
    )
