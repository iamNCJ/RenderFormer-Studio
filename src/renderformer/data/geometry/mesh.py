"""Reusable, storage-agnostic mesh preparation operations."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh

SimplifyBackend = Literal["auto", "igl", "trimesh"]
RemeshBackend = Literal["simplify", "voxel", "cuda-sdf"]


def _validate_mesh(mesh: trimesh.Trimesh, *, source: str) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{source} did not resolve to a single triangle mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"{source} contains no triangles")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{source} contains non-finite vertices")
    return mesh


def load_mesh(path: str | Path, *, process: bool = False) -> trimesh.Trimesh:
    """Load all geometry in *path* as one triangle mesh."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    mesh = trimesh.load_mesh(input_path, force="mesh", process=process)
    return _validate_mesh(mesh, source=str(input_path))


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Validate and export a mesh, creating only its output directory."""
    _validate_mesh(mesh, source="mesh")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    return output_path


def normalize_mesh(mesh: trimesh.Trimesh, *, radius: float = 0.45) -> trimesh.Trimesh:
    """Return a centered copy whose furthest vertex has the requested radius."""
    _validate_mesh(mesh, source="mesh")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive finite number")

    result = mesh.copy()
    center = result.bounds.mean(axis=0)
    vertices = np.asarray(result.vertices, dtype=np.float64) - center
    source_radius = float(np.linalg.norm(vertices, axis=1).max())
    if source_radius <= np.finfo(np.float64).eps:
        raise ValueError("mesh has zero spatial extent")
    result.vertices = vertices * (radius / source_radius)
    return result


def simplify_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
    *,
    backend: SimplifyBackend = "auto",
) -> trimesh.Trimesh:
    """Reduce face count using libigl or trimesh's quadric backend.

    ``igl`` and trimesh's ``fast-simplification`` backend are optional. With
    ``backend='auto'``, libigl is preferred when installed; otherwise trimesh
    is used. Missing backend dependencies fail with their original actionable
    import error instead of silently changing the output.
    """
    _validate_mesh(mesh, source="mesh")
    if target_faces < 4:
        raise ValueError("target_faces must be at least 4")
    if backend not in {"auto", "igl", "trimesh"}:
        raise ValueError(f"unsupported simplification backend: {backend}")
    if len(mesh.faces) <= target_faces:
        return mesh.copy()

    resolved_backend = backend
    if resolved_backend == "auto":
        resolved_backend = "igl" if find_spec("igl") is not None else "trimesh"

    if resolved_backend == "igl":
        import igl

        result = igl.qslim(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            int(target_faces),
        )
        simplified = trimesh.Trimesh(
            vertices=result[1],
            faces=result[2],
            process=False,
        )
    else:
        simplified = mesh.simplify_quadric_decimation(face_count=int(target_faces))

    _validate_mesh(simplified, source="simplified mesh")
    simplified.fix_normals()
    return simplified


def slice_mesh(
    mesh: trimesh.Trimesh,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    cap: bool = False,
) -> trimesh.Trimesh:
    """Return the positive-normal half of a mesh cut by a plane."""
    _validate_mesh(mesh, source="mesh")
    origin = np.asarray(plane_origin, dtype=np.float64)
    normal = np.asarray(plane_normal, dtype=np.float64)
    if origin.shape != (3,) or normal.shape != (3,):
        raise ValueError("plane_origin and plane_normal must each have shape (3,)")
    normal_length = float(np.linalg.norm(normal))
    if not np.isfinite(normal_length) or normal_length <= np.finfo(np.float64).eps:
        raise ValueError("plane_normal must be non-zero and finite")
    result = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=normal / normal_length,
        plane_origin=origin,
        cap=cap,
    )
    return _validate_mesh(result, source="sliced mesh")


def legacy_pymeshlab_remesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 3568,
) -> trimesh.Trimesh:
    """Run the exact optional pymeshlab sequence used by RF1 scene composition.

    ``pymeshlab`` is imported only when a legacy scene explicitly requests
    remeshing. A missing optional dependency therefore fails at the requested
    operation without making ordinary RF1 composition depend on pymeshlab.
    """
    _validate_mesh(mesh, source="mesh")
    import pymeshlab

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices),
            face_matrix=np.asarray(mesh.faces),
        )
    )
    mesh_set.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PercentageValue(0.5),
        featuredeg=30,
        adaptive=False,
    )
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        qualitythr=1.0,
    )
    processed = mesh_set.current_mesh()
    return _validate_mesh(
        trimesh.Trimesh(
            vertices=processed.vertex_matrix(),
            faces=processed.face_matrix(),
            process=False,
        ),
        source="legacy pymeshlab result",
    )


def voxel_remesh(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 256,
    radius: float = 0.45,
) -> trimesh.Trimesh:
    """Create a deterministic watertight surface through voxel reconstruction.

    This is the portable PyPI-backed alternative to the historical CUDA SDF
    implementation. The longest possible axis occupies at most ``resolution``
    voxels after normalization. Marching cubes always produces a closed surface;
    the explicit watertight check keeps that property part of the public contract.
    """

    _validate_mesh(mesh, source="mesh")
    if resolution < 32:
        raise ValueError("resolution must be at least 32")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive finite number")

    normalized = normalize_mesh(mesh, radius=radius)
    pitch = 2.0 * radius / float(resolution - 1)
    voxels = normalized.voxelized(pitch=pitch, method="subdivide").fill()
    result = voxels.marching_cubes
    result.apply_transform(voxels.transform)
    result = normalize_mesh(result, radius=radius)
    _validate_mesh(result, source="voxel marching-cubes mesh")
    if not result.is_watertight:
        raise ValueError("voxel reconstruction did not produce a watertight mesh")
    return result


def sdf_remesh_cuda(
    mesh: trimesh.Trimesh,
    *,
    resolution: int = 256,
    scale: float = 0.8,
    device: str = "cuda",
) -> trimesh.Trimesh:
    """Create a watertight surface with the optional CUDA SDF/DiffMC backend.

    This is the reusable core of the historical remesh workers. It intentionally
    imports torch, ``torchcumesh2sdf`` and ``diso`` only when requested. Output
    is centered and normalized, matching the old data-preparation convention.
    """
    _validate_mesh(mesh, source="mesh")
    if resolution < 32:
        raise ValueError("resolution must be at least 32")
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    if not device.startswith("cuda"):
        raise ValueError("the SDF backend currently requires a CUDA device")

    import torch
    import torchcumesh2sdf
    from diso import DiffMC

    normalized = normalize_mesh(mesh, radius=1.0 / (1.0 + 9.0 / resolution))
    triangles = np.asarray(normalized.triangles, dtype=np.float32)
    triangles -= triangles.min(axis=(0, 1), keepdims=True)
    extent = float(triangles.max())
    if extent <= np.finfo(np.float32).eps:
        raise ValueError("mesh has zero spatial extent")

    band = 3.0 / resolution
    triangles = (triangles / extent + band) / (1.0 + 2.0 * band)
    triangle_tensor = torch.as_tensor(triangles, dtype=torch.float32, device=device)
    triangle_tensor *= (1.0 / (1.0 + 2.0 * band)) * scale
    triangle_tensor = triangle_tensor * 0.5 + 0.5

    diffmc = DiffMC().to(device)
    udf = torchcumesh2sdf.get_udf(triangle_tensor, resolution, band)
    with torch.no_grad():
        vertices, faces = diffmc(udf - 1.0 / resolution)

    first_pass = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    _validate_mesh(first_pass, source="UDF marching-cubes mesh")

    first_pass_triangles = torch.as_tensor(
        np.asarray(first_pass.triangles, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    sdf = torchcumesh2sdf.get_sdf(first_pass_triangles, resolution, band)
    with torch.no_grad():
        vertices, faces = diffmc(sdf)

    result = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    return normalize_mesh(result)


def remesh_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_faces: int | None = None,
    backend: RemeshBackend = "simplify",
    simplify_backend: SimplifyBackend = "auto",
    normalize_radius: float = 0.45,
    voxel_resolution: int = 256,
    sdf_resolution: int = 256,
    sdf_scale: float = 0.8,
    device: str = "cuda",
) -> Path:
    """Normalize/remesh/simplify one file with no storage or cluster coupling."""
    mesh = load_mesh(input_path)
    if backend == "simplify":
        mesh = normalize_mesh(mesh, radius=normalize_radius)
    elif backend == "voxel":
        mesh = voxel_remesh(
            mesh,
            resolution=voxel_resolution,
            radius=normalize_radius,
        )
    elif backend == "cuda-sdf":
        mesh = sdf_remesh_cuda(
            mesh,
            resolution=sdf_resolution,
            scale=sdf_scale,
            device=device,
        )
    else:
        raise ValueError(f"unsupported remesh backend: {backend}")

    if target_faces is not None:
        mesh = simplify_mesh(mesh, target_faces, backend=simplify_backend)
    return export_mesh(mesh, output_path)
