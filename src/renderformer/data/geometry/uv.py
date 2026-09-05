"""UV-unwrapping helpers with a lazy Blender dependency."""

from __future__ import annotations

from pathlib import Path

from .mesh import export_mesh, load_mesh

UV_METHODS = frozenset(
    {
        "unwrap",
        "smart_project",
        "cube_project",
        "cylinder_project",
        "sphere_project",
    }
)


def unwrap_uv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "cube_project",
    normalize: bool = False,
    clean_export: bool = True,
) -> Path:
    """Unwrap a mesh using ``bpy_helper`` without importing Blender at startup."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() == destination.resolve():
        raise ValueError("input_path and output_path must be different")
    if method not in UV_METHODS:
        choices = ", ".join(sorted(UV_METHODS))
        raise ValueError(f"unsupported UV method {method!r}; choose one of: {choices}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    from bpy_helper.mesh import uv_unwrap

    uv_unwrap(
        str(source),
        str(destination),
        method=method,
        normalize=normalize,
    )
    if not destination.is_file():
        raise RuntimeError(f"UV backend did not create {destination}")

    if clean_export:
        export_mesh(load_mesh(destination), destination)
    return destination
