"""Optional mesh preparation utilities.

The default path only depends on NumPy and trimesh. CUDA SDF remeshing and
Blender UV unwrapping import their optional backends when called.
"""

from .mesh import (
    export_mesh,
    legacy_pymeshlab_remesh,
    load_mesh,
    normalize_mesh,
    remesh_file,
    simplify_mesh,
    slice_mesh,
    sdf_remesh_cuda,
    voxel_remesh,
)
from .camera import look_at_to_c2w
from .uv import UV_METHODS, unwrap_uv

__all__ = [
    "UV_METHODS",
    "export_mesh",
    "legacy_pymeshlab_remesh",
    "load_mesh",
    "look_at_to_c2w",
    "normalize_mesh",
    "remesh_file",
    "simplify_mesh",
    "slice_mesh",
    "sdf_remesh_cuda",
    "unwrap_uv",
    "voxel_remesh",
]
