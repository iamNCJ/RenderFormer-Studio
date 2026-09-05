"""Legacy public scene JSON to RF1 H5 composition."""

from __future__ import annotations

from pathlib import Path

from renderformer.data.compat.rf1_scene import prepare_legacy_rf1_scene
from renderformer.data.exporters.common import ExportResult
from renderformer.data.exporters.v1 import export_v1_h5
from renderformer.data.schemas.export_profile import ExportProfile


def compose_rf1_h5(
    scene_config_path: str | Path,
    output_path: str | Path,
    *,
    legacy_texture_quantization: bool = True,
) -> ExportResult:
    """Compose one legacy RF1 scene using only CPU mesh operations."""
    prepared = prepare_legacy_rf1_scene(scene_config_path)
    profile = ExportProfile.from_dict(
        {
            "format": "v1",
            "schema_version": 1,
            "unsupported_policy": "error",
            "allowed_material_types": ["diffuse_specular"],
            "allowed_spatial_variation": ["uniform", "per_triangle"],
            "allow_env_map": False,
            "allow_volume": False,
            "allow_heightmap": False,
        }
    )
    return export_v1_h5(
        prepared,
        profile,
        output_path,
        legacy_texture_quantization=legacy_texture_quantization,
    )
