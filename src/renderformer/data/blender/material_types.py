"""
Material type definitions and BRDF parameterization mappings.
"""

from __future__ import annotations

import os
from pathlib import Path

from renderformer.checkpoints import (
    DEFAULT_V2_MAPPER_SUBFOLDERS,
    DEFAULT_V2_REPOSITORY,
)

# Material type constants
MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR = "svbrdf_diffuse_specular"
MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR = "homo_diffuse_specular"
MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS = "svbrdf_metallic_roughness"
MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS = "homo_metallic_roughness"
MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION = "homo_metallic_roughness_transmission"
MATERIAL_TYPE_HOMO_MEASURED_BRDF = "homo_measured_brdf"

# All material types
ALL_MATERIAL_TYPES = [
    MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
    MATERIAL_TYPE_HOMO_MEASURED_BRDF,
]

# Mapper weights are not bundled with the source tree. Each value can be a
# local directory or a Hugging Face model id accepted by ``from_pretrained``.
MAPPER_ENV_VARS = {
    "diffuse_specular": "RENDERFORMER_DIFFSPEC_MAPPER",
    "metallic_roughness": "RENDERFORMER_METALLIC_MAPPER",
    "metallic_roughness_transmission": "RENDERFORMER_TRANSMISSION_MAPPER",
}
MAPPER_SUBDIRECTORIES = {
    "diffuse_specular": "diffspec_mapper",
    "metallic_roughness": "metallic_mapper",
    "metallic_roughness_transmission": "metallic_transmission_mapper",
}
# Material type to BRDF parameterization mapping
MATERIAL_TYPE_TO_BRDF_TYPE = {
    MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR: "diffuse_specular",
    MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR: "diffuse_specular",
    MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS: "metallic_roughness",
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS: "metallic_roughness",
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION: "metallic_roughness_transmission",
}

# BRDF parameter dimensions
BRDF_PARAM_DIMS = {
    "diffuse_specular": 7,  # diffuse RGB(3) + specular RGB(3) + roughness(1)
    "metallic_roughness": 5,  # base_color RGB(3) + metallic(1) + roughness(1)
    "metallic_roughness_transmission": 7,  # base_color RGB(3) + metallic(1) + roughness(1) + IOR(1) + transmission_weight(1)
}

# Check if material type supports SVBRDF
def supports_svbrdf(material_type: str) -> bool:
    """Check if material type supports SVBRDF (texture maps)."""
    return material_type.startswith("svbrdf_")

# Check if material type supports homogeneous
def supports_homogeneous(material_type: str) -> bool:
    """Check if material type supports homogeneous materials."""
    return material_type.startswith("homo_")

# Get BRDF type from material type
def get_brdf_type(material_type: str) -> str:
    """Get BRDF parameterization type from material type."""
    return MATERIAL_TYPE_TO_BRDF_TYPE[material_type]

# Get mapper path from material type
def get_mapper_checkpoint(material_type: str) -> tuple[str, str | None]:
    """Resolve a mapper source and optional component subfolder."""

    brdf_type = get_brdf_type(material_type)
    variable = MAPPER_ENV_VARS[brdf_type]
    configured = os.environ.get(variable)
    if configured:
        return configured, None

    mapper_root = os.environ.get("RENDERFORMER_BRDF_MAPPER_ROOT")
    if mapper_root:
        return str(Path(mapper_root) / MAPPER_SUBDIRECTORIES[brdf_type]), None

    return DEFAULT_V2_REPOSITORY, DEFAULT_V2_MAPPER_SUBFOLDERS[brdf_type]


# Get BRDF parameter dimension from material type
def get_brdf_param_dim(material_type: str) -> int:
    """Get BRDF parameter dimension from material type."""
    brdf_type = get_brdf_type(material_type)
    return BRDF_PARAM_DIMS[brdf_type]
