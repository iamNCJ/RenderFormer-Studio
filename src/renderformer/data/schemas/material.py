from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MaterialType(str, Enum):
    DIFFUSE_SPECULAR = "diffuse_specular"
    METALLIC_ROUGHNESS = "metallic_roughness"
    METALLIC_ROUGHNESS_TRANSMISSION = "metallic_roughness_transmission"
    MEASURED_BRDF = "measured_brdf"


class MaterialSource(str, Enum):
    HOMOGENEOUS = "homogeneous"
    PROCEDURAL = "procedural"
    SVBRDF_TEXTURE = "svbrdf_texture"


class SpatialVariation(str, Enum):
    UNIFORM = "uniform"
    PER_TRIANGLE = "per_triangle"
    UV_TEXTURE = "uv_texture"


@dataclass(frozen=True)
class MaterialSemantic:
    type: MaterialType
    source: MaterialSource
    spatial_variation: SpatialVariation
    params: dict[str, Any] = field(default_factory=dict)
    texture_map_path: str | None = None
    use_heightmap: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialSemantic":
        return cls(
            type=MaterialType(data["type"]),
            source=MaterialSource(data["source"]),
            spatial_variation=SpatialVariation(data["spatial_variation"]),
            params=dict(data.get("params", {})),
            texture_map_path=data.get("texture_map_path"),
            use_heightmap=bool(data.get("use_heightmap", False)),
        )
