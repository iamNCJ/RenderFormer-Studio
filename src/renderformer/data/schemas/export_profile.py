from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from renderformer.data.schemas.material import MaterialType, SpatialVariation


class ExportFormat(str, Enum):
    V1 = "v1"
    V2 = "v2"


class UnsupportedPolicy(str, Enum):
    ERROR = "error"
    WARN = "warn"
    DROP = "drop"


class TextureExportMode(str, Enum):
    RAW_ONLY = "raw_only"
    ENCODED_ONLY = "encoded_only"
    RAW_AND_ENCODED = "raw_and_encoded"


class EncodeTiming(str, Enum):
    INLINE = "inline"
    POSTPROCESS = "postprocess"


@dataclass(frozen=True)
class TextureExport:
    mode: TextureExportMode = TextureExportMode.RAW_ONLY
    encode_timing: EncodeTiming = EncodeTiming.INLINE
    spatial_encoding: str = "constant_1x1_expanded_to_32"
    texture_encoder: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TextureExport":
        values = dict(data or {})
        return cls(
            mode=TextureExportMode(values.get("mode", "raw_only")),
            encode_timing=EncodeTiming(values.get("encode_timing", "inline")),
            spatial_encoding=values.get(
                "spatial_encoding",
                "constant_1x1_expanded_to_32",
            ),
            texture_encoder=dict(values.get("texture_encoder", {})),
        )


@dataclass(frozen=True)
class ExportProfile:
    format: ExportFormat
    schema_version: int
    unsupported_policy: UnsupportedPolicy = UnsupportedPolicy.ERROR
    allowed_material_types: Sequence[MaterialType] = (MaterialType.DIFFUSE_SPECULAR,)
    allowed_spatial_variation: Sequence[SpatialVariation] = (
        SpatialVariation.UNIFORM,
        SpatialVariation.PER_TRIANGLE,
    )
    # Capability flags: each profile declares whether it carries a given feature.
    # The validator hard-rejects scenes that use a feature the profile disallows.
    allow_env_map: bool = False
    allow_volume: bool = False
    allow_heightmap: bool = False
    material_encoding: dict[str, Any] = field(default_factory=dict)
    texture_export: TextureExport = field(default_factory=TextureExport)
    env_map: dict[str, Any] = field(default_factory=dict)
    volume: dict[str, Any] = field(default_factory=dict)
    render_passes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportProfile":
        try:
            export_format = ExportFormat(data["format"])
        except ValueError as exc:
            raise ValueError(
                f"Unsupported export format: {data.get('format')}"
            ) from exc

        env_map = dict(data.get("env_map", {}))
        volume = dict(data.get("volume", {}))
        # Capability defaults derive from existing sections so older yamls keep working.
        allow_env_map = bool(
            data.get(
                "allow_env_map",
                env_map.get("enabled") or env_map.get("include") or False,
            )
        )
        allow_volume = bool(
            data.get(
                "allow_volume",
                volume.get("enabled") or volume.get("include") or False,
            )
        )
        allow_heightmap = bool(data.get("allow_heightmap", False))
        return cls(
            format=export_format,
            schema_version=int(data["schema_version"]),
            unsupported_policy=UnsupportedPolicy(
                data.get("unsupported_policy", "error")
            ),
            allowed_material_types=tuple(
                MaterialType(value)
                for value in data.get("allowed_material_types", ["diffuse_specular"])
            ),
            allowed_spatial_variation=tuple(
                SpatialVariation(value)
                for value in data.get(
                    "allowed_spatial_variation",
                    ["uniform", "per_triangle"],
                )
            ),
            allow_env_map=allow_env_map,
            allow_volume=allow_volume,
            allow_heightmap=allow_heightmap,
            material_encoding=dict(data.get("material_encoding", {})),
            texture_export=TextureExport.from_dict(data.get("texture_export")),
            env_map=env_map,
            volume=volume,
            render_passes=dict(data.get("render_passes", {})),
        )
