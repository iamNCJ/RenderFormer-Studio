from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from renderformer.data.schemas.scene import SceneSemantic

if TYPE_CHECKING:
    from renderformer.data.blender.generated_config import MaterialConfig


@dataclass(frozen=True)
class RenderResults:
    c2w: np.ndarray
    fov: np.ndarray
    mvp: np.ndarray | None = None
    img: np.ndarray | None = None
    diffuse_img: np.ndarray | None = None
    glossy_img: np.ndarray | None = None
    normal_img: np.ndarray | None = None
    albedo_img: np.ndarray | None = None


@dataclass(frozen=True)
class ObjectPreparedData:
    """Canonical per-object geometry + material data, format-agnostic.

    Exporters consume these to build their format-specific texture channels:
    v1 maps to 13-channel raw diffuse/specular/roughness/normal[NDC]/emissive;
    v2 maps face_diffuse + material through `map_brdf_to_latent` (or uses
    `svbrdf_texture` directly when the material is UV-textured).
    """

    obj_key: str
    triangles: np.ndarray            # (N_i, 3, 3) float32
    vn: np.ndarray                   # (N_i, 3, 3) float32
    face_diffuse: np.ndarray         # (N_i, 3) float32 in [0, 1] — Blender per-face color
    material: "MaterialConfig"       # full original material config
    svbrdf_texture: np.ndarray | None = None  # (N_i, 15 or 16, H, W) when material is SVBRDF


@dataclass(frozen=True)
class PreparedScene:
    scene: SceneSemantic
    objects: list[ObjectPreparedData]
    render_results: RenderResults
    output_stem: str = "scene"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    metadata: dict[str, Any]
