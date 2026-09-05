"""CPU-only adapter for the public RenderFormer V1 compose-scene format."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Literal, Optional

from dacite import Config, from_dict
import numpy as np
import trimesh

from renderformer.data.blender.generated_config import MaterialConfig
from renderformer.data.exporters.common import (
    ObjectPreparedData,
    PreparedScene,
    RenderResults,
)
from renderformer.data.geometry.camera import look_at_to_c2w
from renderformer.data.geometry.mesh import legacy_pymeshlab_remesh
from renderformer.data.schemas.material import (
    MaterialSemantic,
    MaterialSource,
    MaterialType,
    SpatialVariation,
)
from renderformer.data.schemas.scene import ObjectSemantic, SceneSemantic


@dataclass
class LegacyRF1TransformConfig:
    translation: List[float]
    rotation: List[float]
    scale: List[float]
    normalize: bool = True


@dataclass
class LegacyRF1MaterialConfig:
    diffuse: List[float]
    specular: List[float]
    roughness: float
    emissive: List[float]
    smooth_shading: bool
    rand_tri_diffuse_seed: Optional[int] = None
    random_diffuse_max: float = 1.0
    random_diffuse_type: Literal["per-triangle", "per-shading-group"] = (
        "per-shading-group"
    )


@dataclass
class LegacyRF1ObjectConfig:
    mesh_path: str
    material: LegacyRF1MaterialConfig
    transform: LegacyRF1TransformConfig
    remesh: bool = False
    remesh_target_face_num: int = 2048


@dataclass
class LegacyRF1CameraConfig:
    position: List[float]
    look_at: List[float]
    up: List[float]
    fov: float


@dataclass
class LegacyRF1SceneConfig:
    scene_name: str
    version: str
    objects: Dict[str, LegacyRF1ObjectConfig]
    cameras: List[LegacyRF1CameraConfig]


def load_legacy_rf1_scene(path: str | Path) -> LegacyRF1SceneConfig:
    """Load the strict public RF1 JSON schema used by ``scene_processor``."""
    scene_path = Path(path)
    with scene_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return from_dict(
        data_class=LegacyRF1SceneConfig,
        data=data,
        config=Config(check_types=True, strict=True),
    )


def _normalize_to_unit_sphere(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    radius = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max() * 2.0
    mesh.vertices = mesh.vertices / radius
    return mesh


def _load_legacy_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"legacy RF1 object did not resolve to one triangle mesh: {path}")
    return mesh


def _apply_legacy_transform(
    mesh: trimesh.Trimesh,
    transform: LegacyRF1TransformConfig,
) -> trimesh.Trimesh:
    for axis, angle in enumerate(transform.rotation):
        axis_array = np.asarray(
            [1, 0, 0] if axis == 0 else [0, 1, 0] if axis == 1 else [0, 0, 1],
            dtype=float,
        )
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.deg2rad(angle),
            axis_array,
        )
        mesh.apply_transform(rotation_matrix)
    mesh.apply_scale(transform.scale)
    mesh.apply_translation(transform.translation)
    return mesh


def _apply_legacy_shading(
    mesh: trimesh.Trimesh,
    smooth_shading: bool,
) -> trimesh.Trimesh:
    if smooth_shading:
        return trimesh.graph.smooth_shade(mesh, angle=np.radians(30))
    return trimesh.Trimesh(
        vertices=mesh.triangles.reshape(-1, 3),
        faces=np.arange(len(mesh.triangles) * 3).reshape(-1, 3),
        process=False,
    )


def _apply_legacy_diffuse(
    mesh: trimesh.Trimesh,
    material: LegacyRF1MaterialConfig,
) -> trimesh.Trimesh:
    if material.rand_tri_diffuse_seed is None:
        vertex_colors = (np.asarray(material.diffuse) * 255.0).clip(0, 255).astype(int)
        vertex_colors = np.tile(vertex_colors, mesh.vertices.shape[0]).reshape(-1, 3)
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=vertex_colors)
        return mesh

    random_state = np.random.RandomState(material.rand_tri_diffuse_seed)
    split_kwargs: dict[str, object] = {"only_watertight": False}
    if material.random_diffuse_type == "per-triangle":
        split_kwargs["adjacency"] = np.asarray([])

    mesh_parts = []
    for small_mesh in mesh.split(**split_kwargs):
        shared_color = random_state.randint(
            0,
            math.ceil(256 * material.random_diffuse_max),
            (1, 3),
        ).repeat(small_mesh.faces.shape[0], axis=0)
        mesh_parts.append(
            trimesh.Trimesh(
                vertices=small_mesh.vertices,
                faces=small_mesh.faces,
                vertex_normals=small_mesh.vertex_normals,
                face_colors=shared_color,
                process=False,
            )
        )
    return trimesh.util.concatenate(mesh_parts)


def _write_legacy_split_meshes(
    scene: LegacyRF1SceneConfig,
    scene_dir: Path,
    split_dir: Path,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for object_name, obj in scene.objects.items():
        mesh = _load_legacy_mesh(scene_dir / obj.mesh_path)
        if obj.transform.normalize:
            mesh = _normalize_to_unit_sphere(mesh)
        if obj.remesh:
            mesh = legacy_pymeshlab_remesh(mesh, obj.remesh_target_face_num)
        mesh = _apply_legacy_transform(mesh, obj.transform)
        mesh = _apply_legacy_shading(mesh, obj.material.smooth_shading)
        mesh = _apply_legacy_diffuse(mesh, obj.material)

        # Accessing the property forces trimesh to calculate normals before export,
        # matching the legacy scene processor's OBJ round-trip.
        _ = mesh.vertex_normals.shape
        mesh.export(
            split_dir / f"{object_name}.obj",
            include_normals=True,
            include_texture=True,
        )


def _prepared_object(
    object_name: str,
    obj: LegacyRF1ObjectConfig,
    split_dir: Path,
) -> ObjectPreparedData:
    mesh = trimesh.load(
        split_dir / f"{object_name}.obj",
        process=False,
        force="mesh",
    )
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"split RF1 object did not resolve to one triangle mesh: {object_name}")

    triangles = mesh.triangles
    vn = mesh.vertex_normals[mesh.faces]
    face_diffuse = np.asarray(mesh.visual.face_colors)[..., :3] / 255.0
    material = MaterialConfig(
        material_type="homo_diffuse_specular",
        diffuse_specular_params={
            "diffuse": obj.material.diffuse,
            "specular": obj.material.specular,
            "roughness": obj.material.roughness,
        },
        emissive=obj.material.emissive,
        smooth_shading=obj.material.smooth_shading,
        rand_tri_diffuse_seed=obj.material.rand_tri_diffuse_seed,
    )
    return ObjectPreparedData(
        obj_key=object_name,
        triangles=triangles,
        vn=vn,
        face_diffuse=face_diffuse,
        material=material,
    )


def _scene_semantic(scene: LegacyRF1SceneConfig, scene_dir: Path) -> SceneSemantic:
    objects = {}
    for object_name, obj in scene.objects.items():
        spatial_variation = (
            SpatialVariation.PER_TRIANGLE
            if obj.material.rand_tri_diffuse_seed is not None
            else SpatialVariation.UNIFORM
        )
        objects[object_name] = ObjectSemantic(
            mesh_path=str(scene_dir / obj.mesh_path),
            material=MaterialSemantic(
                type=MaterialType.DIFFUSE_SPECULAR,
                source=MaterialSource.HOMOGENEOUS,
                spatial_variation=spatial_variation,
                params={
                    "diffuse": obj.material.diffuse,
                    "specular": obj.material.specular,
                    "roughness": obj.material.roughness,
                    "emissive": obj.material.emissive,
                },
            ),
            is_background=object_name.startswith("background"),
            is_lighting=any(value != 0 for value in obj.material.emissive),
        )
    return SceneSemantic(
        scene_name=scene.scene_name,
        version=scene.version,
        objects=objects,
    )


def _render_results(scene: LegacyRF1SceneConfig) -> RenderResults:
    c2w = np.stack(
        [
            look_at_to_c2w(camera.position, camera.look_at, camera.up)
            for camera in scene.cameras
        ]
    )
    fov = np.asarray([camera.fov for camera in scene.cameras])
    return RenderResults(c2w=c2w, fov=fov)


def prepare_legacy_rf1_scene(scene_config_path: str | Path) -> PreparedScene:
    """Prepare a legacy RF1 scene without importing Blender or running Cycles."""
    scene_path = Path(scene_config_path)
    scene = load_legacy_rf1_scene(scene_path)
    scene_dir = scene_path.parent

    with TemporaryDirectory(prefix="renderformer-rf1-compose-") as work_dir:
        split_dir = Path(work_dir) / "split"
        _write_legacy_split_meshes(scene, scene_dir, split_dir)
        objects = [
            _prepared_object(object_name, obj, split_dir)
            for object_name, obj in scene.objects.items()
        ]

    return PreparedScene(
        scene=_scene_semantic(scene, scene_dir),
        objects=objects,
        render_results=_render_results(scene),
        output_stem=scene_path.stem,
    )
