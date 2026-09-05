from pathlib import Path
from typing import Any

import numpy as np
import simple_exr
import trimesh

from renderformer.data.blender.generated_config import GeneratedConfig, MaterialConfig, VolumeConfig
from renderformer.data.blender.material_types import (
    MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_HOMO_MEASURED_BRDF,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
    MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
    supports_svbrdf,
)
from renderformer.data.blender.renderer import (
    get_mvp_matrix,
    get_projection_matrix,
    is_black_image,
    is_z_fighting,
)
from renderformer.data.blender.texture_sample_fn import texture_sample
from renderformer.data.blender.volume_utils import read_vdb_density, read_vdb_to_micro_8_4
from renderformer.data.exporters.common import ObjectPreparedData, PreparedScene, RenderResults
from renderformer.data.schemas.material import (
    MaterialSemantic,
    MaterialSource,
    MaterialType,
    SpatialVariation,
)
from renderformer.data.schemas.scene import ObjectSemantic, SceneSemantic


RenderTuple = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _semantic_material(material: MaterialConfig) -> MaterialSemantic:
    mapping = {
        MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR: MaterialType.DIFFUSE_SPECULAR,
        MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR: MaterialType.DIFFUSE_SPECULAR,
        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS: MaterialType.METALLIC_ROUGHNESS,
        MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS: MaterialType.METALLIC_ROUGHNESS,
        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION: MaterialType.METALLIC_ROUGHNESS_TRANSMISSION,
        MATERIAL_TYPE_HOMO_MEASURED_BRDF: MaterialType.MEASURED_BRDF,
    }
    source = MaterialSource.SVBRDF_TEXTURE if supports_svbrdf(material.material_type) else MaterialSource.HOMOGENEOUS
    spatial = SpatialVariation.UV_TEXTURE if source is MaterialSource.SVBRDF_TEXTURE else SpatialVariation.UNIFORM
    if material.rand_tri_diffuse_seed is not None:
        spatial = SpatialVariation.PER_TRIANGLE
    return MaterialSemantic(
        type=mapping[material.material_type],
        source=source,
        spatial_variation=spatial,
        texture_map_path=material.texture_map_path,
        use_heightmap=material.use_heightmap,
    )


def _scene_semantic(scene_config: GeneratedConfig) -> SceneSemantic:
    objects = {
        obj_key: ObjectSemantic(
            mesh_path=obj_config.mesh_path,
            is_background=obj_config.is_background,
            is_lighting=obj_config.is_lighting,
            material=_semantic_material(obj_config.material),
        )
        for obj_key, obj_config in scene_config.objects.items()
    }
    return SceneSemantic(
        scene_name=scene_config.scene_name,
        version=scene_config.version,
        objects=objects,
        has_env_map=scene_config.env_map is not None,
        has_volume=bool(scene_config.volumes),
    )


def _read_face_diffuse(mesh: trimesh.Trimesh, material: MaterialConfig, num_triangles: int) -> np.ndarray:
    """Per-face diffuse RGB in [0, 1].

    Mirrors RF1's `mesh.visual.face_colors / 255.` so that `rand_tri_diffuse_seed`
    per-face colors painted by Blender survive into the canonical scene representation.
    Falls back to the homogeneous material diffuse when face_colors aren't populated.
    """
    face_colors = getattr(mesh.visual, "face_colors", None)
    if face_colors is not None and len(face_colors) == num_triangles:
        return np.asarray(face_colors)[:, :3].astype(np.float32) / 255.0
    params = material.diffuse_specular_params or {}
    homo = params.get("diffuse")
    if homo is None and material.metallic_roughness_params is not None:
        homo = material.metallic_roughness_params.get("base_color")
    if homo is None:
        homo = [0.5, 0.5, 0.5]
    return np.broadcast_to(np.asarray(homo, dtype=np.float32), (num_triangles, 3)).astype(np.float32, copy=True)


def _collect_objects(
    scene_config: GeneratedConfig,
    mesh_path: Path,
    texture_size: int,
) -> list[ObjectPreparedData]:
    """Produce per-object canonical data without applying any format-specific encoding.

    For SVBRDF materials the UV-sampled texture (15 or 16 channels at the material's
    natural shape — material.use_heightmap controls heightmap presence) is included
    because building it requires the on-disk PNG assets. For homogeneous materials
    the exporter is responsible for encoding the BRDF + face_diffuse into its target
    channel layout.
    """
    out: list[ObjectPreparedData] = []
    split_mesh_path = mesh_path.parent / "split"

    for obj_key, obj_config in scene_config.objects.items():
        mesh = trimesh.load(split_mesh_path / f"{obj_key}.obj", process=False, force="mesh")
        triangles = mesh.triangles
        vn = mesh.vertex_normals[mesh.faces]
        material = obj_config.material
        face_diffuse = _read_face_diffuse(mesh, material, triangles.shape[0])

        svbrdf_texture: np.ndarray | None = None
        if supports_svbrdf(material.material_type):
            svbrdf_texture = texture_sample(
                mesh,
                str(split_mesh_path / obj_key),
                size=texture_size,
                material_type=material.material_type,
                use_latent=True,
                emissive=material.emissive,
                use_heightmap=material.use_heightmap,
            )

        out.append(
            ObjectPreparedData(
                obj_key=obj_key,
                triangles=triangles.astype(np.float32),
                vn=vn.astype(np.float32),
                face_diffuse=face_diffuse,
                material=material,
                svbrdf_texture=svbrdf_texture,
            )
        )
    return out


def _filter_and_stack_render_results(
    scene_config: GeneratedConfig,
    render_results: list[RenderTuple],
    min_available_view: int,
    test_mode: bool,
) -> RenderResults:
    valid_results = []
    valid_cameras = []
    for idx, (img, c2w, depth, diffuse_img, glossy_img, normal_img, albedo_img) in enumerate(render_results):
        if is_black_image(img) or is_z_fighting(depth):
            continue
        valid_results.append((img, c2w, diffuse_img, glossy_img, normal_img, albedo_img))
        valid_cameras.append(scene_config.cameras[idx])

    if len(valid_results) < min_available_view:
        raise ValueError(f"not enough valid camera views: {len(valid_results)} < {min_available_view}")
    if not valid_results:
        raise ValueError("no valid camera views after filtering")
    if test_mode:
        valid_results = valid_results[:min_available_view]
        valid_cameras = valid_cameras[:min_available_view]

    c2w_list = []
    fov_list = []
    mvp_list = []
    img_list = []
    diffuse_list = []
    glossy_list = []
    normal_list = []
    albedo_list = []
    for idx, (img, c2w, diffuse_img, glossy_img, normal_img, albedo_img) in enumerate(valid_results):
        proj_mtx = get_projection_matrix(np.radians(valid_cameras[idx].fov), 1.0, 0.05, 5.0)
        c2w_list.append(c2w)
        fov_list.append(valid_cameras[idx].fov)
        mvp_list.append(get_mvp_matrix(c2w[None], proj_mtx))
        img_list.append(img)
        diffuse_list.append(diffuse_img)
        glossy_list.append(glossy_img)
        normal_list.append(normal_img)
        albedo_list.append(albedo_img)

    return RenderResults(
        c2w=np.stack(c2w_list).astype(np.float32),
        fov=np.asarray(fov_list, dtype=np.float32),
        mvp=np.stack(mvp_list).astype(np.float32),
        img=np.stack(img_list).astype(np.float32),
        diffuse_img=np.stack(diffuse_list).astype(np.float32),
        glossy_img=np.stack(glossy_list).astype(np.float32),
        normal_img=np.stack(normal_list).astype(np.float32),
        albedo_img=np.stack(albedo_list).astype(np.float32),
    )


def _volume_voxel_grids(
    volume_config: VolumeConfig,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_coord = np.asarray(metadata["min_coord"], dtype=np.float32)
    max_coord = np.asarray(metadata["max_coord"], dtype=np.float32)
    voxel_size = float(metadata["voxel_size"])
    coarse_res = 8
    micro_per = 4

    coarse_idx = np.indices((coarse_res, coarse_res, coarse_res), dtype=np.float32)
    i = coarse_idx[0]
    j = coarse_idx[1]
    k = coarse_idx[2]
    center_i = i * micro_per + micro_per // 2
    center_j = j * micro_per + micro_per // 2
    center_k = k * micro_per + micro_per // 2

    if volume_config.transpose_written:
        vdb_0 = min_coord[0] + center_k
        vdb_1 = min_coord[1] + center_j
        vdb_2 = min_coord[2] + center_i
    else:
        vdb_0 = min_coord[0] + center_i
        vdb_1 = min_coord[1] + center_j
        vdb_2 = min_coord[2] + center_k

    origin_offset = -((min_coord + max_coord) / 2.0 + 0.5) * voxel_size
    half_voxel = voxel_size / 2.0
    canonical = np.stack(
        [
            vdb_0 * voxel_size + origin_offset[0] + half_voxel,
            vdb_1 * voxel_size + origin_offset[1] + half_voxel,
            vdb_2 * voxel_size + origin_offset[2] + half_voxel,
        ],
        axis=-1,
    ).astype(np.float32)

    transform = volume_config.transform
    rotation = np.radians(
        [transform.rotation_x, transform.rotation_y, transform.rotation_z]
    ).astype(np.float32)
    scale = np.asarray(
        [transform.scale_x, transform.scale_y, transform.scale_z],
        dtype=np.float32,
    )
    translation = np.asarray(
        [transform.translation_x, transform.translation_y, transform.translation_z],
        dtype=np.float32,
    )
    rotation_matrix = trimesh.transformations.euler_matrix(
        rotation[0], rotation[1], rotation[2], "sxyz"
    )[:3, :3]
    positions = (canonical * scale) @ rotation_matrix.T + translation

    grid_shape = (coarse_res, coarse_res, coarse_res, 3)
    rotations = np.broadcast_to(rotation, grid_shape).astype(np.float32).copy()
    scales = np.broadcast_to(scale, grid_shape).astype(np.float32).copy()
    scattering = np.broadcast_to(
        np.asarray(volume_config.scattering_scale, dtype=np.float32), grid_shape
    ).copy()
    absorption = np.broadcast_to(
        np.asarray(volume_config.absorption_scale, dtype=np.float32), grid_shape
    ).copy()
    return scattering, absorption, positions.astype(np.float32), rotations, scales


def _compact_micro_volume_tokens(
    volume_data_array: np.ndarray,
    scattering_array: np.ndarray,
    absorption_array: np.ndarray,
    position_array: np.ndarray,
    rotation_array: np.ndarray,
    scale_array: np.ndarray,
    occupancy_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    if volume_data_array.ndim != 7 or volume_data_array.shape[1:] != (8, 4, 8, 4, 8, 4):
        raise ValueError(
            "volume_data_array must have shape (num_volumes, 8, 4, 8, 4, 8, 4), "
            f"got {volume_data_array.shape}"
        )

    micro_blocks = volume_data_array.transpose(0, 1, 3, 5, 2, 4, 6)
    occupancy_mask = micro_blocks.max(axis=(4, 5, 6)) > occupancy_threshold
    occupied_indices = np.argwhere(occupancy_mask)
    num_occupied = occupied_indices.shape[0]

    if num_occupied == 0:
        return {
            "volume_data": np.zeros((0, 4, 4, 4), dtype=np.float32),
            "volume_scattering_scale": np.zeros((0, 3), dtype=np.float32),
            "volume_absorption_scale": np.zeros((0, 3), dtype=np.float32),
            "volume_position": np.zeros((0, 3), dtype=np.float32),
            "volume_rotation": np.zeros((0, 3), dtype=np.float32),
            "volume_scale": np.zeros((0, 3), dtype=np.float32),
            "volume_indices": np.zeros((0,), dtype=np.int32),
            "voxel_indices": np.zeros((0, 3), dtype=np.int32),
        }

    volume_indices = occupied_indices[:, 0]
    voxel_i = occupied_indices[:, 1]
    voxel_j = occupied_indices[:, 2]
    voxel_k = occupied_indices[:, 3]
    return {
        "volume_data": micro_blocks[occupancy_mask].astype(np.float32),
        "volume_scattering_scale": scattering_array[
            volume_indices, voxel_i, voxel_j, voxel_k
        ].astype(np.float32),
        "volume_absorption_scale": absorption_array[
            volume_indices, voxel_i, voxel_j, voxel_k
        ].astype(np.float32),
        "volume_position": position_array[
            volume_indices, voxel_i, voxel_j, voxel_k
        ].astype(np.float32),
        "volume_rotation": rotation_array[
            volume_indices, voxel_i, voxel_j, voxel_k
        ].astype(np.float32),
        "volume_scale": scale_array[volume_indices, voxel_i, voxel_j, voxel_k].astype(
            np.float32
        ),
        "volume_indices": volume_indices.astype(np.int32),
        "voxel_indices": occupied_indices[:, 1:4].astype(np.int32),
    }


def _extra_datasets(scene_config: GeneratedConfig) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if scene_config.env_map is not None:
        env_map_path = Path(scene_config.env_map.env_map_path).expanduser()
        extra["env_map"] = simple_exr.read_exr(str(env_map_path)).astype(np.float32)
        extra["env_map_rotation"] = np.array(
            [
                scene_config.env_map.rotation_x,
                scene_config.env_map.rotation_y,
                scene_config.env_map.rotation_z,
            ],
            dtype=np.float32,
        )
        extra["env_map_strength"] = np.array([scene_config.env_map.strength], dtype=np.float32)

    if scene_config.volumes:
        volume_arrays = []
        scattering = []
        absorption = []
        positions = []
        rotations = []
        scales = []
        for volume_config in scene_config.volumes:
            volume_arrays.append(
                read_vdb_to_micro_8_4(
                    volume_config.vdb_path,
                    transpose_written=volume_config.transpose_written,
                )
            )
            _, metadata = read_vdb_density(volume_config.vdb_path, grid_name="density")
            voxel_grids = _volume_voxel_grids(volume_config, metadata)
            scattering.append(voxel_grids[0])
            absorption.append(voxel_grids[1])
            positions.append(voxel_grids[2])
            rotations.append(voxel_grids[3])
            scales.append(voxel_grids[4])
        if volume_arrays:
            extra.update(
                _compact_micro_volume_tokens(
                    np.asarray(volume_arrays, dtype=np.float32),
                    np.asarray(scattering, dtype=np.float32),
                    np.asarray(absorption, dtype=np.float32),
                    np.asarray(positions, dtype=np.float32),
                    np.asarray(rotations, dtype=np.float32),
                    np.asarray(scales, dtype=np.float32),
                )
            )
    return extra


def build_prepared_scene(
    scene_config: GeneratedConfig,
    mesh_path: str | Path,
    render_results: list[RenderTuple],
    min_available_view: int = 0,
    test_mode: bool = False,
    texture_size: int = 64,
) -> PreparedScene:
    mesh_path = Path(mesh_path)
    objects = _collect_objects(
        scene_config=scene_config,
        mesh_path=mesh_path,
        texture_size=texture_size,
    )
    return PreparedScene(
        scene=_scene_semantic(scene_config),
        objects=objects,
        render_results=_filter_and_stack_render_results(
            scene_config=scene_config,
            render_results=render_results,
            min_available_view=min_available_view,
            test_mode=test_mode,
        ),
        output_stem=scene_config.scene_name,
        extra=_extra_datasets(scene_config),
    )
