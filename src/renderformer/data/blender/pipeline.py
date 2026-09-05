import shutil
from pathlib import Path

from renderformer.data.blender.mesh import generate_scene_from_template, generate_scene_mesh
from renderformer.data.blender.renderer import load_scene_config, scene_to_img
from renderformer.data.blender.scene_builder import build_prepared_scene
from renderformer.data.blender.templates import (
    apply_background_topology_debias,
    load_scene_template,
    load_template_asset_lists,
    override_template_asset_paths,
)
from renderformer.data.exporters.common import PreparedScene


def _portable_frame_asset_path(
    path: str,
    frame_dir: Path,
    *,
    label: str,
) -> Path:
    """Resolve one bundle-local path and reject host-specific dependencies."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        raise ValueError(
            f"prepared frame {label} must be relative to the frame directory: {path}"
        )
    resolved = (frame_dir / candidate).resolve()
    try:
        resolved.relative_to(frame_dir)
    except ValueError as exc:
        raise ValueError(
            f"prepared frame {label} escapes the frame directory: {path}"
        ) from exc
    return resolved


def _resolve_prepared_frame_asset_paths(scene_config, frame_dir: Path) -> None:
    """Resolve portable Blender-export paths relative to their frame directory."""

    def resolve(path: str | None) -> str | None:
        if path is None:
            return None
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = frame_dir / candidate
        return str(candidate.resolve())

    for obj_config in scene_config.objects.values():
        obj_config.mesh_path = resolve(obj_config.mesh_path)
        obj_config.material.texture_map_path = resolve(
            obj_config.material.texture_map_path
        )
        obj_config.material.measured_brdf_npy_path = resolve(
            obj_config.material.measured_brdf_npy_path
        )
    for obj_config in scene_config.lighting:
        obj_config.mesh_path = resolve(obj_config.mesh_path)
        obj_config.material.texture_map_path = resolve(
            obj_config.material.texture_map_path
        )
        obj_config.material.measured_brdf_npy_path = resolve(
            obj_config.material.measured_brdf_npy_path
        )
    if scene_config.env_map is not None:
        scene_config.env_map.env_map_path = resolve(scene_config.env_map.env_map_path)
    for volume_config in scene_config.volumes or []:
        volume_config.vdb_path = resolve(volume_config.vdb_path)


def validate_prepared_frame(frame_dir: str | Path) -> tuple[Path, Path]:
    """Validate the portable layout emitted by the Blender frame exporter."""

    frame_dir = Path(frame_dir).expanduser().resolve()
    config_path = frame_dir / "scene_config.json"
    split_dir = frame_dir / "split"
    if not config_path.is_file():
        raise FileNotFoundError(f"prepared frame is missing scene_config.json: {frame_dir}")
    if not split_dir.is_dir():
        raise FileNotFoundError(f"prepared frame is missing split/: {frame_dir}")

    scene_config = load_scene_config(str(config_path))
    if scene_config.lighting:
        raise ValueError(
            "prepared frame lighting must be stored in objects with "
            "is_lighting=true; the legacy lighting list is not consumed"
        )

    from renderformer.data.blender.material_types import (
        MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
        MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
    )

    for obj_key, obj_config in scene_config.objects.items():
        transform = obj_config.transform
        prepared_transform = {
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_z": 0.0,
            "rotation_x": 0.0,
            "rotation_y": 0.0,
            "rotation_z": 0.0,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "scale_z": 1.0,
            "normalize": False,
        }
        non_identity = [
            field
            for field, expected in prepared_transform.items()
            if getattr(transform, field) != expected
        ]
        if non_identity:
            raise ValueError(
                f"prepared frame object {obj_key!r} must have an identity transform "
                "because its split OBJ is already in world space; non-identity fields: "
                + ", ".join(non_identity)
            )

        mesh_path = _portable_frame_asset_path(
            obj_config.mesh_path,
            frame_dir,
            label=f"object {obj_key!r} mesh_path",
        )
        expected_mesh_path = (split_dir / f"{obj_key}.obj").resolve()
        if mesh_path != expected_mesh_path:
            raise ValueError(
                f"prepared frame object {obj_key!r} mesh_path must be "
                f"split/{obj_key}.obj, got {obj_config.mesh_path}"
            )
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"prepared frame is missing object mesh: "
                f"{mesh_path.relative_to(frame_dir)}"
            )

        material = obj_config.material
        if material.texture_map_path is not None:
            texture_dir = _portable_frame_asset_path(
                material.texture_map_path,
                frame_dir,
                label=f"object {obj_key!r} texture_map_path",
            )
            if not texture_dir.is_dir():
                raise FileNotFoundError(
                    f"prepared frame object {obj_key!r} is missing texture directory: "
                    f"{material.texture_map_path}"
                )
            if material.material_type == MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR:
                required_maps = [
                    "diffuse.png",
                    "specular.png",
                    "roughness.png",
                    "normal.png",
                ]
            elif material.material_type == MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS:
                required_maps = ["metallic.png", "roughness.png", "normal.png"]
                if not any(
                    (texture_dir / name).is_file()
                    for name in ("basecolor.png", "base_color.png")
                ):
                    raise FileNotFoundError(
                        f"prepared frame object {obj_key!r} texture directory is "
                        "missing basecolor.png or base_color.png"
                    )
            else:
                required_maps = []
            if material.use_heightmap:
                required_maps.append("height.png")
            missing_maps = [
                name for name in required_maps if not (texture_dir / name).is_file()
            ]
            if missing_maps:
                raise FileNotFoundError(
                    f"prepared frame object {obj_key!r} texture directory is missing: "
                    + ", ".join(missing_maps)
                )

        if material.measured_brdf_npy_path is not None:
            measured_path = _portable_frame_asset_path(
                material.measured_brdf_npy_path,
                frame_dir,
                label=f"object {obj_key!r} measured_brdf_npy_path",
            )
            if not measured_path.is_file():
                raise FileNotFoundError(
                    f"prepared frame object {obj_key!r} is missing measured BRDF: "
                    f"{material.measured_brdf_npy_path}"
                )

    if scene_config.env_map is not None:
        env_map_path = _portable_frame_asset_path(
            scene_config.env_map.env_map_path,
            frame_dir,
            label="env_map_path",
        )
        if not env_map_path.is_file():
            raise FileNotFoundError(
                f"prepared frame is missing environment map: "
                f"{scene_config.env_map.env_map_path}"
            )
    for volume_idx, volume_config in enumerate(scene_config.volumes or []):
        vdb_path = _portable_frame_asset_path(
            volume_config.vdb_path,
            frame_dir,
            label=f"volume {volume_idx} vdb_path",
        )
        if not vdb_path.is_file():
            raise FileNotFoundError(
                f"prepared frame is missing volume {volume_idx}: {volume_config.vdb_path}"
            )
    return config_path, split_dir


def _validate_preprocessed_prepared_frame(scene_config, split_dir: Path) -> None:
    """Ensure --skip-mesh-generation has every derived SVBRDF latent map."""

    missing = []
    for obj_key, obj_config in scene_config.objects.items():
        if obj_config.material.texture_map_path is None:
            continue
        for latent_idx in range(3):
            latent_path = split_dir / obj_key / f"latent_{latent_idx}.png"
            if not latent_path.is_file():
                missing.append(latent_path)
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ValueError(
            "--skip-mesh-generation requires an already-preprocessed prepared "
            f"frame; missing latent maps: {formatted}"
        )


def _prepare_legacy_h5_mesh_copy(scene_config, mesh_path: Path) -> Path:
    """Reproduce the legacy animation export's H5-only smooth-normal pass."""

    smooth_obj_keys = [
        obj_key
        for obj_key, obj_config in scene_config.objects.items()
        if obj_config.material.smooth_shading
    ]
    if not smooth_obj_keys:
        return mesh_path

    import numpy as np
    import trimesh

    source_split_dir = mesh_path.parent / "split"
    h5_mesh_root = mesh_path.parent / "h5_export_mesh"
    if h5_mesh_root.exists():
        raise ValueError(
            f"prepared-frame staging path already exists: {h5_mesh_root}"
        )
    h5_split_dir = h5_mesh_root / "split"
    shutil.copytree(source_split_dir, h5_split_dir)
    h5_scene_obj = h5_mesh_root / "scene.obj"
    if mesh_path.is_file():
        shutil.copy2(mesh_path, h5_scene_obj)

    for obj_key in smooth_obj_keys:
        obj_path = h5_split_dir / f"{obj_key}.obj"
        mesh = trimesh.load(str(obj_path), process=False, force="mesh")
        mesh.merge_vertices()
        mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(30))
        mesh.export(str(obj_path), include_normals=True)
    return h5_scene_obj


def generate_prepared_frame(
    frame_dir: str | Path,
    output_dir: str | Path,
    resolution: int = 256,
    spp: int = 4096,
    texture_crop_res: int = 192,
    texture_target_res: int = 128,
    texture_size: int = 64,
    min_views: int = 0,
    test_mode: bool = False,
    use_heightmap_format: bool = True,
    save_images: bool = False,
    save_blend: bool = False,
    include_background: bool = False,
    skip_mesh_generation: bool = False,
) -> PreparedScene:
    """Generate from a Blender-exported frame without re-transforming its meshes."""

    config_path, source_split_dir = validate_prepared_frame(frame_dir)
    frame_dir = config_path.parent
    if skip_mesh_generation:
        source_scene_config = load_scene_config(str(config_path))
        _validate_preprocessed_prepared_frame(source_scene_config, source_split_dir)
    output_dir = Path(output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(frame_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "prepared-frame output_dir must not be the source frame or one of "
            "its descendants"
        )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"prepared-frame output_dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                f"prepared-frame output_dir must be empty or not exist: {output_dir}"
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(frame_dir, output_dir, dirs_exist_ok=True)

    mesh_path = output_dir / "scene.obj"
    rendered_obj_path = output_dir / "rendered_scene.obj"
    output_image_path = output_dir / "rendered_image.png"
    staged_config_path = output_dir / config_path.name
    scene_config = load_scene_config(str(staged_config_path))
    _resolve_prepared_frame_asset_paths(scene_config, output_dir)
    if not skip_mesh_generation:
        generate_scene_mesh(
            scene_config,
            str(mesh_path),
            texture_crop_res=texture_crop_res,
            texture_target_res=texture_target_res,
            force_heightmap=use_heightmap_format,
            skip_transform=True,
        )

    render_results = scene_to_img(
        scene_config,
        str(mesh_path),
        str(rendered_obj_path),
        str(output_image_path),
        save_img=save_images,
        resolution=resolution,
        save_scene_obj=False,
        spp=spp,
        local_model=False,
        dump_blend_file=save_blend,
        skip_rendering=False,
        include_background=include_background,
    )
    h5_mesh_path = _prepare_legacy_h5_mesh_copy(scene_config, mesh_path)
    return build_prepared_scene(
        scene_config=scene_config,
        mesh_path=h5_mesh_path,
        render_results=render_results,
        min_available_view=min_views,
        test_mode=test_mode,
        texture_size=texture_size,
    )


def generate_prepared_scene(
    scene_config_path: str | Path,
    output_dir: str | Path,
    resolution: int = 256,
    spp: int = 4096,
    texture_crop_res: int = 192,
    texture_target_res: int = 128,
    texture_size: int = 64,
    min_views: int = 0,
    test_mode: bool = False,
    use_heightmap_format: bool = True,
    save_images: bool = False,
    save_blend: bool = False,
    include_background: bool = False,
    skip_mesh_generation: bool = False,
) -> PreparedScene:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "scene.obj"
    rendered_obj_path = output_dir / "rendered_scene.obj"
    output_image_path = output_dir / "rendered_image.png"

    scene_config = load_scene_config(str(scene_config_path))
    if not skip_mesh_generation:
        generate_scene_mesh(
            scene_config,
            str(mesh_path),
            texture_crop_res=texture_crop_res,
            texture_target_res=texture_target_res,
            force_heightmap=use_heightmap_format,
        )

    render_results = scene_to_img(
        scene_config,
        str(mesh_path),
        str(rendered_obj_path),
        str(output_image_path),
        save_img=save_images,
        resolution=resolution,
        save_scene_obj=False,
        spp=spp,
        local_model=False,
        dump_blend_file=save_blend,
        skip_rendering=False,
        include_background=include_background,
    )
    return build_prepared_scene(
        scene_config=scene_config,
        mesh_path=mesh_path,
        render_results=render_results,
        min_available_view=min_views,
        test_mode=test_mode,
        texture_size=texture_size,
    )


def generate_scene_config_from_template(
    template_path: str | Path,
    output_dir: str | Path,
    num_views: int,
    template_root: str | Path | None = None,
    texture_crop_res: int = 192,
    texture_target_res: int = 128,
    *,
    object_list_path: str | Path | None = None,
    object_root: str | Path | None = None,
    texture_list_path: str | Path | None = None,
    texture_root: str | Path | None = None,
    env_map_list_path: str | Path | None = None,
    env_map_root: str | Path | None = None,
    background_topology_debias: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_config_path = output_dir / "scene_config.json"
    mesh_path = output_dir / "scene.obj"

    template = load_scene_template(template_path, template_root=template_root)
    if background_topology_debias:
        template = apply_background_topology_debias(
            template, template_path=template_path
        )
    template = override_template_asset_paths(
        template,
        object_list_path=(
            Path(object_list_path).resolve() if object_list_path is not None else None
        ),
        object_root=Path(object_root).resolve() if object_root is not None else None,
        texture_list_path=(
            Path(texture_list_path).resolve() if texture_list_path is not None else None
        ),
        texture_root=Path(texture_root).resolve() if texture_root is not None else None,
        env_map_list_path=(
            Path(env_map_list_path).resolve() if env_map_list_path is not None else None
        ),
        env_map_root=Path(env_map_root).resolve() if env_map_root is not None else None,
    )
    object_list, texture_list, env_map_list = load_template_asset_lists(template)
    generate_scene_from_template(
        template,
        str(output_dir),
        object_list,
        texture_list,
        str(scene_config_path),
        str(mesh_path),
        num_views,
        texture_crop_res=texture_crop_res,
        texture_target_res=texture_target_res,
        env_map_list=env_map_list,
    )
    return scene_config_path


def generate_prepared_scene_from_template(
    template_path: str | Path,
    output_dir: str | Path,
    num_views: int = 4,
    template_root: str | Path | None = None,
    resolution: int = 256,
    spp: int = 4096,
    texture_crop_res: int = 192,
    texture_target_res: int = 128,
    texture_size: int = 64,
    min_views: int = 0,
    test_mode: bool = False,
    use_heightmap_format: bool = True,
    save_images: bool = False,
    save_blend: bool = False,
    include_background: bool = False,
    *,
    object_list_path: str | Path | None = None,
    object_root: str | Path | None = None,
    texture_list_path: str | Path | None = None,
    texture_root: str | Path | None = None,
    env_map_list_path: str | Path | None = None,
    env_map_root: str | Path | None = None,
    background_topology_debias: bool = False,
) -> PreparedScene:
    scene_config_path = generate_scene_config_from_template(
        template_path=template_path,
        output_dir=output_dir,
        num_views=num_views,
        template_root=template_root,
        object_list_path=object_list_path,
        object_root=object_root,
        texture_list_path=texture_list_path,
        texture_root=texture_root,
        env_map_list_path=env_map_list_path,
        env_map_root=env_map_root,
        background_topology_debias=background_topology_debias,
        texture_crop_res=texture_crop_res,
        texture_target_res=texture_target_res,
    )
    return generate_prepared_scene(
        scene_config_path=scene_config_path,
        output_dir=output_dir,
        resolution=resolution,
        spp=spp,
        texture_crop_res=texture_crop_res,
        texture_target_res=texture_target_res,
        texture_size=texture_size,
        min_views=min_views,
        test_mode=test_mode,
        use_heightmap_format=use_heightmap_format,
        save_images=save_images,
        save_blend=save_blend,
        include_background=include_background,
        skip_mesh_generation=True,
    )
