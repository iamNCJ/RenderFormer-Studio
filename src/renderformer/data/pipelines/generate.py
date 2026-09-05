from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Sequence

from renderformer.data.blender.pipeline import (
    generate_prepared_frame,
    generate_prepared_scene,
    generate_prepared_scene_from_template,
)
from renderformer.data.exporters.common import ExportResult, PreparedScene
from renderformer.data.exporters.v1 import export_v1_h5
from renderformer.data.exporters.v2 import export_v2_h5
from renderformer.data.schemas.export_profile import ExportFormat, ExportProfile
from renderformer.data.schemas.io import load_export_profile


@dataclass(frozen=True)
class ExportJob:
    name: str
    profile: ExportProfile
    output_path: Path


def _load_export_jobs(
    profile_paths: Sequence[str | Path],
    output_dir: Path,
) -> list[ExportJob]:
    jobs = []
    for profile_value in profile_paths:
        profile_path = Path(profile_value)
        profile = load_export_profile(profile_path)
        jobs.append(
            ExportJob(
                profile_path.stem,
                profile,
                output_dir / f"{profile_path.stem}.h5",
            )
        )
    return jobs


def _validate_fresh_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"prepared-frame output_dir must not be a symlink: {output_dir}")
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f"prepared-frame output_dir is not a directory: {output_dir}")
    if any(output_dir.iterdir()):
        raise ValueError(
            f"prepared-frame output_dir must be empty or not exist: {output_dir}"
        )


def run_exports(
    prepared_scene: PreparedScene,
    jobs: Sequence[ExportJob],
) -> list[ExportResult]:
    results: list[ExportResult] = []
    for job in jobs:
        if job.profile.format is ExportFormat.V1:
            result = export_v1_h5(prepared_scene, job.profile, job.output_path)
        elif job.profile.format is ExportFormat.V2:
            result = export_v2_h5(prepared_scene, job.profile, job.output_path)
        else:
            raise ValueError(f"Unsupported export format for job {job.name}: {job.profile.format}")
        results.append(result)
    return results


def run_blender_generate(
    scene_config_path: str | Path | None,
    output_dir: str | Path,
    profile_paths: Sequence[str | Path],
    template_path: str | Path | None = None,
    template_root: str | Path | None = None,
    num_views: int = 4,
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
    *,
    prepared_frame_path: str | Path | None = None,
    object_list_path: str | Path | None = None,
    object_root: str | Path | None = None,
    texture_list_path: str | Path | None = None,
    texture_root: str | Path | None = None,
    env_map_list_path: str | Path | None = None,
    env_map_root: str | Path | None = None,
    background_topology_debias: bool = False,
) -> list[ExportResult]:
    output_dir = Path(output_dir).expanduser().resolve()
    asset_overrides = (
        object_list_path,
        object_root,
        texture_list_path,
        texture_root,
        env_map_list_path,
        env_map_root,
    )
    if template_path is None and any(value is not None for value in asset_overrides):
        raise ValueError("Template asset overrides require template_path")
    if template_path is None and background_topology_debias:
        raise ValueError("background_topology_debias requires template_path")
    source_count = sum(
        source is not None
        for source in (scene_config_path, template_path, prepared_frame_path)
    )
    if source_count != 1:
        raise ValueError(
            "Exactly one of scene_config_path, template_path, or "
            "prepared_frame_path must be provided"
        )
    export_jobs = _load_export_jobs(profile_paths, output_dir)
    if prepared_frame_path is not None:
        frame_dir = Path(prepared_frame_path).expanduser().resolve()
        if output_dir == frame_dir or output_dir.is_relative_to(frame_dir):
            raise ValueError(
                "prepared-frame output_dir must not be the source frame or one of "
                "its descendants"
            )
        _validate_fresh_output_dir(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.staging-",
                dir=output_dir.parent,
            )
        )
        try:
            prepared = generate_prepared_frame(
                frame_dir=frame_dir,
                output_dir=staging_dir,
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
                skip_mesh_generation=skip_mesh_generation,
            )
            results = run_exports(
                prepared,
                [
                    ExportJob(
                        job.name,
                        job.profile,
                        staging_dir / job.output_path.name,
                    )
                    for job in export_jobs
                ],
            )
            if output_dir.exists():
                output_dir.rmdir()
            staging_dir.replace(output_dir)
            return [
                ExportResult(
                    output_path=output_dir / result.output_path.name,
                    metadata=result.metadata,
                )
                for result in results
            ]
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
    elif template_path is not None:
        prepared = generate_prepared_scene_from_template(
            template_path=Path(template_path),
            template_root=Path(template_root) if template_root is not None else None,
            object_list_path=object_list_path,
            object_root=object_root,
            texture_list_path=texture_list_path,
            texture_root=texture_root,
            env_map_list_path=env_map_list_path,
            env_map_root=env_map_root,
            background_topology_debias=background_topology_debias,
            output_dir=output_dir,
            num_views=num_views,
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
        )
    elif scene_config_path is not None:
        prepared = generate_prepared_scene(
            scene_config_path=Path(scene_config_path),
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
            skip_mesh_generation=skip_mesh_generation,
        )
    else:
        raise AssertionError("validated generation source was not dispatched")
    return run_exports(prepared, export_jobs)
