from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from dacite import Config, from_dict
from jsoncomment import JsonComment

from renderformer.data.blender.template_dataclass import Material, SceneTemplate


_BACKGROUND_NORMAL_AXIS = {
    "plane.obj": "z",
    "wall0.obj": "y",
    "wall1.obj": "x",
    "wall2.obj": "x",
    "wall2-flip.obj": "x",
}
_ONE_SIDED_TRANSLATION_CORRECTIONS = {
    ("wall0.obj", "y", (-0.25, -0.15)): [-0.25, 0.25],
    ("wall1.obj", "x", (-0.25, -0.15)): [-0.25, 0.25],
    ("wall1.obj", "y", (-0.25, -0.15)): [-0.25, 0.25],
    ("wall1.obj", "y", (-0.25, 0.05)): [-0.25, 0.25],
}
_RAISED_FLOOR_TEMPLATE_PATHS = {
    Path("v1") / dataset / "more-tri" / template_name
    for dataset in (
        "250416_high_res_fixed_bsdf_dev",
        "250428_oxl_high_res",
    )
    for template_name in (
        "plane-multi-object.jsonc",
        "wall-multi-object.jsonc",
    )
}


def _resolve_path(path: str | None, template_root: Path) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(template_root / path_obj)


def infer_template_root(template_path: str | Path) -> Path:
    template_path = Path(template_path).resolve()
    for parent in template_path.parents:
        if parent.name == "templates":
            return parent.parent
    return template_path.parent


def packaged_template_root() -> Path:
    return Path(__file__).resolve().parents[1] / "templates"


def _resolve_manifest_path(manifest_name: str | Path) -> Path:
    manifest_path = Path(manifest_name)
    if manifest_path.is_absolute():
        return manifest_path
    if manifest_path.suffix != ".json":
        manifest_path = manifest_path.with_suffix(".json")
    return packaged_template_root() / manifest_path


def _packaged_path_for_legacy_template(dataset_name: str, legacy_path: str) -> str:
    legacy = Path(legacy_path)
    if legacy.parts and legacy.parts[0] == "templates":
        legacy = Path(*legacy.parts[1:])
    return str(Path("v1") / dataset_name / legacy)


def _expand_dataset_manifest(manifest: dict) -> dict:
    if "templates" in manifest:
        return manifest

    expanded = deepcopy(manifest)
    templates = []
    for dataset in expanded.get("datasets", []):
        dataset_name = dataset["name"]
        for template in dataset["templates"]:
            legacy_path = template["legacy_path"]
            packaged_path = template.get("path")
            if packaged_path is None:
                packaged_path = _packaged_path_for_legacy_template(
                    dataset_name, legacy_path
                )
            templates.append(
                {
                    "path": packaged_path,
                    "legacy_path": legacy_path,
                    "dataset": dataset_name,
                    "group": dataset_name,
                    "partition_key": dataset.get("partition_key"),
                    "resolution": dataset.get("resolution"),
                    "weight": template.get("weight", 1),
                }
            )
    expanded["templates"] = templates
    return expanded


DEFAULT_TRAINING_MANIFEST = "training_v2_200m_curriculum_manifest.json"


def load_training_template_manifest(
    manifest_name: str | Path = DEFAULT_TRAINING_MANIFEST,
) -> dict:
    manifest_path = _resolve_manifest_path(manifest_name)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return _expand_dataset_manifest(manifest)


def iter_training_template_entries(
    manifest_name: str | Path = DEFAULT_TRAINING_MANIFEST,
) -> list[dict]:
    manifest = load_training_template_manifest(manifest_name)
    return list(manifest["templates"])


def iter_training_template_paths(
    manifest_name: str | Path = DEFAULT_TRAINING_MANIFEST,
) -> list[Path]:
    root = packaged_template_root()
    return [root / entry["path"] for entry in iter_training_template_entries(manifest_name)]


def resolve_template_asset_paths(template: SceneTemplate, template_root: str | Path) -> SceneTemplate:
    template_root = Path(template_root).resolve()
    resolved = deepcopy(template)

    for obj in resolved.background_scene:
        obj.mesh_path = str(_resolve_path(obj.mesh_path, template_root))

    placement = resolved.object_placement
    placement.bounding_mesh = str(_resolve_path(placement.bounding_mesh, template_root))
    placement.objects.list_path = str(_resolve_path(placement.objects.list_path, template_root))
    placement.objects.base_path = str(_resolve_path(placement.objects.base_path, template_root))

    resolved.camera.shell_path = str(_resolve_path(resolved.camera.shell_path, template_root))
    resolved.lighting.mesh_path = str(_resolve_path(resolved.lighting.mesh_path, template_root))
    resolved.lighting.shell_path = str(_resolve_path(resolved.lighting.shell_path, template_root))

    material = resolved.material
    material.texture_map_list_path = _resolve_path(material.texture_map_list_path, template_root)
    material.texture_map_base_path = _resolve_path(material.texture_map_base_path, template_root)

    if resolved.env_map is not None:
        resolved.env_map.list_path = str(_resolve_path(resolved.env_map.list_path, template_root))
        resolved.env_map.base_path = str(_resolve_path(resolved.env_map.base_path, template_root))

    return resolved


def override_template_asset_paths(
    template: SceneTemplate,
    *,
    object_list_path: str | Path | None = None,
    object_root: str | Path | None = None,
    texture_list_path: str | Path | None = None,
    texture_root: str | Path | None = None,
    env_map_list_path: str | Path | None = None,
    env_map_root: str | Path | None = None,
) -> SceneTemplate:
    """Return a copy with runtime asset locations applied.

    Asset overrides are deliberately kept outside ``SceneTemplate`` so the
    packaged template schema remains a description of scene distributions,
    not a record of one machine's storage layout.
    """

    overridden = deepcopy(template)
    objects = overridden.object_placement.objects
    if object_list_path is not None:
        objects.list_path = str(object_list_path)
    if object_root is not None:
        objects.base_path = str(object_root)

    material = overridden.material
    if texture_list_path is not None:
        material.texture_map_list_path = str(texture_list_path)
    if texture_root is not None:
        material.texture_map_base_path = str(texture_root)

    if env_map_list_path is not None or env_map_root is not None:
        if overridden.env_map is None:
            raise ValueError("Environment-map overrides require a template with env_map")
        if env_map_list_path is not None:
            overridden.env_map.list_path = str(env_map_list_path)
        if env_map_root is not None:
            overridden.env_map.base_path = str(env_map_root)

    return overridden


def apply_background_topology_debias(
    template: SceneTemplate, *, template_path: str | Path
) -> SceneTemplate:
    """Return a corrected copy of a V1 room template.

    The rules reproduce the May 2025 background-render policy without retaining
    a second copy of every template. Background rotation is derived from each
    mesh's normal axis, while placement edits are limited to the exact ranges
    corrected by the historical templates. The raised-floor repair is restricted
    to the canonical packaged ``more-tri`` plane and wall templates; copied or
    renamed external templates still receive mesh-based corrections, but are not
    implicitly classified as those historical templates.
    """

    corrected = deepcopy(template)
    source_path = Path(template_path).resolve()
    try:
        packaged_relative_path = source_path.relative_to(
            packaged_template_root().resolve()
        )
    except ValueError:
        packaged_relative_path = None
    repair_raised_floor = packaged_relative_path in _RAISED_FLOOR_TEMPLATE_PATHS
    for background in corrected.background_scene:
        mesh_path = Path(background.mesh_path)
        mesh_name = mesh_path.name
        try:
            normal_axis = _BACKGROUND_NORMAL_AXIS[mesh_name]
        except KeyError as exc:
            raise ValueError(
                "background topology de-bias has no mesh-orientation rule for "
                f"{background.mesh_path!r}"
            ) from exc

        rotation = background.transform.rotation
        if rotation[normal_axis] != [0, 0]:
            rotation[normal_axis] = [-180, 180]
        for axis in ("x", "y", "z"):
            if axis != normal_axis and rotation[axis] == [-0.5, 0.5]:
                rotation[axis] = [-3, 3]

        translation = background.transform.translation
        for axis in ("x", "y", "z"):
            replacement = _ONE_SIDED_TRANSLATION_CORRECTIONS.get(
                (mesh_name, axis, tuple(translation[axis]))
            )
            if replacement is not None:
                translation[axis] = list(replacement)
        if (
            mesh_name == "plane.obj"
            and repair_raised_floor
            and translation["z"] == [0.15, 0.25]
        ):
            translation["z"] = [-0.05, 0.25]

        if mesh_name == "wall2.obj":
            background.mesh_path = str(mesh_path.with_name("wall2-flip.obj"))

    return corrected


def load_scene_template(
    template_path: str | Path,
    template_root: str | Path | None = None,
    *,
    resolve_assets: bool = True,
) -> SceneTemplate:
    parser = JsonComment()
    template_path = Path(template_path)
    with template_path.open("r", encoding="utf-8") as f:
        template_dict = parser.load(f)
    template = from_dict(data_class=SceneTemplate, data=template_dict, config=Config(strict=True))
    if not resolve_assets:
        return template
    root = Path(template_root).resolve() if template_root is not None else infer_template_root(template_path)
    return resolve_template_asset_paths(template, root)


def read_list_file(path: str | Path) -> list[str]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            return [
                line
                for raw_line in f
                if (line := raw_line.strip()) and not line.startswith("#")
            ]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Asset list not found: {path}. Supply a runtime asset override or run "
            "`renderformer data validate-assets` to inspect the template requirements."
        ) from exc


def sampled_svbrdf_material_types(
    material: Material,
    *,
    exclude_transmission: bool = False,
) -> set[str]:
    """Return SVBRDF types that the material sampler can select with nonzero weight."""

    from renderformer.data.blender.material_types import (
        ALL_MATERIAL_TYPES,
        MATERIAL_TYPE_HOMO_MEASURED_BRDF,
        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
        supports_svbrdf,
    )

    available_types = [
        material_type
        for material_type in ALL_MATERIAL_TYPES
        if material_type != MATERIAL_TYPE_HOMO_MEASURED_BRDF
    ]
    if material.allowed_material_types is not None:
        available_types = [
            material_type
            for material_type in available_types
            if material_type in material.allowed_material_types
        ]
    if exclude_transmission:
        available_types = [
            material_type
            for material_type in available_types
            if material_type != MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION
        ]

    probabilities = material.material_type_probabilities
    if probabilities is None:
        return {
            material_type
            for material_type in available_types
            if supports_svbrdf(material_type)
        }

    weights = {
        material_type: probabilities[material_type]
        for material_type in available_types
        if material_type in probabilities
    }
    types_without_weight = [
        material_type for material_type in available_types if material_type not in weights
    ]
    if types_without_weight:
        specified_weight = sum(weights.values())
        if specified_weight >= 1.0:
            fallback_weight = 1.0 / len(types_without_weight)
        else:
            fallback_weight = (1.0 - specified_weight) / len(types_without_weight)
        weights.update(
            {material_type: fallback_weight for material_type in types_without_weight}
        )

    if weights and sum(weights.values()) == 0:
        weights = {material_type: 1.0 for material_type in available_types}

    return {
        material_type
        for material_type, weight in weights.items()
        if weight > 0 and supports_svbrdf(material_type)
    }


def template_sampled_svbrdf_material_types(template: SceneTemplate) -> set[str]:
    """Return SVBRDF types reachable in any object-sampling context."""

    material_types: set[str] = set()
    if template.object_placement.objects.count["max"] > 0:
        material_types.update(sampled_svbrdf_material_types(template.material))
    if template.background_scene:
        material_types.update(
            sampled_svbrdf_material_types(
                template.material,
                exclude_transmission=True,
            )
        )
    return material_types


def template_requires_texture_assets(template: SceneTemplate) -> bool:
    """Whether generation needs the template's texture list and root.

    A legacy template with no texture configuration falls back to homogeneous
    diffuse/specular sampling. A configured list is only required when an
    SVBRDF type has nonzero sampling probability.
    """

    material = template.material
    has_explicit_texture_sampling = (
        material.texture_map_list_path is not None
        or material.texture_map_base_path is not None
        or material.allowed_material_types is not None
        or material.material_type_probabilities is not None
    )
    return bool(template_sampled_svbrdf_material_types(template)) and has_explicit_texture_sampling


def load_template_asset_lists(template: SceneTemplate) -> tuple[list[str], list[str], list[str]]:
    objects = template.object_placement.objects
    object_list = read_list_file(objects.list_path) if objects.count["max"] > 0 else []
    if objects.count["max"] > 0 and not object_list:
        raise ValueError(f"Required object asset list is empty: {objects.list_path}")

    material = template.material
    if template_requires_texture_assets(template):
        if material.texture_map_list_path is None or material.texture_map_base_path is None:
            raise ValueError(
                "Template can sample an SVBRDF material but does not configure both "
                "texture_map_list_path and texture_map_base_path"
            )
        texture_list = read_list_file(material.texture_map_list_path)
        if not texture_list:
            raise ValueError(
                f"Required texture asset list is empty: {material.texture_map_list_path}"
            )
    else:
        texture_list = []

    if template.env_map is not None:
        env_map_list = read_list_file(template.env_map.list_path)
        if not env_map_list:
            raise ValueError(
                f"Required environment-map asset list is empty: {template.env_map.list_path}"
            )
    else:
        env_map_list = []
    return object_list, texture_list, env_map_list


def normalize_template_ratios(ratio_dict: dict[str, float]) -> dict[str, float]:
    total = sum(ratio_dict.values())
    if total <= 0:
        raise ValueError("template ratio weights must sum to a positive value")
    return {key: value / total for key, value in ratio_dict.items()}
