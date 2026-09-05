"""Validation for external assets referenced by packaged scene templates."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Sequence

from renderformer.data.blender.material_types import (
    MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
)
from renderformer.data.blender.template_dataclass import SceneTemplate
from renderformer.data.blender.templates import (
    infer_template_root,
    iter_training_template_paths,
    load_scene_template,
    read_list_file,
    resolve_template_asset_paths,
    template_sampled_svbrdf_material_types,
    template_requires_texture_assets,
)


AssetKind = Literal["object", "texture", "env-map"]

FINAL_TRAINING_MANIFESTS = (
    "training_v1_final_manifest.json",
    "training_v2_200m_curriculum_manifest.json",
)


@dataclass(frozen=True)
class AssetLocation:
    kind: AssetKind
    list_path: Path
    root_path: Path


@dataclass(frozen=True)
class AssetManifest:
    path: Path
    collections: dict[str, AssetLocation]


@dataclass(frozen=True)
class AssetValidationReport:
    template_count: int
    collection_count: int
    checked_entry_count: int
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass
class _CollectionRequirement:
    kind: AssetKind
    placeholder_list_path: str
    placeholder_root_path: str
    list_path: Path
    root_path: Path
    source_templates: set[Path] = field(default_factory=set)
    texture_material_types: set[str] = field(default_factory=set)


def _manifest_path(value: object, *, field_name: str, base_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Asset manifest field {field_name!r} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def load_asset_manifest(path: str | Path) -> AssetManifest:
    """Load placeholder-to-location mappings from a schema-v1 JSON manifest.

    Relative ``list_path`` and ``root_path`` values are resolved against the
    directory containing the asset manifest, never against the process CWD.
    """

    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Asset manifest must be a JSON object: {manifest_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported asset manifest schema_version in {manifest_path}; expected 1"
        )
    collections_payload = payload.get("collections")
    if not isinstance(collections_payload, dict):
        raise ValueError(f"Asset manifest 'collections' must be an object: {manifest_path}")

    collections: dict[str, AssetLocation] = {}
    for placeholder, raw_location in collections_payload.items():
        if not isinstance(placeholder, str) or not placeholder:
            raise ValueError("Asset manifest collection keys must be non-empty strings")
        if not isinstance(raw_location, dict):
            raise ValueError(f"Asset collection {placeholder!r} must be an object")
        expected_fields = {"kind", "list_path", "root_path"}
        unknown_fields = set(raw_location) - expected_fields
        missing_fields = expected_fields - set(raw_location)
        if unknown_fields or missing_fields:
            raise ValueError(
                f"Asset collection {placeholder!r} fields mismatch; "
                f"missing={sorted(missing_fields)}, unknown={sorted(unknown_fields)}"
            )
        kind = raw_location["kind"]
        if kind not in {"object", "texture", "env-map"}:
            raise ValueError(
                f"Asset collection {placeholder!r} has unsupported kind {kind!r}"
            )
        collections[placeholder] = AssetLocation(
            kind=kind,
            list_path=_manifest_path(
                raw_location["list_path"],
                field_name=f"collections.{placeholder}.list_path",
                base_path=manifest_path.parent,
            ),
            root_path=_manifest_path(
                raw_location["root_path"],
                field_name=f"collections.{placeholder}.root_path",
                base_path=manifest_path.parent,
            ),
        )
    return AssetManifest(path=manifest_path, collections=collections)


def final_training_template_paths(
    manifest_names: Sequence[str | Path] = FINAL_TRAINING_MANIFESTS,
) -> list[Path]:
    """Return the de-duplicated template union for the final V1/V2 manifests."""

    return sorted(
        {
            path.resolve()
            for manifest_name in manifest_names
            for path in iter_training_template_paths(manifest_name)
        }
    )


def _fixed_asset_paths(template: SceneTemplate) -> list[tuple[str, Path]]:
    paths = [
        (f"background mesh {obj.name!r}", Path(obj.mesh_path))
        for obj in template.background_scene
    ]
    paths.extend(
        [
            ("object-placement bounding mesh", Path(template.object_placement.bounding_mesh)),
            ("camera shell", Path(template.camera.shell_path)),
            ("lighting mesh", Path(template.lighting.mesh_path)),
            ("lighting shell", Path(template.lighting.shell_path)),
        ]
    )
    return paths


def _safe_relative_entry(entry: str) -> bool:
    path = Path(entry)
    return not path.is_absolute() and ".." not in path.parts


def _texture_files_for(material_types: set[str]) -> tuple[set[str], bool]:
    required = {"roughness.png", "normal.png"}
    needs_base_color_alternative = False
    if MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR in material_types:
        required.update({"diffuse.png", "specular.png"})
    if MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS in material_types:
        required.add("metallic.png")
        needs_base_color_alternative = True
    return required, needs_base_color_alternative


def validate_template_assets(
    template_paths: Sequence[str | Path],
    *,
    asset_manifest_path: str | Path | None = None,
    template_root: str | Path | None = None,
    check_entries: bool = True,
) -> AssetValidationReport:
    """Validate fixed and external assets needed by a set of templates.

    Collection entries are checked from their metadata lists; this function
    does not discover assets by traversing collection directories.
    """

    paths = sorted({Path(path).expanduser().resolve() for path in template_paths})
    if not paths:
        raise ValueError("At least one template path is required for asset validation")
    manifest = load_asset_manifest(asset_manifest_path) if asset_manifest_path else None
    manifest_collections = manifest.collections if manifest is not None else {}
    explicit_template_root = (
        Path(template_root).expanduser().resolve() if template_root is not None else None
    )

    issues: list[str] = []
    issue_keys: set[tuple[str, str]] = set()
    collections: dict[tuple[AssetKind, Path, Path], _CollectionRequirement] = {}

    def add_issue(code: str, identity: str | Path, message: str) -> None:
        key = (code, str(identity))
        if key not in issue_keys:
            issue_keys.add(key)
            issues.append(message)

    def add_collection(
        *,
        kind: AssetKind,
        placeholder_list_path: str,
        placeholder_root_path: str,
        default_list_path: str | None,
        default_root_path: str | None,
        source_template: Path,
        texture_material_types: set[str] | None = None,
    ) -> None:
        location = manifest_collections.get(placeholder_list_path)
        if location is not None and location.kind != kind:
            add_issue(
                "manifest-kind",
                placeholder_list_path,
                f"{source_template}: asset manifest maps {placeholder_list_path!r} as "
                f"{location.kind!r}, but the template requires {kind!r}",
            )
            return
        if location is not None:
            list_path = location.list_path
            root_path = location.root_path
        else:
            if default_list_path is None or default_root_path is None:
                add_issue(
                    "missing-config",
                    f"{source_template}:{kind}",
                    f"{source_template}: required {kind} assets do not configure both a list and root",
                )
                return
            list_path = Path(default_list_path)
            root_path = Path(default_root_path)

        key = (kind, list_path, root_path)
        requirement = collections.get(key)
        if requirement is None:
            requirement = _CollectionRequirement(
                kind=kind,
                placeholder_list_path=placeholder_list_path,
                placeholder_root_path=placeholder_root_path,
                list_path=list_path,
                root_path=root_path,
            )
            collections[key] = requirement
        requirement.source_templates.add(source_template)
        if texture_material_types:
            requirement.texture_material_types.update(texture_material_types)

    for template_path in paths:
        if not template_path.is_file():
            add_issue(
                "template",
                template_path,
                f"Template file is missing: {template_path}",
            )
            continue
        raw_template = load_scene_template(template_path, resolve_assets=False)
        resolved_root = explicit_template_root or infer_template_root(template_path)
        resolved_template = resolve_template_asset_paths(raw_template, resolved_root)

        for label, fixed_path in _fixed_asset_paths(resolved_template):
            if not fixed_path.is_file():
                add_issue(
                    "fixed",
                    fixed_path,
                    f"{template_path}: missing {label}: {fixed_path}",
                )

        raw_objects = raw_template.object_placement.objects
        resolved_objects = resolved_template.object_placement.objects
        if raw_objects.count["max"] > 0:
            add_collection(
                kind="object",
                placeholder_list_path=raw_objects.list_path,
                placeholder_root_path=raw_objects.base_path,
                default_list_path=resolved_objects.list_path,
                default_root_path=resolved_objects.base_path,
                source_template=template_path,
            )

        if template_requires_texture_assets(raw_template):
            raw_material = raw_template.material
            resolved_material = resolved_template.material
            placeholder = raw_material.texture_map_list_path
            placeholder_root = raw_material.texture_map_base_path
            if placeholder is None or placeholder_root is None:
                add_issue(
                    "missing-config",
                    f"{template_path}:texture",
                    f"{template_path}: SVBRDF sampling requires texture list and root fields",
                )
            else:
                add_collection(
                    kind="texture",
                    placeholder_list_path=placeholder,
                    placeholder_root_path=placeholder_root,
                    default_list_path=resolved_material.texture_map_list_path,
                    default_root_path=resolved_material.texture_map_base_path,
                    source_template=template_path,
                    texture_material_types=template_sampled_svbrdf_material_types(
                        raw_template
                    ),
                )

        if raw_template.env_map is not None:
            assert resolved_template.env_map is not None
            add_collection(
                kind="env-map",
                placeholder_list_path=raw_template.env_map.list_path,
                placeholder_root_path=raw_template.env_map.base_path,
                default_list_path=resolved_template.env_map.list_path,
                default_root_path=resolved_template.env_map.base_path,
                source_template=template_path,
            )

    checked_entry_count = 0
    for requirement in collections.values():
        label = (
            f"{requirement.kind} collection {requirement.placeholder_list_path!r} "
            f"(root placeholder {requirement.placeholder_root_path!r})"
        )
        list_exists = requirement.list_path.is_file()
        root_exists = requirement.root_path.is_dir()
        if not list_exists:
            add_issue(
                "list",
                requirement.list_path,
                f"Missing list for {label}: {requirement.list_path}",
            )
        if not root_exists:
            add_issue(
                "root",
                requirement.root_path,
                f"Missing root directory for {label}: {requirement.root_path}",
            )
        if not list_exists:
            continue

        entries = read_list_file(requirement.list_path)
        if not entries:
            add_issue(
                "empty-list",
                requirement.list_path,
                f"Asset list is empty for {label}: {requirement.list_path}",
            )
            continue
        if not check_entries or not root_exists:
            continue

        required_texture_files, needs_base_color = _texture_files_for(
            requirement.texture_material_types
        )
        for entry in entries:
            checked_entry_count += 1
            entry_identity = f"{requirement.list_path}:{entry}"
            if not _safe_relative_entry(entry):
                add_issue(
                    "entry-path",
                    entry_identity,
                    f"{label} entry must be a relative path without '..': {entry!r}",
                )
                continue

            if requirement.kind == "object":
                object_path = requirement.root_path / entry
                if not object_path.is_file():
                    add_issue(
                        "object-entry",
                        object_path,
                        f"Missing object listed by {requirement.list_path}: {object_path}",
                    )
            elif requirement.kind == "env-map":
                if entry.lower().endswith(".exr"):
                    add_issue(
                        "env-suffix",
                        entry_identity,
                        f"Environment-map list entries omit '.exr' because runtime appends it: {entry!r}",
                    )
                    continue
                env_path = requirement.root_path / f"{entry}.exr"
                if not env_path.is_file():
                    add_issue(
                        "env-entry",
                        env_path,
                        f"Missing environment map listed by {requirement.list_path}: {env_path}",
                    )
            else:
                texture_dir = requirement.root_path / entry
                if not texture_dir.is_dir():
                    add_issue(
                        "texture-entry",
                        texture_dir,
                        f"Missing texture directory listed by {requirement.list_path}: {texture_dir}",
                    )
                    continue
                for filename in sorted(required_texture_files):
                    texture_path = texture_dir / filename
                    if not texture_path.is_file():
                        add_issue(
                            "texture-file",
                            texture_path,
                            f"Missing required texture map: {texture_path}",
                        )
                if needs_base_color and not any(
                    (texture_dir / filename).is_file()
                    for filename in ("basecolor.png", "base_color.png")
                ):
                    add_issue(
                        "texture-base-color",
                        texture_dir,
                        f"Texture directory needs basecolor.png or base_color.png: {texture_dir}",
                    )

    return AssetValidationReport(
        template_count=len(paths),
        collection_count=len(collections),
        checked_entry_count=checked_entry_count,
        issues=tuple(issues),
    )
