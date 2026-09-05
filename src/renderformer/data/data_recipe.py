"""Strict, portable data-generation recipes and deterministic batch execution.

The recipes in ``configs/data`` are release receipts, not claims that the
historical byte stream can be regenerated.  Historical workers did not record
their stop count, random seed, or list order, so every new run must provide an
explicit sample count and seed.  The resulting task manifest is the source of
truth for that run.
"""

from __future__ import annotations

from bisect import bisect
from dataclasses import asdict, dataclass
import hashlib
from itertools import accumulate
import json
import math
from multiprocessing import current_process, get_context
import os
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from renderformer.data.schemas.io import load_export_profile, load_mapping


_RECIPE_KEYS = {
    "schema_version",
    "recipe_id",
    "family",
    "partition_key",
    "export_profile",
    "assets",
    "template_corrections",
    "render",
    "templates",
    "postprocess",
    "historical",
    "known_gaps",
    "artifacts",
}
_TEMPLATE_CORRECTION_KEYS = {
    "background_topology_debias",
    "env_map_channel_shuffle",
}
_RENDER_KEYS = {
    "resolution",
    "num_views",
    "spp",
    "texture_size",
    "texture_crop_resolution",
    "texture_target_resolution",
    "min_views",
    "include_background",
    "use_heightmap_format",
}
_V2_POSTPROCESS_KEYS = {
    "enabled",
    "texture_encoder",
    "model_id",
    "revision",
    "subfolder",
    "torch_dtype",
    "triangle_batch_size",
    "keep_raw",
    "drop_keys",
    "process_env_map",
}
_KNOWN_GAPS = ("sample_budget", "seed", "list_order")
_RECIPE_ID_RE = re.compile(r"^(v1|v2)/[a-z0-9][a-z0-9_-]*$")
_TASK_PLANNER_VERSION = "weighted-random-v2-affine32"
_MAX_TASKS = 2**32


def _expect_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return dict(value)


def _expect_exact_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        raise ValueError(
            f"{context} fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _expect_bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a boolean")
    return value


def _expect_positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _expect_nonnegative_int(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _expect_relative_path(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must stay below the repository or output root")
    return path.as_posix()


def _find_repository_root(recipe_path: Path) -> Path:
    for candidate in (recipe_path.parent, *recipe_path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "renderformer" / "data"
        ).is_dir():
            return candidate
    raise ValueError(
        f"Cannot locate repository root from data recipe {recipe_path}; "
        "run recipes from a source checkout"
    )


@dataclass(frozen=True)
class TemplateSpec:
    path: str
    weight: float


@dataclass(frozen=True)
class RenderSpec:
    resolution: int
    num_views: int
    spp: int
    texture_size: int
    texture_crop_resolution: int | None
    texture_target_resolution: int | None
    min_views: int
    include_background: bool
    use_heightmap_format: bool


@dataclass(frozen=True)
class TemplateCorrectionsSpec:
    background_topology_debias: bool
    env_map_channel_shuffle: bool


@dataclass(frozen=True)
class PostprocessSpec:
    enabled: bool
    texture_encoder: str | None = None
    model_id: str | None = None
    revision: str | None = None
    subfolder: str | None = None
    torch_dtype: str | None = None
    triangle_batch_size: int | None = None
    keep_raw: bool = False
    drop_keys: tuple[str, ...] = ()
    process_env_map: bool = False

    def encoder_config(self) -> dict[str, Any]:
        if not self.enabled:
            raise ValueError("V1 recipes do not have a texture postprocess encoder")
        return {
            "type": self.texture_encoder,
            "model_id": self.model_id,
            "revision": self.revision,
            "subfolder": self.subfolder,
            "torch_dtype": self.torch_dtype,
            "triangle_batch_size": self.triangle_batch_size,
        }


@dataclass(frozen=True)
class ArtifactSource:
    recipe: str
    historical_rows: int | None


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    path: str
    sources: tuple[ArtifactSource, ...]
    total_rows: int | None


@dataclass(frozen=True)
class DataRecipe:
    path: Path
    repository_root: Path
    schema_version: int
    recipe_id: str
    family: str
    partition_key: str
    export_profile: str
    require_external_manifest: bool
    template_corrections: TemplateCorrectionsSpec
    render: RenderSpec
    templates: tuple[TemplateSpec, ...]
    postprocess: PostprocessSpec
    artifacts: tuple[ArtifactSpec, ...]
    known_gaps: tuple[str, ...]

    @property
    def export_profile_path(self) -> Path:
        return self.repository_root / self.export_profile

    def template_path(self, template: TemplateSpec) -> Path:
        return self.repository_root / template.path

    @property
    def weight_sum(self) -> float:
        return sum(template.weight for template in self.templates)

    def artifact(self, name: str) -> ArtifactSpec:
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        choices = ", ".join(artifact.name for artifact in self.artifacts)
        raise KeyError(f"Unknown artifact {name!r} for {self.recipe_id}; choose {choices}")

    def describe(self) -> dict[str, Any]:
        return {
            "recipe": self.recipe_id,
            "family": self.family,
            "partition_key": self.partition_key,
            "export_profile": self.export_profile,
            "background_topology_debias": (
                self.template_corrections.background_topology_debias
            ),
            "env_map_channel_shuffle": (
                self.template_corrections.env_map_channel_shuffle
            ),
            "resolution": self.render.resolution,
            "num_views": self.render.num_views,
            "spp": self.render.spp,
            "texture_size": self.render.texture_size,
            "template_count": len(self.templates),
            "template_weight_sum": self.weight_sum,
            "postprocess": self.postprocess.texture_encoder or "none",
            "historical_artifacts": {
                artifact.name: artifact.path for artifact in self.artifacts
            },
            "runtime_output": "OUTPUT_DIR/paths.txt (new seeded generation)",
            "historical_equivalence": False,
            "known_gaps": list(self.known_gaps),
        }


def _parse_template_corrections(
    raw: object, family: str
) -> TemplateCorrectionsSpec:
    data = _expect_mapping(raw, "template_corrections")
    _expect_exact_keys(
        data, _TEMPLATE_CORRECTION_KEYS, "template_corrections"
    )
    background_topology_debias = _expect_bool(
        data["background_topology_debias"],
        "template_corrections.background_topology_debias",
    )
    expected_background_topology_debias = family == "v1"
    if background_topology_debias is not expected_background_topology_debias:
        raise ValueError(
            "template_corrections.background_topology_debias must be "
            f"{str(expected_background_topology_debias).lower()} for {family} recipes"
        )
    env_map_channel_shuffle = _expect_bool(
        data["env_map_channel_shuffle"],
        "template_corrections.env_map_channel_shuffle",
    )
    expected_env_map_channel_shuffle = family == "v2"
    if env_map_channel_shuffle is not expected_env_map_channel_shuffle:
        raise ValueError(
            "template_corrections.env_map_channel_shuffle must be "
            f"{str(expected_env_map_channel_shuffle).lower()} for {family} recipes"
        )
    return TemplateCorrectionsSpec(
        background_topology_debias=background_topology_debias,
        env_map_channel_shuffle=env_map_channel_shuffle,
    )


def _parse_render(raw: object, family: str) -> RenderSpec:
    data = _expect_mapping(raw, "render")
    _expect_exact_keys(data, _RENDER_KEYS, "render")
    crop_resolution = data["texture_crop_resolution"]
    target_resolution = data["texture_target_resolution"]
    if family == "v1":
        if crop_resolution is not None or target_resolution is not None:
            raise ValueError(
                "V1 historical recipes must record unused texture crop/target values as null"
            )
    else:
        crop_resolution = _expect_positive_int(
            crop_resolution, "render.texture_crop_resolution"
        )
        target_resolution = _expect_positive_int(
            target_resolution, "render.texture_target_resolution"
        )
    return RenderSpec(
        resolution=_expect_positive_int(data["resolution"], "render.resolution"),
        num_views=_expect_positive_int(data["num_views"], "render.num_views"),
        spp=_expect_positive_int(data["spp"], "render.spp"),
        texture_size=_expect_positive_int(
            data["texture_size"], "render.texture_size"
        ),
        texture_crop_resolution=crop_resolution,
        texture_target_resolution=target_resolution,
        min_views=_expect_nonnegative_int(data["min_views"], "render.min_views"),
        include_background=_expect_bool(
            data["include_background"], "render.include_background"
        ),
        use_heightmap_format=_expect_bool(
            data["use_heightmap_format"], "render.use_heightmap_format"
        ),
    )


def _parse_templates(raw: object) -> tuple[TemplateSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("templates must be a non-empty list")
    templates: list[TemplateSpec] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        data = _expect_mapping(value, f"templates[{index}]")
        _expect_exact_keys(data, {"path", "weight"}, f"templates[{index}]")
        path = _expect_relative_path(data["path"], f"templates[{index}].path")
        weight_value = data["weight"]
        if type(weight_value) not in (int, float):
            raise ValueError(f"templates[{index}].weight must be numeric")
        weight = float(weight_value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"templates[{index}].weight must be finite and positive")
        if path in seen:
            raise ValueError(f"duplicate template path: {path}")
        seen.add(path)
        templates.append(TemplateSpec(path=path, weight=weight))
    return tuple(templates)


def _parse_postprocess(raw: object, family: str) -> PostprocessSpec:
    data = _expect_mapping(raw, "postprocess")
    if family == "v1":
        _expect_exact_keys(data, {"enabled"}, "postprocess")
        if _expect_bool(data["enabled"], "postprocess.enabled"):
            raise ValueError("V1 data recipes must not enable V2 postprocessing")
        return PostprocessSpec(enabled=False)

    _expect_exact_keys(data, _V2_POSTPROCESS_KEYS, "postprocess")
    if not _expect_bool(data["enabled"], "postprocess.enabled"):
        raise ValueError("V2 data recipes must enable texture/env/volume postprocessing")
    strings: dict[str, str] = {}
    for key in ("texture_encoder", "model_id", "revision", "subfolder", "torch_dtype"):
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"postprocess.{key} must be a non-empty string")
        strings[key] = value
    drop_keys = data["drop_keys"]
    if (
        not isinstance(drop_keys, list)
        or not drop_keys
        or any(not isinstance(value, str) or not value for value in drop_keys)
        or len(set(drop_keys)) != len(drop_keys)
    ):
        raise ValueError("postprocess.drop_keys must be a non-empty unique string list")
    process_env_map = _expect_bool(
        data["process_env_map"], "postprocess.process_env_map"
    )
    if not process_env_map:
        raise ValueError(
            "V2 data recipes must set postprocess.process_env_map=true; "
            "the released batch postprocessor always encodes environment maps"
        )
    return PostprocessSpec(
        enabled=True,
        texture_encoder=strings["texture_encoder"],
        model_id=strings["model_id"],
        revision=strings["revision"],
        subfolder=strings["subfolder"],
        torch_dtype=strings["torch_dtype"],
        triangle_batch_size=_expect_positive_int(
            data["triangle_batch_size"], "postprocess.triangle_batch_size"
        ),
        keep_raw=_expect_bool(data["keep_raw"], "postprocess.keep_raw"),
        drop_keys=tuple(drop_keys),
        process_env_map=True,
    )


def _parse_artifacts(raw: object) -> tuple[ArtifactSpec, ...]:
    data = _expect_mapping(raw, "artifacts")
    if not data:
        raise ValueError("artifacts must not be empty")
    artifacts: list[ArtifactSpec] = []
    paths: set[str] = set()
    for name, value in data.items():
        if not isinstance(name, str) or not name or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"invalid artifact key: {name!r}")
        artifact = _expect_mapping(value, f"artifacts.{name}")
        _expect_exact_keys(
            artifact, {"path", "sources", "total_rows"}, f"artifacts.{name}"
        )
        path = _expect_relative_path(artifact["path"], f"artifacts.{name}.path")
        if path in paths:
            raise ValueError(f"duplicate artifact path: {path}")
        paths.add(path)
        raw_sources = artifact["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"artifacts.{name}.sources must be a non-empty list")
        sources: list[ArtifactSource] = []
        for index, source_value in enumerate(raw_sources):
            source = _expect_mapping(
                source_value, f"artifacts.{name}.sources[{index}]"
            )
            _expect_exact_keys(
                source,
                {"recipe", "historical_rows"},
                f"artifacts.{name}.sources[{index}]",
            )
            recipe_id = source["recipe"]
            if not isinstance(recipe_id, str) or not _RECIPE_ID_RE.fullmatch(recipe_id):
                raise ValueError(
                    f"artifacts.{name}.sources[{index}].recipe is invalid"
                )
            historical_rows = source["historical_rows"]
            if historical_rows is not None:
                historical_rows = _expect_positive_int(
                    historical_rows,
                    f"artifacts.{name}.sources[{index}].historical_rows",
                )
            sources.append(ArtifactSource(recipe_id, historical_rows))
        total_rows = artifact["total_rows"]
        if total_rows is not None:
            total_rows = _expect_positive_int(total_rows, f"artifacts.{name}.total_rows")
        known_rows = [source.historical_rows for source in sources]
        if all(rows is not None for rows in known_rows):
            expected_total = sum(int(rows) for rows in known_rows)
            if total_rows != expected_total:
                raise ValueError(
                    f"artifacts.{name}.total_rows={total_rows!r} does not match "
                    f"known source rows {expected_total}"
                )
        artifacts.append(ArtifactSpec(name, path, tuple(sources), total_rows))
    return tuple(artifacts)


def load_data_recipe(
    path: str | Path, *, repository_root: str | Path | None = None
) -> DataRecipe:
    """Load and validate one schema-v1 data recipe without importing Blender."""

    recipe_path = Path(path).expanduser().resolve()
    raw = load_mapping(recipe_path)
    _expect_exact_keys(raw, _RECIPE_KEYS, "data recipe")
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise ValueError("data recipe schema_version must be integer 1")
    recipe_id = raw["recipe_id"]
    if not isinstance(recipe_id, str) or not _RECIPE_ID_RE.fullmatch(recipe_id):
        raise ValueError(f"invalid recipe_id: {recipe_id!r}")
    family = raw["family"]
    if family not in {"v1", "v2"} or not recipe_id.startswith(f"{family}/"):
        raise ValueError("family must be v1/v2 and match recipe_id")
    partition_key = raw["partition_key"]
    if not isinstance(partition_key, str) or not partition_key.strip():
        raise ValueError("partition_key must be a non-empty historical identifier")
    export_profile = _expect_relative_path(raw["export_profile"], "export_profile")

    assets = _expect_mapping(raw["assets"], "assets")
    _expect_exact_keys(assets, {"require_external_manifest"}, "assets")
    require_external_manifest = _expect_bool(
        assets["require_external_manifest"], "assets.require_external_manifest"
    )

    historical = _expect_mapping(raw["historical"], "historical")
    _expect_exact_keys(historical, set(_KNOWN_GAPS), "historical")
    non_null_history = [key for key in _KNOWN_GAPS if historical[key] is not None]
    if non_null_history:
        raise ValueError(
            "unrecoverable historical fields must remain null: "
            + ", ".join(non_null_history)
        )
    known_gaps = raw["known_gaps"]
    if not isinstance(known_gaps, list) or tuple(known_gaps) != _KNOWN_GAPS:
        raise ValueError(
            "known_gaps must explicitly list sample_budget, seed, list_order in order"
        )

    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else _find_repository_root(recipe_path)
    )
    recipe = DataRecipe(
        path=recipe_path,
        repository_root=root,
        schema_version=1,
        recipe_id=recipe_id,
        family=family,
        partition_key=partition_key,
        export_profile=export_profile,
        require_external_manifest=require_external_manifest,
        template_corrections=_parse_template_corrections(
            raw["template_corrections"], family
        ),
        render=_parse_render(raw["render"], family),
        templates=_parse_templates(raw["templates"]),
        postprocess=_parse_postprocess(raw["postprocess"], family),
        artifacts=_parse_artifacts(raw["artifacts"]),
        known_gaps=tuple(known_gaps),
    )
    validate_data_recipe_paths(recipe)
    return recipe


def validate_data_recipe_paths(recipe: DataRecipe) -> None:
    if not recipe.export_profile_path.is_file():
        raise FileNotFoundError(
            f"Export profile does not exist: {recipe.export_profile_path}"
        )
    profile = load_export_profile(recipe.export_profile_path)
    if profile.format.value != recipe.family:
        raise ValueError(
            f"Export profile format {profile.format.value!r} does not match "
            f"recipe family {recipe.family!r}"
        )
    missing = [
        str(recipe.template_path(template))
        for template in recipe.templates
        if not recipe.template_path(template).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing recipe templates: " + ", ".join(missing))


@dataclass(frozen=True)
class PlannedTask:
    index: int
    seed: int
    template: str
    raw_output: str
    final_output: str


def _validate_task_plan_inputs(num_samples: int, seed: int) -> None:
    num_samples = _expect_positive_int(num_samples, "num_samples")
    if num_samples > _MAX_TASKS:
        raise ValueError(
            f"num_samples must be at most {_MAX_TASKS}; the task seed permutation "
            "has a 32-bit domain"
        )
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")


def _task_seed_parameters(seed: int) -> tuple[int, int]:
    if type(seed) is not int or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    # Persisted task-plan protocol tag, not the CLI name. Renaming it would
    # change every deterministic recipe seed produced before this move.
    key = hashlib.sha256(
        b"renderformer-data-task-seed-v1\0" + seed.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(key[:4], "little") | 1, int.from_bytes(
        key[4:8], "little"
    )


def task_seed_for_index(seed: int, index: int) -> int:
    """Map a task index to a collision-free 32-bit seed for one root seed.

    The odd affine multiplier is a permutation of the 32-bit integer domain,
    so indices in ``[0, 2**32)`` cannot collide.  Its key material is derived
    from the root seed without consuming the template-selection RNG stream.
    """

    if type(index) is not int or index < 0 or index >= _MAX_TASKS:
        raise ValueError(f"index must be an integer in [0, {_MAX_TASKS})")
    multiplier, offset = _task_seed_parameters(seed)
    return (multiplier * index + offset) & 0xFFFFFFFF


def iter_task_seeds(seed: int, count: int) -> Iterator[int]:
    """Yield collision-free per-task seeds with constant planner memory."""

    _validate_task_plan_inputs(count, seed)
    multiplier, offset = _task_seed_parameters(seed)
    for index in range(count):
        yield (multiplier * index + offset) & 0xFFFFFFFF


def iter_data_tasks(
    recipe: DataRecipe, *, num_samples: int, seed: int
) -> Iterator[PlannedTask]:
    """Yield a deterministic plan without materializing all tasks in memory."""

    _validate_task_plan_inputs(num_samples, seed)
    template_rng = random.Random(seed)
    cumulative_weights = list(
        accumulate(template.weight for template in recipe.templates)
    )
    total_weight = cumulative_weights[-1]
    highest_index = len(recipe.templates) - 1
    task_seeds = iter_task_seeds(seed, num_samples)
    profile_name = Path(recipe.export_profile).stem + ".h5"
    for index, task_seed in enumerate(task_seeds):
        template_index = bisect(
            cumulative_weights,
            template_rng.random() * total_weight,
            0,
            highest_index,
        )
        template = recipe.templates[template_index]
        sample_dir = f"{index:08d}"
        if recipe.family == "v1":
            raw_output = f"scenes/{sample_dir}/{profile_name}"
            final_output = raw_output
        else:
            raw_output = f"raw/{sample_dir}/{profile_name}"
            final_output = f"scenes/{sample_dir}/rendered_scene.h5"
        yield PlannedTask(
            index=index,
            seed=task_seed,
            template=template.path,
            raw_output=raw_output,
            final_output=final_output,
        )


def plan_data_tasks(
    recipe: DataRecipe, *, num_samples: int, seed: int
) -> Iterator[PlannedTask]:
    """Return a lazy deterministic task plan.

    Callers that intentionally need a small in-memory snapshot can wrap this
    iterator in ``tuple(...)``. Release-scale execution consumes it directly.
    """

    return iter_data_tasks(recipe, num_samples=num_samples, seed=seed)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_asset_placeholders(recipe: DataRecipe) -> set[str]:
    from renderformer.data.blender.templates import (
        load_scene_template,
        template_requires_texture_assets,
    )

    placeholders: set[str] = set()
    for template in recipe.templates:
        scene = load_scene_template(recipe.template_path(template), resolve_assets=False)
        if scene.object_placement.objects.count["max"] > 0:
            placeholders.add(scene.object_placement.objects.list_path)
        if template_requires_texture_assets(scene):
            texture_list = scene.material.texture_map_list_path
            if texture_list is not None:
                placeholders.add(texture_list)
        if scene.env_map is not None:
            placeholders.add(scene.env_map.list_path)
    return placeholders


def _input_fingerprints(
    recipe: DataRecipe, asset_manifest_path: str | Path | None
) -> dict[str, object]:
    """Fingerprint only metadata-declared inputs; never scan asset directories."""

    tracked_inputs = {
        "recipe": {
            "path": str(recipe.path),
            "sha256": _file_sha256(recipe.path),
        },
        "export_profile": {
            "path": str(recipe.export_profile_path.resolve()),
            "sha256": _file_sha256(recipe.export_profile_path),
        },
        "templates": [
            {
                "path": str(recipe.template_path(template).resolve()),
                "sha256": _file_sha256(recipe.template_path(template)),
            }
            for template in recipe.templates
        ],
    }
    if asset_manifest_path is None:
        tracked_inputs["asset_manifest"] = None
        return tracked_inputs

    from renderformer.data.assets import load_asset_manifest

    manifest = load_asset_manifest(asset_manifest_path)
    collections = []
    for placeholder in sorted(_required_asset_placeholders(recipe)):
        location = manifest.collections.get(placeholder)
        if location is None:
            if Path(placeholder).parts[:1] == ("external",):
                raise KeyError(
                    f"Asset manifest {manifest.path} does not map {placeholder!r}"
                )
            continue
        collections.append(
            {
                "placeholder": placeholder,
                "kind": location.kind,
                "list_path": str(location.list_path),
                "list_sha256": _file_sha256(location.list_path),
                "root_path": str(location.root_path),
            }
        )
    tracked_inputs["asset_manifest"] = {
        "path": str(manifest.path),
        "sha256": _file_sha256(manifest.path),
        "collections": collections,
    }
    return tracked_inputs


def _task_json_line(task: PlannedTask) -> bytes:
    return (
        json.dumps(asdict(task), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_task_file(path: Path, tasks: Iterable[PlannedTask]) -> tuple[str, int]:
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("wb") as handle:
        for task in tasks:
            line = _task_json_line(task)
            handle.write(line)
            digest.update(line)
            count += 1
    os.replace(temporary, path)
    return digest.hexdigest(), count


def _task_stream_sha256(tasks: Iterable[PlannedTask]) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for task in tasks:
        digest.update(_task_json_line(task))
        count += 1
    return digest.hexdigest(), count


def write_task_manifest(
    recipe: DataRecipe,
    tasks: Iterable[PlannedTask],
    *,
    output_dir: Path,
    num_samples: int,
    seed: int,
    mode: str,
    input_fingerprints: Mapping[str, object],
) -> Path:
    """Create or strictly validate a resumable task manifest.

    Individual tasks live in a JSONL sidecar so million-task plans can be
    written and checked without holding a giant JSON list in memory.
    """

    manifest_path = output_dir / "task_manifest.json"
    task_file = output_dir / "tasks.jsonl"
    manifest_exists = manifest_path.exists()
    if manifest_exists and not task_file.is_file():
        raise FileNotFoundError(
            f"Task manifest exists but its task file is missing: {task_file}"
        )
    if task_file.exists():
        expected_task_sha256, task_count = _task_stream_sha256(tasks)
        actual_task_sha256 = _file_sha256(task_file)
        if actual_task_sha256 != expected_task_sha256:
            raise ValueError(
                f"Existing task plan does not match recipe/seed/count: {task_file}"
            )
    else:
        expected_task_sha256, task_count = _write_task_file(task_file, tasks)
    if task_count != num_samples:
        raise RuntimeError(
            f"Task planner produced {task_count} tasks, expected {num_samples}"
        )
    payload = {
        "schema_version": 2,
        "recipe_id": recipe.recipe_id,
        "recipe_sha256": _file_sha256(recipe.path),
        "num_samples": num_samples,
        "seed": seed,
        "mode": mode,
        "planner": _TASK_PLANNER_VERSION,
        "inputs": dict(input_fingerprints),
        "input_fingerprint_sha256": _canonical_sha256(input_fingerprints),
        "tasks": {
            "path": task_file.name,
            "sha256": expected_task_sha256,
            "count": task_count,
            "format": "jsonl",
        },
        "historical_equivalence": False,
        "known_gaps": list(recipe.known_gaps),
    }
    if manifest_exists:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "Existing task_manifest.json does not match the requested "
                "recipe hash, seed, count, mode, or input fingerprints"
            )
    else:
        _atomic_write_text(
            manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return manifest_path


def _task_asset_overrides(template_path: Path, asset_manifest: object) -> dict[str, str]:
    from renderformer.data.assets import AssetManifest
    from renderformer.data.blender.templates import (
        load_scene_template,
        template_requires_texture_assets,
    )

    if not isinstance(asset_manifest, AssetManifest):
        raise TypeError("asset_manifest must be an AssetManifest")
    template = load_scene_template(template_path, resolve_assets=False)
    overrides: dict[str, str] = {}

    def apply(
        placeholder: str | None,
        expected_kind: str,
        list_key: str,
        root_key: str,
    ) -> None:
        if placeholder is None:
            return
        location = asset_manifest.collections.get(placeholder)
        if location is None:
            if Path(placeholder).parts[:1] == ("external",):
                raise KeyError(
                    f"Asset manifest {asset_manifest.path} does not map {placeholder!r}"
                )
            return
        if location.kind != expected_kind:
            raise ValueError(
                f"Asset manifest maps {placeholder!r} as {location.kind!r}, "
                f"expected {expected_kind!r}"
            )
        overrides[list_key] = str(location.list_path)
        overrides[root_key] = str(location.root_path)

    objects = template.object_placement.objects
    if objects.count["max"] > 0:
        apply(objects.list_path, "object", "object_list_path", "object_root")
    if template_requires_texture_assets(template):
        apply(
            template.material.texture_map_list_path,
            "texture",
            "texture_list_path",
            "texture_root",
        )
    if template.env_map is not None:
        apply(
            template.env_map.list_path,
            "env-map",
            "env_map_list_path",
            "env_map_root",
        )
    return overrides


_WORKER_RECIPE: DataRecipe | None = None
_WORKER_OUTPUT_DIR: Path | None = None
_WORKER_ASSET_MANIFEST: object | None = None
_WORKER_RECIPE_SHA256: str | None = None
_WORKER_INPUT_FINGERPRINT: str | None = None


def _initialize_generation_worker(
    recipe_path: str,
    output_dir: str,
    asset_manifest_path: str | None,
    recipe_sha256: str,
    input_fingerprint: str,
) -> None:
    global _WORKER_RECIPE, _WORKER_OUTPUT_DIR, _WORKER_ASSET_MANIFEST
    global _WORKER_RECIPE_SHA256, _WORKER_INPUT_FINGERPRINT
    _WORKER_RECIPE = load_data_recipe(recipe_path)
    _WORKER_OUTPUT_DIR = Path(output_dir)
    _WORKER_RECIPE_SHA256 = recipe_sha256
    _WORKER_INPUT_FINGERPRINT = input_fingerprint
    if asset_manifest_path is not None:
        from renderformer.data.assets import load_asset_manifest

        _WORKER_ASSET_MANIFEST = load_asset_manifest(asset_manifest_path)
    else:
        _WORKER_ASSET_MANIFEST = None


def _task_output_identity(
    recipe: DataRecipe,
    task: PlannedTask,
    *,
    recipe_sha256: str,
    input_fingerprint: str,
    stage: str,
) -> str:
    return _canonical_sha256(
        {
            "recipe_id": recipe.recipe_id,
            "recipe_sha256": recipe_sha256,
            "input_fingerprint_sha256": input_fingerprint,
            "task": asdict(task),
            "stage": stage,
        }
    )


def _mark_task_output(
    path: Path,
    recipe: DataRecipe,
    task: PlannedTask,
    *,
    recipe_sha256: str,
    input_fingerprint: str,
    stage: str,
) -> None:
    import h5py

    from renderformer.data.h5.validate import validate_h5_file

    validate_h5_file(path, recipe.family)
    identity = _task_output_identity(
        recipe,
        task,
        recipe_sha256=recipe_sha256,
        input_fingerprint=input_fingerprint,
        stage=stage,
    )
    with h5py.File(path, "r+") as handle:
        handle.attrs["renderformer_task_identity"] = identity
        handle.attrs["renderformer_recipe_id"] = recipe.recipe_id
        handle.attrs["renderformer_recipe_sha256"] = recipe_sha256
        handle.attrs["renderformer_input_fingerprint_sha256"] = input_fingerprint
        handle.attrs["renderformer_task_index"] = task.index
        handle.attrs["renderformer_task_seed"] = task.seed
        handle.attrs["renderformer_task_template"] = task.template
        handle.attrs["renderformer_task_stage"] = stage


def _validate_task_output(
    path: Path,
    recipe: DataRecipe,
    task: PlannedTask,
    *,
    recipe_sha256: str,
    input_fingerprint: str,
    stage: str,
) -> None:
    import h5py

    from renderformer.data.h5.validate import validate_h5_file

    validate_h5_file(path, recipe.family)
    expected = _task_output_identity(
        recipe,
        task,
        recipe_sha256=recipe_sha256,
        input_fingerprint=input_fingerprint,
        stage=stage,
    )
    with h5py.File(path, "r") as handle:
        actual = handle.attrs.get("renderformer_task_identity")
    if actual != expected:
        raise ValueError(
            f"Existing output does not match task {task.index} and cannot be "
            f"resumed safely: {path}"
        )


def _worker_label() -> str:
    match = re.search(r"(\d+)$", current_process().name)
    return f"W{match.group(1) if match else '0'}"


def _run_generation_task(task: PlannedTask) -> tuple[PlannedTask, str]:
    if (
        _WORKER_RECIPE is None
        or _WORKER_OUTPUT_DIR is None
        or _WORKER_RECIPE_SHA256 is None
        or _WORKER_INPUT_FINGERPRINT is None
    ):
        raise RuntimeError("generation worker was not initialized")
    recipe = _WORKER_RECIPE
    raw_path = _WORKER_OUTPUT_DIR / task.raw_output
    if raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite generated sample: {raw_path}")
    if _WORKER_ASSET_MANIFEST is None:
        raise ValueError("generation requires an external asset manifest")

    random.seed(task.seed)
    np.random.seed(task.seed)
    template_path = recipe.repository_root / task.template
    overrides = _task_asset_overrides(template_path, _WORKER_ASSET_MANIFEST)
    from renderformer.data.pipelines.generate import run_blender_generate

    generate_kwargs: dict[str, Any] = {}
    if recipe.render.texture_crop_resolution is not None:
        generate_kwargs["texture_crop_res"] = recipe.render.texture_crop_resolution
    if recipe.render.texture_target_resolution is not None:
        generate_kwargs["texture_target_res"] = recipe.render.texture_target_resolution
    results = run_blender_generate(
        scene_config_path=None,
        template_path=template_path,
        template_root=None,
        output_dir=raw_path.parent,
        profile_paths=[recipe.export_profile_path],
        num_views=recipe.render.num_views,
        resolution=recipe.render.resolution,
        spp=recipe.render.spp,
        texture_size=recipe.render.texture_size,
        min_views=recipe.render.min_views,
        use_heightmap_format=recipe.render.use_heightmap_format,
        include_background=recipe.render.include_background,
        background_topology_debias=(
            recipe.template_corrections.background_topology_debias
        ),
        **overrides,
        **generate_kwargs,
    )
    if (
        len(results) != 1
        or Path(results[0].output_path).resolve() != raw_path.resolve()
    ):
        raise RuntimeError(
            f"Generation output mismatch for task {task.index}: expected {raw_path}, "
            f"got {[str(result.output_path) for result in results]}"
        )
    _mark_task_output(
        raw_path,
        recipe,
        task,
        recipe_sha256=_WORKER_RECIPE_SHA256,
        input_fingerprint=_WORKER_INPUT_FINGERPRINT,
        stage="final" if recipe.family == "v1" else "raw",
    )
    print(f"[{_worker_label()}] generated task {task.index}", flush=True)
    return task, str(raw_path)


def _validate_runtime_assets(
    recipe: DataRecipe, asset_manifest_path: str | Path, *, check_entries: bool
) -> None:
    from renderformer.data.assets import validate_template_assets

    report = validate_template_assets(
        [recipe.template_path(template) for template in recipe.templates],
        asset_manifest_path=asset_manifest_path,
        check_entries=check_entries,
    )
    if not report.ok:
        raise ValueError(
            "Data-recipe asset validation failed:\n- " + "\n- ".join(report.issues)
        )


def _task_action(
    recipe: DataRecipe,
    task: PlannedTask,
    output_dir: Path,
    *,
    recipe_sha256: str,
    input_fingerprint: str,
) -> str:
    raw_path = output_dir / task.raw_output
    final_path = output_dir / task.final_output
    if final_path.exists():
        _validate_task_output(
            final_path,
            recipe,
            task,
            recipe_sha256=recipe_sha256,
            input_fingerprint=input_fingerprint,
            stage="final",
        )
        if recipe.family == "v2" and raw_path.exists():
            _validate_task_output(
                raw_path,
                recipe,
                task,
                recipe_sha256=recipe_sha256,
                input_fingerprint=input_fingerprint,
                stage="raw",
            )
        return "complete"
    if recipe.family == "v2" and raw_path.exists():
        _validate_task_output(
            raw_path,
            recipe,
            task,
            recipe_sha256=recipe_sha256,
            input_fingerprint=input_fingerprint,
            stage="raw",
        )
        return "postprocess"
    return "generate"


def _write_run_state(
    path: Path,
    *,
    recipe: DataRecipe,
    task_manifest_sha256: str,
    input_fingerprint: str,
    mode: str,
    num_samples: int,
    completed: int,
    status: str,
) -> None:
    identity = {
        "schema_version": 1,
        "recipe_id": recipe.recipe_id,
        "task_manifest_sha256": task_manifest_sha256,
        "input_fingerprint_sha256": input_fingerprint,
        "mode": mode,
        "num_samples": num_samples,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_identity = {key: existing.get(key) for key in identity}
        if existing_identity != identity:
            raise ValueError(f"Existing run state does not match this task plan: {path}")
    payload = {**identity, "completed": completed, "status": status}
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_or_validate_paths_file(
    path: Path, tasks: Iterable[PlannedTask], output_dir: Path
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(f"{(output_dir / task.final_output).resolve()}\n")
    if path.exists():
        if _file_sha256(path) != _file_sha256(temporary):
            temporary.unlink()
            raise ValueError(f"Existing paths file does not match the task plan: {path}")
        temporary.unlink()
    else:
        os.replace(temporary, path)


def run_data_recipe_batch(
    recipe: DataRecipe,
    *,
    output_dir: str | Path,
    num_samples: int,
    seed: int,
    num_workers: int = 1,
    asset_manifest_path: str | Path | None = None,
    dry_run: bool = False,
    check_asset_entries: bool = True,
) -> Path:
    """Plan and run one recipe without discovering inputs from the filesystem.

    Scene generation is process-parallel and streamed through the parent.
    V2 learned postprocessing reuses one encoder and starts as raw scenes
    arrive. Re-running the same invocation resumes outputs whose embedded task
    identity is valid; mismatched or corrupt outputs fail before new work.
    """

    num_workers = _expect_positive_int(num_workers, "num_workers")
    _validate_task_plan_inputs(num_samples, seed)
    output_path = Path(output_dir).expanduser().resolve()
    if recipe.require_external_manifest and not dry_run:
        if asset_manifest_path is None:
            raise ValueError("real generation requires --asset-manifest")
    if asset_manifest_path is not None:
        _validate_runtime_assets(
            recipe, asset_manifest_path, check_entries=check_asset_entries
        )
    output_path.mkdir(parents=True, exist_ok=True)
    mode = "dry-run" if dry_run else "generate"
    inputs = _input_fingerprints(recipe, asset_manifest_path)
    recipe_sha256 = _file_sha256(recipe.path)
    input_fingerprint = _canonical_sha256(inputs)
    manifest_path = write_task_manifest(
        recipe,
        iter_data_tasks(recipe, num_samples=num_samples, seed=seed),
        output_dir=output_path,
        num_samples=num_samples,
        seed=seed,
        mode=mode,
        input_fingerprints=inputs,
    )
    if dry_run:
        return manifest_path

    action_counts = {"complete": 0, "postprocess": 0, "generate": 0}
    for task in iter_data_tasks(recipe, num_samples=num_samples, seed=seed):
        action = _task_action(
            recipe,
            task,
            output_path,
            recipe_sha256=recipe_sha256,
            input_fingerprint=input_fingerprint,
        )
        action_counts[action] += 1
    print(
        "[resume] "
        f"complete={action_counts['complete']} "
        f"postprocess={action_counts['postprocess']} "
        f"generate={action_counts['generate']}",
        flush=True,
    )

    task_manifest_sha256 = _file_sha256(manifest_path)
    state_path = output_path / "run_state.json"
    completed = action_counts["complete"]
    _write_run_state(
        state_path,
        recipe=recipe,
        task_manifest_sha256=task_manifest_sha256,
        input_fingerprint=input_fingerprint,
        mode=mode,
        num_samples=num_samples,
        completed=completed,
        status="running",
    )
    progress_interval = max(1, num_samples // 100)

    def record_completion(task: PlannedTask) -> None:
        nonlocal completed
        completed += 1
        if completed % progress_interval == 0 or completed == num_samples:
            print(
                f"[progress] completed {completed}/{num_samples} "
                f"(last task {task.index})",
                flush=True,
            )
            _write_run_state(
                state_path,
                recipe=recipe,
                task_manifest_sha256=task_manifest_sha256,
                input_fingerprint=input_fingerprint,
                mode=mode,
                num_samples=num_samples,
                completed=completed,
                status="running",
            )

    initializer_args = (
        str(recipe.path),
        str(output_path),
        str(Path(asset_manifest_path).expanduser().resolve())
        if asset_manifest_path is not None
        else None,
        recipe_sha256,
        input_fingerprint,
    )
    encoder_config: dict[str, Any] | None = None
    texture_encoder: object | None = None
    if recipe.family == "v2":
        encoder_config = recipe.postprocess.encoder_config()

    def finish_generated_task(task: PlannedTask, raw_path_value: str) -> None:
        nonlocal texture_encoder
        raw_path = Path(raw_path_value)
        if recipe.family == "v2":
            from renderformer.data.textures.encoders import build_texture_encoder
            from renderformer.data.textures.postprocess import postprocess_v2_h5

            assert encoder_config is not None
            if texture_encoder is None:
                texture_encoder = build_texture_encoder(encoder_config)
            final_path = output_path / task.final_output
            if final_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite postprocessed sample: {final_path}"
                )
            postprocess_v2_h5(
                raw_path,
                final_path,
                encoder_config,
                recipe.postprocess.keep_raw,
                texture_encoder=texture_encoder,
                drop_keys=recipe.postprocess.drop_keys,
            )
            _mark_task_output(
                final_path,
                recipe,
                task,
                recipe_sha256=recipe_sha256,
                input_fingerprint=input_fingerprint,
                stage="final",
            )
            print(f"[postprocess] completed task {task.index}", flush=True)
        record_completion(task)

    if recipe.family == "v2" and action_counts["postprocess"]:
        for task in iter_data_tasks(recipe, num_samples=num_samples, seed=seed):
            if (
                _task_action(
                    recipe,
                    task,
                    output_path,
                    recipe_sha256=recipe_sha256,
                    input_fingerprint=input_fingerprint,
                )
                == "postprocess"
            ):
                finish_generated_task(task, str(output_path / task.raw_output))

    generate_count = action_counts["generate"]
    if generate_count:
        generate_tasks = (
            task
            for task in iter_data_tasks(recipe, num_samples=num_samples, seed=seed)
            if _task_action(
                recipe,
                task,
                output_path,
                recipe_sha256=recipe_sha256,
                input_fingerprint=input_fingerprint,
            )
            == "generate"
        )
        worker_count = min(num_workers, generate_count)
        if worker_count == 1:
            _initialize_generation_worker(*initializer_args)
            for task in generate_tasks:
                generated_task, raw_path = _run_generation_task(task)
                finish_generated_task(generated_task, raw_path)
        else:
            context = get_context("spawn")
            with context.Pool(
                processes=worker_count,
                initializer=_initialize_generation_worker,
                initargs=initializer_args,
            ) as pool:
                for generated_task, raw_path in pool.imap_unordered(
                    _run_generation_task, generate_tasks, chunksize=1
                ):
                    finish_generated_task(generated_task, raw_path)

    for task in iter_data_tasks(recipe, num_samples=num_samples, seed=seed):
        final_path = output_path / task.final_output
        if not final_path.is_file():
            raise RuntimeError(
                f"Batch completed with missing output for task {task.index}: {final_path}"
            )
        _validate_task_output(
            final_path,
            recipe,
            task,
            recipe_sha256=recipe_sha256,
            input_fingerprint=input_fingerprint,
            stage="final",
        )
    paths_file = output_path / "paths.txt"
    _write_or_validate_paths_file(
        paths_file,
        iter_data_tasks(recipe, num_samples=num_samples, seed=seed),
        output_path,
    )
    run_manifest = {
        "schema_version": 3,
        "recipe_id": recipe.recipe_id,
        "output_kind": "new_seeded_generation",
        "task_manifest": manifest_path.name,
        "task_manifest_sha256": task_manifest_sha256,
        "input_fingerprint_sha256": input_fingerprint,
        "paths_file": paths_file.name,
        "num_samples": num_samples,
        "historical_equivalence": False,
    }
    run_manifest_path = output_path / "run_manifest.json"
    if run_manifest_path.exists():
        existing_run_manifest = json.loads(
            run_manifest_path.read_text(encoding="utf-8")
        )
        if existing_run_manifest != run_manifest:
            raise ValueError(
                f"Existing run manifest does not match this task plan: {run_manifest_path}"
            )
    else:
        _atomic_write_text(
            run_manifest_path,
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        )
    _write_run_state(
        state_path,
        recipe=recipe,
        task_manifest_sha256=task_manifest_sha256,
        input_fingerprint=input_fingerprint,
        mode=mode,
        num_samples=num_samples,
        completed=num_samples,
        status="complete",
    )
    return paths_file


def describe_data_recipe(recipe: DataRecipe) -> str:
    lines = []
    for key, value in recipe.describe().items():
        if isinstance(value, dict):
            value = ",".join(f"{name}={path}" for name, path in value.items())
        elif isinstance(value, list):
            value = ",".join(str(item) for item in value)
        lines.append(f"{key}\t{value}")
    return "\n".join(lines)


__all__ = [
    "ArtifactSource",
    "ArtifactSpec",
    "DataRecipe",
    "PlannedTask",
    "PostprocessSpec",
    "RenderSpec",
    "TemplateCorrectionsSpec",
    "TemplateSpec",
    "describe_data_recipe",
    "iter_data_tasks",
    "iter_task_seeds",
    "load_data_recipe",
    "plan_data_tasks",
    "run_data_recipe_batch",
    "task_seed_for_index",
    "validate_data_recipe_paths",
    "write_task_manifest",
]
