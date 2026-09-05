"""Deterministic, sharded material-sphere EXR generation with PyPI Blender."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from heapq import nsmallest
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
from typing import Any
import uuid

import yaml


PIPELINE_ID = "renderformer-material-spheres-v1"
PLAN_FORMAT = "renderformer-material-sphere-plan-v1"
RECORD_FORMAT = "renderformer-material-sphere-record-v1"
FAMILY_NAMES = (
    "principled_metallic",
    "diffuse_specular",
    "principled_transmission",
)
REQUIRED_RUNTIME = {
    "bpy": "4.5.10",
    "bpy-helper": "0.0.13",
    "simple-exr": "0.0.0",
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "recipe_id",
    "pipeline_id",
    "preprocessing",
    "render",
    "principled_fixed",
    "families",
}
_PREPROCESSING_KEYS = {"mode", "alpha_policy", "output_channels"}
_RENDER_KEYS = {
    "resolution",
    "samples_per_pixel",
    "sphere_segments",
    "sphere_rings",
    "camera_position",
    "camera_target",
    "camera_up",
    "fov_degrees",
    "environment_strength",
    "environment_map_sha256",
    "cycles_seed",
    "threads",
}
_PRINCIPLED_FIXED_KEYS = {
    "subsurface_weight",
    "specular_ior_level",
    "specular_tint_rgb",
    "anisotropic",
    "anisotropic_rotation",
    "sheen_weight",
    "sheen_tint_rgb",
    "coat_weight",
    "coat_roughness",
    "emission_rgb",
    "alpha",
}
_FAMILY_KEYS = {
    "principled_metallic": {
        "count",
        "validation_count",
        "base_color_range",
        "metallic_range",
        "roughness_range",
        "transmission_weight",
        "ior",
    },
    "diffuse_specular": {
        "count",
        "validation_count",
        "diffuse_color_range",
        "specular_color_range",
        "roughness_range",
    },
    "principled_transmission": {
        "count",
        "validation_count",
        "base_color_range",
        "metallic_range",
        "roughness_range",
        "transmission_power",
        "ior_range",
    },
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_or_match(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f"existing file does not match this material plan: {path}")
        return
    _atomic_write(path, content)


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch; missing={missing}, unknown={unknown}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return [_finite_float(component, f"{name}[{index}]") for index, component in enumerate(value)]


def _range(value: Any, name: str, *, positive: bool = False) -> list[float]:
    result = _vector(value, 2, name)
    if result[0] > result[1]:
        raise ValueError(f"{name} minimum exceeds maximum")
    if positive and result[0] <= 0:
        raise ValueError(f"{name} minimum must be positive")
    return result


def load_material_sphere_recipe(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    """Load and strictly validate the public material-sphere YAML receipt."""

    recipe_path = Path(path).expanduser().resolve()
    if not recipe_path.is_file():
        raise FileNotFoundError(recipe_path)
    with recipe_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    recipe = _mapping(raw, "material-sphere recipe")
    _expect_exact_keys(recipe, _TOP_LEVEL_KEYS, "material-sphere recipe")
    if recipe["schema_version"] != 1:
        raise ValueError("material-sphere schema_version must be 1")
    if not isinstance(recipe["recipe_id"], str) or not recipe["recipe_id"]:
        raise ValueError("recipe_id must be a non-empty string")
    if recipe["pipeline_id"] != PIPELINE_ID:
        raise ValueError(f"pipeline_id must be {PIPELINE_ID!r}")

    preprocessing = _mapping(recipe["preprocessing"], "preprocessing")
    _expect_exact_keys(preprocessing, _PREPROCESSING_KEYS, "preprocessing")
    if preprocessing != {
        "mode": "log10_1p",
        "alpha_policy": "premultiply_then_drop",
        "output_channels": 3,
    }:
        raise ValueError(
            "preprocessing must lock mode=log10_1p, "
            "alpha_policy=premultiply_then_drop, and output_channels=3"
        )

    render = _mapping(recipe["render"], "render")
    _expect_exact_keys(render, _RENDER_KEYS, "render")
    if _positive_int(render["resolution"], "render.resolution") != 256:
        raise ValueError("the material-autoencoder contract requires resolution=256")
    for name in ("samples_per_pixel", "sphere_segments", "sphere_rings", "threads"):
        _positive_int(render[name], f"render.{name}")
    for name in ("camera_position", "camera_target", "camera_up"):
        _vector(render[name], 3, f"render.{name}")
    if _finite_float(render["fov_degrees"], "render.fov_degrees") <= 0:
        raise ValueError("render.fov_degrees must be positive")
    if _finite_float(render["environment_strength"], "render.environment_strength") < 0:
        raise ValueError("render.environment_strength must be non-negative")
    environment_sha256 = render["environment_map_sha256"]
    if (
        not isinstance(environment_sha256, str)
        or len(environment_sha256) != 64
        or any(character not in "0123456789abcdef" for character in environment_sha256)
    ):
        raise ValueError("render.environment_map_sha256 must be a lowercase SHA-256")
    _nonnegative_int(render["cycles_seed"], "render.cycles_seed")

    fixed = _mapping(recipe["principled_fixed"], "principled_fixed")
    _expect_exact_keys(fixed, _PRINCIPLED_FIXED_KEYS, "principled_fixed")
    for name, value in fixed.items():
        if name.endswith("_rgb"):
            _vector(value, 3, f"principled_fixed.{name}")
        else:
            _finite_float(value, f"principled_fixed.{name}")

    families = _mapping(recipe["families"], "families")
    _expect_exact_keys(families, set(FAMILY_NAMES), "families")
    for family_name in FAMILY_NAMES:
        family = _mapping(families[family_name], f"families.{family_name}")
        _expect_exact_keys(family, _FAMILY_KEYS[family_name], f"families.{family_name}")
        count = _positive_int(family["count"], f"families.{family_name}.count")
        validation_count = _nonnegative_int(
            family["validation_count"],
            f"families.{family_name}.validation_count",
        )
        if validation_count >= count:
            raise ValueError(f"families.{family_name} must retain at least one train sample")
        for name, value in family.items():
            if name.endswith("_range"):
                _range(
                    value,
                    f"families.{family_name}.{name}",
                    positive=name == "roughness_range",
                )
            elif name not in {"count", "validation_count"}:
                _finite_float(value, f"families.{family_name}.{name}")
        if family_name == "principled_transmission" and family["transmission_power"] <= 0:
            raise ValueError("transmission_power must be positive")

    return recipe_path, recipe, _sha256_file(recipe_path)


def describe_material_sphere_recipe(path: str | Path) -> dict[str, Any]:
    _, recipe, _ = load_material_sphere_recipe(path)
    return {
        "recipe_id": recipe["recipe_id"],
        "pipeline_id": PIPELINE_ID,
        "resolution": recipe["render"]["resolution"],
        "samples_per_pixel": recipe["render"]["samples_per_pixel"],
        "preprocessing": recipe["preprocessing"],
        "families": {
            name: {
                "count": recipe["families"][name]["count"],
                "validation_count": recipe["families"][name]["validation_count"],
            }
            for name in FAMILY_NAMES
        },
    }


def _derived_seed(root_seed: int, *parts: object) -> int:
    payload = _canonical_json([root_seed, *parts])
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _sample_range(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def _sample_rgb(rng: random.Random, bounds: list[float]) -> list[float]:
    return [_sample_range(rng, bounds) for _ in range(3)]


def _sample_roughness(rng: random.Random, bounds: list[float]) -> float:
    minimum, maximum = (float(bounds[0]), float(bounds[1]))
    # Preserve the historical log10-uniform implementation, including epsilon.
    value = 10 ** rng.uniform(math.log10(minimum + 1e-6), math.log10(maximum + 1e-6))
    return min(max(value, minimum), maximum)


def _sample_material(family_name: str, family: Mapping[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    if family_name == "principled_metallic":
        return {
            "base_color_rgb": _sample_rgb(rng, family["base_color_range"]),
            "metallic": _sample_range(rng, family["metallic_range"]),
            "roughness": _sample_roughness(rng, family["roughness_range"]),
            "transmission_weight": float(family["transmission_weight"]),
            "ior": float(family["ior"]),
        }
    if family_name == "diffuse_specular":
        return {
            "diffuse_color_rgb": _sample_rgb(rng, family["diffuse_color_range"]),
            "specular_color_rgb": _sample_rgb(rng, family["specular_color_range"]),
            "roughness": _sample_roughness(rng, family["roughness_range"]),
        }
    if family_name == "principled_transmission":
        return {
            "base_color_rgb": _sample_rgb(rng, family["base_color_range"]),
            "metallic": _sample_range(rng, family["metallic_range"]),
            "roughness": _sample_roughness(rng, family["roughness_range"]),
            "transmission_weight": rng.random() ** float(family["transmission_power"]),
            "ior": _sample_range(rng, family["ior_range"]),
        }
    raise ValueError(f"unsupported material family: {family_name}")


def _build_tasks(recipe: Mapping[str, Any], seed: int, num_shards: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    global_index = 0
    for family_name in FAMILY_NAMES:
        family = recipe["families"][family_name]
        count = int(family["count"])
        validation_count = int(family["validation_count"])
        validation_indices = set(
            nsmallest(
                validation_count,
                range(count),
                key=lambda index: (
                    _derived_seed(seed, family_name, index, "validation-split"),
                    index,
                ),
            )
        )
        for family_index in range(count):
            task_seed = _derived_seed(seed, family_name, family_index, "material")
            sample_id = f"{family_name}-{family_index:06d}"
            payload = {
                "id": sample_id,
                "global_index": global_index,
                "family": family_name,
                "family_index": family_index,
                "seed": task_seed,
                "shard": global_index % num_shards,
                "split": "validation" if family_index in validation_indices else "train",
                "relative_exr_path": f"samples/{family_name}/{sample_id}/render.exr",
                "material": _sample_material(family_name, family, task_seed),
            }
            payload["task_sha256"] = _sha256_bytes(_canonical_json(payload))
            tasks.append(payload)
            global_index += 1
    return tasks


def _runtime_plan(
    recipe: Mapping[str, Any],
    recipe_sha256: str,
    environment_map: Path,
    seed: int,
    num_shards: int,
) -> tuple[dict[str, Any], bytes]:
    tasks = _build_tasks(recipe, seed, num_shards)
    if num_shards > len(tasks):
        raise ValueError(f"num_shards={num_shards} exceeds task count {len(tasks)}")
    task_bytes = b"".join(_canonical_json(task) + b"\n" for task in tasks)
    plan = {
        "format": PLAN_FORMAT,
        "pipeline_id": PIPELINE_ID,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha256,
        "recipe": dict(recipe),
        "environment_map": {
            "sha256": _sha256_file(environment_map),
        },
        "seed": seed,
        "num_shards": num_shards,
        "task_count": len(tasks),
        "train_count": sum(task["split"] == "train" for task in tasks),
        "validation_count": sum(task["split"] == "validation" for task in tasks),
        "tasks_file": "tasks.jsonl",
        "tasks_sha256": _sha256_bytes(task_bytes),
        "required_runtime": REQUIRED_RUNTIME,
    }
    return plan, task_bytes


def _resolve_runtime_inputs(
    recipe_path: str | Path,
    environment_map: str | Path,
    seed: int,
    num_shards: int,
) -> tuple[dict[str, Any], bytes, str]:
    seed = _nonnegative_int(seed, "seed")
    num_shards = _positive_int(num_shards, "num_shards")
    if seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    _, recipe, recipe_sha256 = load_material_sphere_recipe(recipe_path)
    environment_path = Path(environment_map).expanduser().resolve()
    if environment_path.suffix.lower() != ".exr" or not environment_path.is_file():
        raise FileNotFoundError(f"environment map must be an existing .exr file: {environment_path}")
    environment_sha256 = _sha256_file(environment_path)
    if environment_sha256 != recipe["render"]["environment_map_sha256"]:
        raise ValueError(
            "environment map does not match render.environment_map_sha256: "
            f"{environment_path} has {environment_sha256}"
        )
    plan, task_bytes = _runtime_plan(
        recipe,
        recipe_sha256,
        environment_path,
        seed,
        num_shards,
    )
    return plan, task_bytes, _sha256_bytes(_canonical_json(plan))


def plan_material_spheres(
    *,
    recipe_path: str | Path,
    environment_map: str | Path,
    output_dir: str | Path,
    seed: int,
    num_shards: int,
) -> Path:
    """Write or verify an immutable plan and deterministic task JSONL."""

    plan, task_bytes, _ = _resolve_runtime_inputs(
        recipe_path, environment_map, seed, num_shards
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_or_match(root / "tasks.jsonl", task_bytes)
    _write_or_match(root / "material-sphere-plan.json", _canonical_json(plan) + b"\n")
    return root / "material-sphere-plan.json"


def _load_planned_tasks(
    *,
    recipe_path: str | Path,
    environment_map: str | Path,
    output_dir: str | Path,
    seed: int,
    num_shards: int,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], str]:
    root = Path(output_dir).expanduser().resolve()
    plan_path = root / "material-sphere-plan.json"
    task_path = root / "tasks.jsonl"
    if not plan_path.is_file() or not task_path.is_file():
        raise FileNotFoundError(
            f"material plan is incomplete under {root}; run the plan action first"
        )
    expected_plan, expected_tasks, plan_sha256 = _resolve_runtime_inputs(
        recipe_path, environment_map, seed, num_shards
    )
    if plan_path.read_bytes() != _canonical_json(expected_plan) + b"\n":
        raise ValueError(f"material plan does not match this invocation: {plan_path}")
    if task_path.read_bytes() != expected_tasks:
        raise ValueError(f"material tasks do not match this invocation: {task_path}")
    tasks = [json.loads(line) for line in expected_tasks.splitlines()]
    return root, expected_plan, tasks, plan_sha256


def _require_render_runtime() -> tuple[Any, Any, dict[str, Any]]:
    mismatches = []
    roots: dict[str, Path] = {}
    for distribution, expected in REQUIRED_RUNTIME.items():
        try:
            installed = metadata.distribution(distribution)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{distribution} is not installed (expected {expected})")
            continue
        actual = installed.version
        roots[distribution] = Path(installed.locate_file("")).resolve()
        if actual != expected:
            mismatches.append(f"{distribution}=={actual} (expected {expected})")
    if mismatches:
        raise RuntimeError("material rendering requires: " + "; ".join(mismatches))

    # Import the environment helper before bpy so bpy cannot inject a user
    # extension directory that shadows the pinned PyPI package.
    from bpy_helper.camera import create_camera, look_at_to_c2w
    from bpy_helper.light import set_env_light
    from bpy_helper.material import create_specular_roughness_material
    from bpy_helper.utils import stdout_redirected
    import bpy
    import bpy_helper
    import simple_exr

    if tuple(bpy.app.version[:3]) != (4, 5, 10):
        raise RuntimeError(f"material rendering requires bpy runtime 4.5.10, got {bpy.app.version}")
    for distribution, module in (
        ("bpy", bpy),
        ("bpy-helper", bpy_helper),
        ("simple-exr", simple_exr),
    ):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(roots[distribution]):
            raise RuntimeError(
                f"{distribution} resolved outside its PyPI distribution: {module_path}"
            )
    helpers = {
        "create_camera": create_camera,
        "look_at_to_c2w": look_at_to_c2w,
        "set_env_light": set_env_light,
        "create_specular_roughness_material": create_specular_roughness_material,
        "stdout_redirected": stdout_redirected,
    }
    return bpy, simple_exr, helpers


def _configure_cycles_device(bpy: Any, device: str, device_index: int | None) -> str:
    scene = bpy.context.scene
    if device == "cpu":
        if device_index is not None:
            raise ValueError("device_index is valid only for a GPU backend")
        scene.cycles.device = "CPU"
        return "CPU"
    if device_index is None or device_index < 0:
        raise ValueError("a non-negative device_index is required for GPU rendering")
    backend = device.upper()
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = backend
    preferences.get_devices()
    candidates = [item for item in preferences.devices if item.type == backend]
    if device_index >= len(candidates):
        available = [f"{item.type}:{item.name}" for item in preferences.devices]
        raise RuntimeError(
            f"requested {backend} device index {device_index}; available={available}"
        )
    selected = candidates[device_index]
    for item in preferences.devices:
        item.use = item == selected
    scene.cycles.device = "GPU"
    return f"{selected.type}:{selected.name}"


def _setup_scene(
    plan: Mapping[str, Any],
    environment_map: str | Path,
    device: str,
    device_index: int | None,
):
    bpy, simple_exr, helpers = _require_render_runtime()
    render = plan["recipe"]["render"]
    # Factory-empty state avoids inheriting user startup files while keeping
    # the historical scene geometry and every output-affecting setting explicit.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(render["resolution"])
    scene.render.resolution_y = int(render["resolution"])
    scene.render.resolution_percentage = 100
    scene.cycles.samples = int(render["samples_per_pixel"])
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.threads_mode = "FIXED"
    scene.render.threads = int(render["threads"])
    selected_device = _configure_cycles_device(bpy, device, device_index)

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=1,
        segments=int(render["sphere_segments"]),
        ring_count=int(render["sphere_rings"]),
        enter_editmode=False,
        align="WORLD",
        location=(0, 0, 0),
    )
    bpy.ops.object.shade_smooth()
    sphere = bpy.context.object
    camera_matrix = helpers["look_at_to_c2w"](
        camera_position=render["camera_position"],
        target_position=render["camera_target"],
        up_dir=render["camera_up"],
    )
    camera = helpers["create_camera"](camera_matrix, fov=float(render["fov_degrees"]))
    scene.camera = camera
    helpers["set_env_light"](
        str(Path(environment_map).expanduser().resolve()),
        strength=float(render["environment_strength"]),
    )
    return bpy, simple_exr, helpers, sphere, selected_device


def _create_material(
    bpy: Any,
    helpers: Mapping[str, Any],
    task: Mapping[str, Any],
    fixed: Mapping[str, Any],
):
    properties = task["material"]
    family = task["family"]
    if family == "diffuse_specular":
        return helpers["create_specular_roughness_material"](
            diffuse_color=tuple(properties["diffuse_color_rgb"]),
            specular_color=tuple(properties["specular_color_rgb"]),
            roughness=float(properties["roughness"]),
            material_name=f"material-{task['id']}",
        )

    material = bpy.data.materials.new(name=f"material-{task['id']}")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    values = {
        "Base Color": (*properties["base_color_rgb"], 1.0),
        "Subsurface Weight": fixed["subsurface_weight"],
        "Metallic": properties["metallic"],
        "Specular IOR Level": fixed["specular_ior_level"],
        "Specular Tint": (*fixed["specular_tint_rgb"], 1.0),
        "Roughness": properties["roughness"],
        "Anisotropic": fixed["anisotropic"],
        "Anisotropic Rotation": fixed["anisotropic_rotation"],
        "Sheen Weight": fixed["sheen_weight"],
        "Sheen Tint": (*fixed["sheen_tint_rgb"], 1.0),
        "Coat Weight": fixed["coat_weight"],
        "Coat Roughness": fixed["coat_roughness"],
        "Transmission Weight": properties["transmission_weight"],
        "Emission Color": (*fixed["emission_rgb"], 1.0),
        "Alpha": fixed["alpha"],
        "IOR": properties["ior"],
    }
    for input_name, value in values.items():
        node_input = bsdf.inputs.get(input_name)
        if node_input is None:
            raise RuntimeError(f"bpy 4.5.10 Principled BSDF is missing input {input_name!r}")
        node_input.default_value = value
    return material


def _validate_exr(path: Path, simple_exr: Any) -> dict[str, Any]:
    import numpy as np

    try:
        image = np.asarray(simple_exr.read_exr(str(path)))
    except Exception as error:
        raise RuntimeError(f"failed to read material EXR {path}: {error}") from error
    if image.shape != (256, 256, 3):
        raise ValueError(f"material EXR must be 256x256 RGB, got {image.shape}: {path}")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(f"material EXR must be floating point, got {image.dtype}: {path}")
    if not np.isfinite(image).all() or not (image >= 0).all():
        raise ValueError(f"material EXR must contain finite nonnegative linear RGB: {path}")
    return {
        "sha256": _sha256_file(path),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
    }


def _sample_metadata(plan_sha256: str, plan: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": RECORD_FORMAT,
        "pipeline_id": PIPELINE_ID,
        "plan_sha256": plan_sha256,
        "task": dict(task),
        "render": plan["recipe"]["render"],
        "preprocessing": plan["recipe"]["preprocessing"],
        "environment_map_sha256": plan["environment_map"]["sha256"],
    }


def _validate_completed_sample(
    root: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    task: Mapping[str, Any],
    simple_exr: Any,
) -> dict[str, Any]:
    sample_dir = root / Path(task["relative_exr_path"]).parent
    exr_path = root / task["relative_exr_path"]
    metadata_path = sample_dir / "metadata.json"
    record_path = root / "records" / f"{task['id']}.json"
    expected_metadata = _canonical_json(_sample_metadata(plan_sha256, plan, task)) + b"\n"
    if not metadata_path.is_file() or metadata_path.read_bytes() != expected_metadata:
        raise ValueError(f"sample metadata is missing or does not match its task: {metadata_path}")
    if not exr_path.is_file():
        raise FileNotFoundError(exr_path)
    output = _validate_exr(exr_path, simple_exr)
    expected_record = {
        "format": RECORD_FORMAT,
        "plan_sha256": plan_sha256,
        "task_sha256": task["task_sha256"],
        "id": task["id"],
        "relative_exr_path": task["relative_exr_path"],
        "output": output,
    }
    content = _canonical_json(expected_record) + b"\n"
    if record_path.exists():
        if not record_path.is_file() or record_path.read_bytes() != content:
            raise ValueError(f"sample record or EXR changed after publication: {record_path}")
    else:
        _atomic_write(record_path, content)
    return expected_record


def _render_task(
    root: Path,
    plan: Mapping[str, Any],
    plan_sha256: str,
    task: Mapping[str, Any],
    bpy: Any,
    simple_exr: Any,
    helpers: Mapping[str, Any],
    sphere: Any,
) -> dict[str, Any]:
    sample_dir = root / Path(task["relative_exr_path"]).parent
    exr_path = root / task["relative_exr_path"]
    metadata_path = sample_dir / "metadata.json"
    record_path = root / "records" / f"{task['id']}.json"
    expected_metadata = _canonical_json(_sample_metadata(plan_sha256, plan, task)) + b"\n"
    if record_path.exists():
        return _validate_completed_sample(root, plan, plan_sha256, task, simple_exr)

    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_or_match(metadata_path, expected_metadata)
    if exr_path.exists():
        return _validate_completed_sample(root, plan, plan_sha256, task, simple_exr)

    material = _create_material(
        bpy,
        helpers,
        task,
        plan["recipe"]["principled_fixed"],
    )
    sphere.data.materials.clear()
    sphere.data.materials.append(material)
    raw_path = sample_dir / f".{task['id']}.{uuid.uuid4().hex}.rgba.exr"
    processed_path = sample_dir / f".{task['id']}.{uuid.uuid4().hex}.rgb.exr"
    try:
        bpy.context.scene.cycles.seed = int(plan["recipe"]["render"]["cycles_seed"])
        bpy.context.scene.render.filepath = str(raw_path)
        with helpers["stdout_redirected"]():
            bpy.ops.render.render(write_still=True)

        import numpy as np

        rgba = np.asarray(simple_exr.read_exr(str(raw_path)))
        if rgba.shape != (256, 256, 4) or not np.issubdtype(rgba.dtype, np.floating):
            raise ValueError(f"Cycles output must be 256x256 RGBA float, got {rgba.shape}/{rgba.dtype}")
        if not np.isfinite(rgba).all():
            raise ValueError(f"Cycles output contains non-finite values for {task['id']}")
        alpha = rgba[..., 3:4]
        if not ((alpha >= 0) & (alpha <= 1)).all():
            raise ValueError(f"Cycles alpha is outside [0, 1] for {task['id']}")
        rgb = rgba[..., :3] * alpha
        if not (rgb >= 0).all():
            raise ValueError(f"Cycles output contains negative linear RGB for {task['id']}")
        simple_exr.write_exr(str(processed_path), np.ascontiguousarray(rgb, dtype=np.float32))
        with processed_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(processed_path, exr_path)
    finally:
        if raw_path.exists():
            raw_path.unlink()
        if processed_path.exists():
            processed_path.unlink()
        sphere.data.materials.clear()
        bpy.data.materials.remove(material)
    return _validate_completed_sample(root, plan, plan_sha256, task, simple_exr)


def render_material_sphere_shard(
    *,
    recipe_path: str | Path,
    environment_map: str | Path,
    output_dir: str | Path,
    seed: int,
    num_shards: int,
    shard_index: int,
    device: str = "cpu",
    device_index: int | None = None,
) -> Path:
    """Render one explicit shard, skipping only byte-verified completed tasks."""

    if device not in {"cpu", "cuda", "optix", "hip", "metal", "oneapi"}:
        raise ValueError(f"unsupported Cycles device backend: {device}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    root, plan, tasks, plan_sha256 = _load_planned_tasks(
        recipe_path=recipe_path,
        environment_map=environment_map,
        output_dir=output_dir,
        seed=seed,
        num_shards=num_shards,
    )
    bpy, simple_exr, helpers, sphere, selected_device = _setup_scene(
        plan, environment_map, device, device_index
    )
    selected = [task for task in tasks if task["shard"] == shard_index]
    records = []
    for completed, task in enumerate(selected, start=1):
        record = _render_task(
            root,
            plan,
            plan_sha256,
            task,
            bpy,
            simple_exr,
            helpers,
            sphere,
        )
        records.append(record)
        print(
            f"[shard {shard_index}] {completed}/{len(selected)} {task['id']}",
            flush=True,
        )
    receipt = {
        "format": "renderformer-material-sphere-shard-v1",
        "plan_sha256": plan_sha256,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "device": selected_device,
        "task_ids": [task["id"] for task in selected],
        "output_sha256": [record["output"]["sha256"] for record in records],
    }
    path = root / "shards" / f"shard-{shard_index:05d}.json"
    _write_or_match(path, _canonical_json(receipt) + b"\n")
    return path


def finalize_material_spheres(
    *,
    recipe_path: str | Path,
    environment_map: str | Path,
    output_dir: str | Path,
    seed: int,
    num_shards: int,
) -> Path:
    """Verify every planned EXR and publish the strict training JSONL manifest."""

    root, plan, tasks, plan_sha256 = _load_planned_tasks(
        recipe_path=recipe_path,
        environment_map=environment_map,
        output_dir=output_dir,
        seed=seed,
        num_shards=num_shards,
    )
    import simple_exr

    records = [
        _validate_completed_sample(root, plan, plan_sha256, task, simple_exr)
        for task in tasks
    ]
    manifest_bytes = b"".join(
        _canonical_json(
            {
                "id": task["id"],
                "path": task["relative_exr_path"],
                "split": task["split"],
            }
        )
        + b"\n"
        for task in tasks
    )
    manifest_path = root / "material-training.jsonl"
    _write_or_match(manifest_path, manifest_bytes)
    counts = {
        family: {
            split: sum(
                task["family"] == family and task["split"] == split for task in tasks
            )
            for split in ("train", "validation")
        }
        for family in FAMILY_NAMES
    }
    receipt = {
        "format": "renderformer-material-sphere-dataset-v1",
        "plan_sha256": plan_sha256,
        "tasks_sha256": plan["tasks_sha256"],
        "manifest": manifest_path.name,
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "counts": counts,
        "output_sha256": [record["output"]["sha256"] for record in records],
    }
    _write_or_match(
        root / "material-sphere-receipt.json",
        _canonical_json(receipt) + b"\n",
    )
    return manifest_path


__all__ = [
    "FAMILY_NAMES",
    "PIPELINE_ID",
    "describe_material_sphere_recipe",
    "finalize_material_spheres",
    "load_material_sphere_recipe",
    "plan_material_spheres",
    "render_material_sphere_shard",
]
