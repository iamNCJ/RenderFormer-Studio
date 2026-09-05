import json
from pathlib import Path
from typing import Any

import yaml

from renderformer.data.schemas.export_profile import ExportProfile
from renderformer.data.schemas.material import MaterialSemantic
from renderformer.data.schemas.scene import ObjectSemantic, SceneSemantic


def load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_export_profile(path: str | Path) -> ExportProfile:
    return ExportProfile.from_dict(load_mapping(path))


def load_scene_semantic(path: str | Path) -> SceneSemantic:
    data = load_mapping(path)
    objects = {}
    for name, obj_data in dict(data.get("objects", {})).items():
        objects[name] = ObjectSemantic(
            mesh_path=obj_data["mesh_path"],
            material=MaterialSemantic.from_dict(obj_data["material"]),
            is_background=bool(obj_data.get("is_background", False)),
            is_lighting=bool(obj_data.get("is_lighting", False)),
        )
    return SceneSemantic(
        scene_name=data["scene_name"],
        version=str(data.get("version", "")),
        objects=objects,
        has_env_map=bool(data.get("has_env_map", False)),
        has_volume=bool(data.get("has_volume", False)),
    )
