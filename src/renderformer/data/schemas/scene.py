from dataclasses import dataclass

from renderformer.data.schemas.material import MaterialSemantic


@dataclass(frozen=True)
class ObjectSemantic:
    mesh_path: str
    material: MaterialSemantic
    is_background: bool = False
    is_lighting: bool = False


@dataclass(frozen=True)
class SceneSemantic:
    scene_name: str
    version: str
    objects: dict[str, ObjectSemantic]
    has_env_map: bool = False
    has_volume: bool = False

    def feature_set(self) -> set[str]:
        features: set[str] = set()
        if self.has_env_map:
            features.add("env_map")
        if self.has_volume:
            features.add("volume")

        for obj in self.objects.values():
            material = obj.material
            features.add(f"material:{material.type.value}")
            features.add(f"source:{material.source.value}")
            features.add(f"spatial:{material.spatial_variation.value}")
            if material.use_heightmap:
                features.add("heightmap")

        return features
