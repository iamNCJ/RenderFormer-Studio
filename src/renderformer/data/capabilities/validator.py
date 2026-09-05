from dataclasses import dataclass

from renderformer.data.schemas.export_profile import ExportProfile, UnsupportedPolicy
from renderformer.data.schemas.scene import SceneSemantic


@dataclass(frozen=True)
class ValidationIssue:
    object_name: str
    reason: str


class UnsupportedFeatureError(ValueError):
    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        message = "; ".join(f"{issue.object_name}: {issue.reason}" for issue in issues)
        super().__init__(message)


def validate_scene_for_export(
    scene: SceneSemantic,
    profile: ExportProfile,
) -> list[ValidationIssue]:
    """Validate a scene against an export profile.

    Pure profile-driven: every check reads `profile.allow_*` or `profile.allowed_*`
    fields. No `if profile.format is V1/V2` branches; the profile yaml is the sole
    source of truth for what each format supports.

    Issues escalate to `UnsupportedFeatureError` when
    `profile.unsupported_policy is ERROR`; otherwise they are returned for the
    caller to log or drop.
    """
    issues: list[ValidationIssue] = []

    if scene.has_volume and not profile.allow_volume:
        issues.append(
            ValidationIssue(
                object_name="<scene>",
                reason="scene has volume but profile.allow_volume is False",
            )
        )
    if scene.has_env_map and not profile.allow_env_map:
        issues.append(
            ValidationIssue(
                object_name="<scene>",
                reason="scene has env_map but profile.allow_env_map is False",
            )
        )

    for object_name, obj in scene.objects.items():
        material = obj.material
        if material.type not in profile.allowed_material_types:
            issues.append(
                ValidationIssue(
                    object_name=object_name,
                    reason=(
                        f"material.type={material.type.value} is not in "
                        f"profile.allowed_material_types"
                    ),
                )
            )
        if material.spatial_variation not in profile.allowed_spatial_variation:
            issues.append(
                ValidationIssue(
                    object_name=object_name,
                    reason=(
                        f"material.spatial_variation={material.spatial_variation.value} "
                        f"is not in profile.allowed_spatial_variation"
                    ),
                )
            )
        if material.use_heightmap and not profile.allow_heightmap:
            issues.append(
                ValidationIssue(
                    object_name=object_name,
                    reason="material.use_heightmap=True but profile.allow_heightmap is False",
                )
            )

    if issues and profile.unsupported_policy is UnsupportedPolicy.ERROR:
        raise UnsupportedFeatureError(issues)
    return issues
