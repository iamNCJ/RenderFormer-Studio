from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Dict, Any

@dataclass
class Transform:
    translation: dict[str, List[float]]  # x, y, z ranges
    rotation: dict[str, List[float]]     # x, y, z ranges
    scale: dict[str, List[float]]        # x, y, z ranges

@dataclass
class BackgroundObject:
    name: str
    mesh_path: str
    transform: Transform

@dataclass
class ObjectConstraints:
    sphere_scale_range: List[float]
    min_spacing: float
    max_attempts: int

@dataclass
class Objects:
    list_path: str
    base_path: str
    count: dict[str, int]  # min, max
    rotation: dict[str, List[float]]  # x, y, z ranges

@dataclass
class ObjectPlacement:
    bounding_mesh: str
    object_constraints: ObjectConstraints
    objects: Objects

@dataclass
class LookAt:
    target: List[float]
    deviation_range: List[float]

@dataclass
class Up:
    direction: List[float]
    deviation_range: List[float]

@dataclass
class Camera:
    shell_path: str
    look_at: LookAt
    up: Up
    fov: List[float]
    distance_range: List[float]

@dataclass
class Lighting:
    mesh_path: str
    shell_path: str
    min_num: int
    max_num: int
    look_at: LookAt  # Reusing the LookAt dataclass from camera config
    distance_range: List[float]
    power_range: List[float]
    rotation: dict[str, List[float]]  # x, y, z ranges
    scale: dict[str, List[float]]     # x, y, z ranges
    white_light: bool = False  # If True, all lights are pure white (no random color tint)

@dataclass
class Material:
    # Diffuse-specular parameters (existing)
    spec_diffuse_sum_range: List[float]
    spec_diffuse_ratio_range: List[float]
    roughness_range: List[float]
    diffuse_rgb_range: List[float]
    smooth_shading_ratio: float
    rand_tri_diffuse_ratio: float
    texture_map_ratio: Optional[float]
    texture_map_list_path: Optional[str]
    texture_map_base_path: Optional[str]
    
    # Metallic-roughness parameters (new)
    base_color_range: Optional[List[float]] = None  # RGB range for base_color, default to diffuse_rgb_range
    metallic_range: Optional[List[float]] = None  # Metallic range [0, 1], default to [0.0, 1.0]
    
    # Transmission parameters (for metallic_roughness_transmission)
    transmission_weight_range: Optional[List[float]] = None  # Transmission weight range [0, 1], default to [0.0, 1.0]
    ior_range: Optional[List[float]] = None  # IOR range [1, 2.5], default to [1.0, 2.5]
    
    # Heightmap support (optional, for SVBRDF materials only)
    use_heightmap: bool = False  # Whether to use heightmap for displacement mapping (only applies to SVBRDF materials)
    
    # Allowed material types (optional, if None, all types are allowed)
    allowed_material_types: Optional[List[str]] = None  # List of allowed material types, e.g. ["svbrdf_diffuse_specular", "homo_diffuse_specular"]
    
    # Material type probabilities (optional, if None, equal probability)
    # Dictionary mapping material type to probability weight (will be automatically normalized)
    # You can use integers or floats - they will be normalized automatically
    # Example: {"svbrdf_diffuse_specular": 3, "homo_diffuse_specular": 2, ...} or {"svbrdf_diffuse_specular": 0.3, ...}
    material_type_probabilities: Optional[Dict[str, float]] = None
    
    # If True, specular color uses the same value for all RGB channels (monochromatic specular)
    mono_specular: bool = False
    
    def __post_init__(self):
        """Set default values for optional fields."""
        if self.base_color_range is None:
            self.base_color_range = self.diffuse_rgb_range
        if self.metallic_range is None:
            self.metallic_range = [0.0, 1.0]
        if self.transmission_weight_range is None:
            self.transmission_weight_range = [0.0, 1.0]
        if self.ior_range is None:
            self.ior_range = [1.0, 2.5]
        
        # Validate allowed_material_types if provided
        if self.allowed_material_types is not None:
            from renderformer.data.blender.material_types import ALL_MATERIAL_TYPES
            for mt in self.allowed_material_types:
                if mt not in ALL_MATERIAL_TYPES:
                    raise ValueError(f"Unknown material type in allowed_material_types: {mt}. Valid types: {ALL_MATERIAL_TYPES}")
        
        # Validate material_type_probabilities if provided
        if self.material_type_probabilities is not None:
            from renderformer.data.blender.material_types import ALL_MATERIAL_TYPES
            for mt, prob in self.material_type_probabilities.items():
                if mt not in ALL_MATERIAL_TYPES:
                    raise ValueError(f"Unknown material type in material_type_probabilities: {mt}. Valid types: {ALL_MATERIAL_TYPES}")
                if prob < 0:
                    raise ValueError(f"Material type probability weight must be non-negative, got {prob} for {mt}. Note: weights will be automatically normalized.")
            
            # If allowed_material_types is specified, check that probabilities only include allowed types
            if self.allowed_material_types is not None:
                for mt in self.material_type_probabilities.keys():
                    if mt not in self.allowed_material_types:
                        raise ValueError(f"Material type {mt} in material_type_probabilities is not in allowed_material_types")

@dataclass
class EnvMap:
    list_path: str
    base_path: str
    rotation: dict[str, List[float]]  # x, y, z ranges
    strength_range: List[float]

@dataclass
class VolumePlacement:
    probability: float  # Probability of volume appearing (0-1)
    count: dict[str, int]  # min, max - number of volumes
    transform: Transform  # Random scale/rotate/translate
    scattering_scale_range: List[float]  # [min, max] for scattering scale
    absorption_scale_range: List[float]  # [min, max] for absorption scale
    volume_generation_params: Optional[Dict[str, Any]] = None  # Optional: freq, tau0, tau1, etc.

@dataclass
class SceneTemplate:
    scene_name: str
    version: str
    background_scene: List[BackgroundObject]
    object_placement: ObjectPlacement
    camera: Camera
    lighting: Lighting
    material: Material
    env_map: Optional[EnvMap] = None
    volume: Optional[VolumePlacement] = None
