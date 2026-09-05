from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class TransformConfig:
    translation_x: float
    translation_y: float
    translation_z: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    scale_x: float
    scale_y: float
    scale_z: float
    normalize: bool = True

@dataclass
class MaterialConfig:
    # Material type (one of the supported types in material_types.py)
    material_type: str
    
    # SVBRDF related fields (for svbrdf_* types)
    texture_map_path: Optional[str] = None
    texture_map_aug_random_seed: Optional[int] = None
    use_heightmap: bool = False  # Whether to use heightmap for displacement mapping
    
    # Homogeneous parameters (use corresponding field based on material_type)
    # For diffuse_specular types
    diffuse_specular_params: Optional[Dict[str, Any]] = None  # {diffuse: [r,g,b], specular: [r,g,b], roughness: float}
    
    # For metallic_roughness types
    metallic_roughness_params: Optional[Dict[str, Any]] = None  # {base_color: [r,g,b], metallic: float, roughness: float}
    
    # For metallic_roughness_transmission type
    metallic_roughness_transmission_params: Optional[Dict[str, Any]] = None  # {base_color: [r,g,b], metallic: float, roughness: float, transmission_weight: float, ior: float}

    # For homo_measured_brdf type: path to .npy latent (shape (9,) or equivalent flat 9 values)
    measured_brdf_npy_path: Optional[str] = None
    
    # Common fields
    emissive: Optional[List[float]] = None  # Default to [0,0,0]
    smooth_shading: bool = True
    rand_tri_diffuse_seed: Optional[int] = None  # For triangle group random sampling (homogeneous only)
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.emissive is None:
            self.emissive = [0.0, 0.0, 0.0]
        
        # Validate material type
        from renderformer.data.blender.material_types import ALL_MATERIAL_TYPES
        if self.material_type not in ALL_MATERIAL_TYPES:
            raise ValueError(f"Unknown material_type: {self.material_type}")
        
        # Validate SVBRDF fields
        from renderformer.data.blender.material_types import supports_svbrdf
        if supports_svbrdf(self.material_type):
            if self.texture_map_path is None:
                raise ValueError(f"SVBRDF material type {self.material_type} requires texture_map_path")
            # texture_map_aug_random_seed can be None for no-crop mode (e.g. Blender animation export)
            # Heightmap can only be used with SVBRDF materials
            if self.use_heightmap and self.texture_map_path is None:
                raise ValueError(f"use_heightmap=True requires texture_map_path for SVBRDF materials")
        else:
            # Homogeneous materials should not have texture_map_path
            if self.texture_map_path is not None:
                raise ValueError(f"Homogeneous material type {self.material_type} should not have texture_map_path")
            # Heightmap can only be used with SVBRDF materials
            if self.use_heightmap:
                raise ValueError(f"use_heightmap=True is only supported for SVBRDF materials, not {self.material_type}")
        
        # Validate homogeneous parameters based on material type (only for homogeneous types)
        from renderformer.data.blender.material_types import supports_homogeneous
        if supports_homogeneous(self.material_type):
            # Only homogeneous materials need these parameters
            if self.material_type == "homo_diffuse_specular":
                if self.diffuse_specular_params is None:
                    raise ValueError(f"Material type {self.material_type} requires diffuse_specular_params")
            elif self.material_type == "homo_metallic_roughness":
                if self.metallic_roughness_params is None:
                    raise ValueError(f"Material type {self.material_type} requires metallic_roughness_params")
            elif self.material_type == "homo_metallic_roughness_transmission":
                if self.metallic_roughness_transmission_params is None:
                    raise ValueError(f"Material type {self.material_type} requires metallic_roughness_transmission_params")
            elif self.material_type == "homo_measured_brdf":
                if self.measured_brdf_npy_path is None:
                    raise ValueError(f"Material type {self.material_type} requires measured_brdf_npy_path")

@dataclass
class ObjectConfig:
    mesh_path: str
    transform: TransformConfig
    is_background: bool
    is_lighting: bool
    material: MaterialConfig

@dataclass
class CameraConfig:
    position: List[float]
    look_at: List[float]
    up: List[float]
    fov: float

@dataclass
class EnvMapConfig:
    env_map_path: str
    rotation_x: float
    rotation_y: float
    rotation_z: float
    strength: float

@dataclass
class VolumeConfig:
    vdb_path: str  # VDB file path (saved in cache directory)
    transform: TransformConfig  # Volume transform in scene
    scattering_scale: List[float]  # Scattering scale (RGB, 3 values)
    absorption_scale: List[float]  # Absorption scale (RGB, 3 values)
    seed: int  # Seed used to generate volume
    transpose_written: bool = True  # Whether VDB was written with transpose (needed for reading)

@dataclass
class GeneratedConfig:
    scene_name: str
    version: str
    objects: Dict[str, ObjectConfig]
    cameras: List[CameraConfig]  # Changed from single camera to a list of cameras
    lighting: List[ObjectConfig]
    env_map: Optional[EnvMapConfig] = None
    volumes: Optional[List[VolumeConfig]] = None  # List of volumes in the scene
    wall_holes: Optional[Dict[str, List[Dict[str, List[float]]]]] = None
