import os
import math
import json
import numpy as np
import trimesh
import trimesh.visual
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List
import random
from renderformer.data.blender.template_dataclass import *
from renderformer.data.blender.volume_utils import generate_volume_vdb, create_volume_occupancy_mesh
from renderformer.data.blender.generated_config import TransformConfig, MaterialConfig, ObjectConfig, CameraConfig, GeneratedConfig, EnvMapConfig, VolumeConfig
import cv2
import torch
import torchvision.transforms.functional as TF

def random_transform(transform: Transform) -> Dict[str, float]:
    """Generate random transformation values within specified ranges"""
    result = {}
    for axis in ['x', 'y', 'z']:
        result[f'translation_{axis}'] = random.uniform(transform.translation[axis][0], transform.translation[axis][1])
        result[f'rotation_{axis}'] = random.uniform(transform.rotation[axis][0], transform.rotation[axis][1])
        result[f'scale_{axis}'] = random.uniform(transform.scale[axis][0], transform.scale[axis][1])
    return result

def sample_material(material: Material, texture_list: List[str], exclude_transmission: bool = False) -> MaterialConfig:
    """
    Sample material properties. First randomly selects material type based on probabilities,
    unless exclude_transmission=True or allowed_material_types is specified, then filters accordingly.
    Probabilities are automatically renormalized after filtering.
    
    Args:
        material: Material configuration
        texture_list: List of available textures
        exclude_transmission: If True, exclude transmission material type (for background objects that are not watertight)
    """
    from renderformer.data.blender.material_types import (
        MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
        MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
        MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
        MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
        MATERIAL_TYPE_HOMO_MEASURED_BRDF,
        ALL_MATERIAL_TYPES,
    )
    
    # Start with all material types
    available_types = ALL_MATERIAL_TYPES.copy()
    # Measured latent material is a manual hack type, not for random sampling.
    available_types = [mt for mt in available_types if mt != MATERIAL_TYPE_HOMO_MEASURED_BRDF]

    if (
        not texture_list
        and material.allowed_material_types is None
        and material.material_type_probabilities is None
    ):
        available_types = [MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR]
    
    # Filter by allowed_material_types from template if specified
    if material.allowed_material_types is not None:
        available_types = [mt for mt in available_types if mt in material.allowed_material_types]
        if len(available_types) == 0:
            raise ValueError(f"No valid material types available. allowed_material_types: {material.allowed_material_types}")
    
    # Filter by exclude_transmission if needed
    if exclude_transmission:
        available_types = [mt for mt in available_types if mt != MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION]
        if len(available_types) == 0:
            raise ValueError("No material types available after excluding transmission")
    
    # Build probability distribution for available types
    if material.material_type_probabilities is not None:
        # Extract probability weights for available types only (weights will be normalized automatically)
        probabilities = {}
        for mt in available_types:
            if mt in material.material_type_probabilities:
                probabilities[mt] = material.material_type_probabilities[mt]
        
        # If some available types don't have explicit probabilities, assign equal weight to them
        types_without_prob = [mt for mt in available_types if mt not in probabilities]
        if types_without_prob:
            # Calculate total weight of explicitly specified types
            total_specified_weight = sum(probabilities.values()) if probabilities else 0.0
            
            # Assign equal weight to types without explicit probability
            # If total_specified_weight >= 1.0, we still assign equal small weights
            # Otherwise, distribute remaining weight equally
            if total_specified_weight >= 1.0:
                # All weight is already assigned, assign small equal weight to remaining types
                equal_weight = 1.0 / len(types_without_prob)
            else:
                # Distribute remaining weight equally
                equal_weight = (1.0 - total_specified_weight) / len(types_without_prob)
            
            for mt in types_without_prob:
                probabilities[mt] = equal_weight
        
        # Normalize probabilities (in case total != 1.0)
        total_weight = sum(probabilities.values())
        if total_weight > 0:
            for mt in probabilities:
                probabilities[mt] /= total_weight
        else:
            # Fallback to equal probability if all weights are zero
            equal_prob = 1.0 / len(available_types)
            probabilities = {mt: equal_prob for mt in available_types}
        
        # Sample based on probabilities
        material_type = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
    else:
        # Equal probability among available types
        material_type = random.choice(available_types)
    
    # Sample based on material type
    if material_type == MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR:
        return _sample_svbrdf_diffuse_specular(material, texture_list)
    elif material_type == MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR:
        return _sample_homo_diffuse_specular(material)
    elif material_type == MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS:
        return _sample_svbrdf_metallic_roughness(material, texture_list)
    elif material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS:
        return _sample_homo_metallic_roughness(material)
    elif material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION:
        return _sample_homo_metallic_roughness_transmission(material)
    else:
        raise ValueError(f"Unknown material type: {material_type}")


def _sample_svbrdf_diffuse_specular(material: Material, texture_list: List[str]) -> MaterialConfig:
    """Sample SVBRDF diffuse-specular material."""
    from renderformer.data.blender.material_types import MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR
    
    assert len(texture_list) > 0, "Texture list is empty"
    assert material.texture_map_base_path is not None, "Texture map base path is not provided"
    
    texture_map_path = str(Path(material.texture_map_base_path) / random.choice(texture_list))
    texture_map_aug_random_seed = random.randint(0, 10000)
    
    smooth_shading = random.random() < material.smooth_shading_ratio
    
    return MaterialConfig(
        material_type=MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
        texture_map_path=texture_map_path,
        texture_map_aug_random_seed=texture_map_aug_random_seed,
        use_heightmap=material.use_heightmap,
        diffuse_specular_params=None,  # Not used for SVBRDF
        metallic_roughness_params=None,
        metallic_roughness_transmission_params=None,
        emissive=[0.0, 0.0, 0.0],
        smooth_shading=smooth_shading,
        rand_tri_diffuse_seed=None,
    )


def _sample_homo_diffuse_specular(material: Material) -> MaterialConfig:
    """Sample homogeneous diffuse-specular material."""
    from renderformer.data.blender.material_types import MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR
    
    spec_diffuse_sum = random.uniform(*material.spec_diffuse_sum_range)
    spec_diffuse_ratio = random.uniform(*material.spec_diffuse_ratio_range)
    log_bias = 0.05
    eps = 0.005
    roughness = np.clip(
        np.exp(random.uniform(np.log(material.roughness_range[0] + log_bias), 
                            np.log(material.roughness_range[1] + log_bias + eps))) - log_bias,
        material.roughness_range[0],
        material.roughness_range[1]
    )
    diffuse_rgb = [random.uniform(*material.diffuse_rgb_range) for _ in range(3)]
    if material.mono_specular:
        specular = [spec_diffuse_sum * spec_diffuse_ratio] * 3
    else:
        specular = [spec_diffuse_sum * spec_diffuse_ratio * random.uniform(0., 1.) for _ in range(3)]
    diffuse_max = spec_diffuse_sum * (1 - spec_diffuse_ratio)
    diffuse = [diffuse_max * c for c in diffuse_rgb]
    
    smooth_shading = random.random() < material.smooth_shading_ratio
    rand_tri_diffuse = random.random() < material.rand_tri_diffuse_ratio if smooth_shading else False
    rand_seed = random.randint(0, 10000) if rand_tri_diffuse else None
    
    return MaterialConfig(
        material_type=MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
        texture_map_path=None,
        texture_map_aug_random_seed=None,
        diffuse_specular_params={
            'diffuse': diffuse,
            'specular': specular,
            'roughness': roughness,
        },
        metallic_roughness_params=None,
        metallic_roughness_transmission_params=None,
        emissive=[0.0, 0.0, 0.0],
        smooth_shading=smooth_shading,
        rand_tri_diffuse_seed=rand_seed,
    )


def _sample_svbrdf_metallic_roughness(material: Material, texture_list: List[str]) -> MaterialConfig:
    """Sample SVBRDF metallic-roughness material."""
    from renderformer.data.blender.material_types import MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS
    
    assert len(texture_list) > 0, "Texture list is empty"
    assert material.texture_map_base_path is not None, "Texture map base path is not provided"
    
    texture_map_path = str(Path(material.texture_map_base_path) / random.choice(texture_list))
    texture_map_aug_random_seed = random.randint(0, 10000)
    
    smooth_shading = random.random() < material.smooth_shading_ratio
    
    return MaterialConfig(
        material_type=MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
        texture_map_path=texture_map_path,
        texture_map_aug_random_seed=texture_map_aug_random_seed,
        use_heightmap=material.use_heightmap,
        diffuse_specular_params=None,
        metallic_roughness_params=None,  # Not used for SVBRDF
        metallic_roughness_transmission_params=None,
        emissive=[0.0, 0.0, 0.0],
        smooth_shading=smooth_shading,
        rand_tri_diffuse_seed=None,
    )


def _sample_homo_metallic_roughness(material: Material) -> MaterialConfig:
    """Sample homogeneous metallic-roughness material."""
    from renderformer.data.blender.material_types import MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS
    
    base_color_rgb = [random.uniform(*material.base_color_range) for _ in range(3)]
    metallic = random.uniform(*material.metallic_range)
    log_bias = 0.05
    eps = 0.005
    roughness = np.clip(
        np.exp(random.uniform(np.log(material.roughness_range[0] + log_bias), 
                            np.log(material.roughness_range[1] + log_bias + eps))) - log_bias,
        material.roughness_range[0],
        material.roughness_range[1]
    )
    
    smooth_shading = random.random() < material.smooth_shading_ratio
    rand_tri_diffuse = random.random() < material.rand_tri_diffuse_ratio if smooth_shading else False
    rand_seed = random.randint(0, 10000) if rand_tri_diffuse else None
    
    return MaterialConfig(
        material_type=MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
        texture_map_path=None,
        texture_map_aug_random_seed=None,
        diffuse_specular_params=None,
        metallic_roughness_params={
            'base_color': base_color_rgb,
            'metallic': metallic,
            'roughness': roughness,
        },
        metallic_roughness_transmission_params=None,
        emissive=[0.0, 0.0, 0.0],
        smooth_shading=smooth_shading,
        rand_tri_diffuse_seed=rand_seed,
    )


def _sample_homo_metallic_roughness_transmission(material: Material) -> MaterialConfig:
    """Sample homogeneous metallic-roughness-transmission material."""
    from renderformer.data.blender.material_types import MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION
    
    base_color_rgb = [random.uniform(*material.base_color_range) for _ in range(3)]
    metallic = random.uniform(*material.metallic_range)
    log_bias = 0.05
    eps = 0.005
    roughness = np.clip(
        np.exp(random.uniform(np.log(material.roughness_range[0] + log_bias), 
                            np.log(material.roughness_range[1] + log_bias + eps))) - log_bias,
        material.roughness_range[0],
        material.roughness_range[1]
    )
    transmission_weight = random.uniform(*material.transmission_weight_range)
    ior = random.uniform(*material.ior_range)
    
    smooth_shading = random.random() < material.smooth_shading_ratio
    rand_tri_diffuse = random.random() < material.rand_tri_diffuse_ratio if smooth_shading else False
    rand_seed = random.randint(0, 10000) if rand_tri_diffuse else None
    
    return MaterialConfig(
        material_type=MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
        texture_map_path=None,
        texture_map_aug_random_seed=None,
        diffuse_specular_params=None,
        metallic_roughness_params=None,
        metallic_roughness_transmission_params={
            'base_color': base_color_rgb,
            'metallic': metallic,
            'roughness': roughness,
            'transmission_weight': transmission_weight,
            'ior': ior,
        },
        emissive=[0.0, 0.0, 0.0],
        smooth_shading=smooth_shading,
        rand_tri_diffuse_seed=rand_seed,
    )

def sample_env_map(env_map: EnvMap, env_map_list: List[str]) -> EnvMapConfig:
    """Sample environment map properties: randomly select an env map, rotation, and strength"""
    # Randomly select an environment map from the list
    assert len(env_map_list) > 0, "Environment map list is empty"
    assert env_map.base_path is not None, "Environment map base path is not provided"
    env_map_path = str(Path(env_map.base_path) / f"{random.choice(env_map_list)}.exr")
    
    # Sample rotation for each axis
    rotation_x = random.uniform(env_map.rotation['x'][0], env_map.rotation['x'][1])
    rotation_y = random.uniform(env_map.rotation['y'][0], env_map.rotation['y'][1])
    rotation_z = random.uniform(env_map.rotation['z'][0], env_map.rotation['z'][1])
    
    # Sample strength
    strength = random.uniform(env_map.strength_range[0], env_map.strength_range[1])
    
    return EnvMapConfig(
        env_map_path=env_map_path,
        rotation_x=rotation_x,
        rotation_y=rotation_y,
        rotation_z=rotation_z,
        strength=strength
    )

def generate_background_object_config(obj: BackgroundObject, material: Material, texture_list: List[str]) -> ObjectConfig:
    """Generate configuration for a single object"""
    transform_dict = random_transform(obj.transform)
    transform = TransformConfig(
        translation_x=transform_dict['translation_x'],
        translation_y=transform_dict['translation_y'],
        translation_z=transform_dict['translation_z'],
        rotation_x=transform_dict['rotation_x'],
        rotation_y=transform_dict['rotation_y'],
        rotation_z=transform_dict['rotation_z'],
        scale_x=transform_dict['scale_x'],
        scale_y=transform_dict['scale_y'],
        scale_z=transform_dict['scale_z']
    )
    # Background objects should not use transmission materials (not watertight)
    material_config = sample_material(material, texture_list, exclude_transmission=True)
    return ObjectConfig(
        mesh_path=obj.mesh_path,
        transform=transform,
        is_background=True,
        is_lighting=False,
        material=material_config
    )

def generate_camera_config(camera: Camera) -> CameraConfig:
    """Generate camera configuration"""
    # Generate random camera position on shell
    shell_mesh: trimesh.Trimesh = trimesh.load(camera.shell_path)  # type: ignore
    shell_mesh.apply_scale(0.5)
    position = trimesh.sample.sample_surface(shell_mesh, 1)[0][0]

    # Add deviation to look_at target
    target = np.array(camera.look_at.target).astype(float)
    deviation = np.random.uniform(
        camera.look_at.deviation_range[0],
        camera.look_at.deviation_range[1],
        3
    )
    target += deviation

    # Generate up vector with deviation
    up = np.array(camera.up.direction).astype(float)
    up_deviation = np.random.uniform(
        camera.up.deviation_range[0],
        camera.up.deviation_range[1],
        3
    )
    up += up_deviation
    up = up / np.linalg.norm(up)

    vdir = target - position
    vdir = vdir / np.linalg.norm(vdir)
    position = target - vdir * np.random.uniform(camera.distance_range[0], camera.distance_range[1])

    # Modified FOV sampling - mean at 1/4 of range, 2 sigma covers half the range
    # fov_range = camera.fov[1] - camera.fov[0]
    # fov_mean = camera.fov[0] + fov_range * 0.25  # mean at 1/4 of range
    # fov_std = fov_range * 0.15  # 2 sigma = range/2, so sigma = range/4
    # print(f"FOV range: {fov_range}, mean: {fov_mean}, std: {fov_std}")
    # fov = np.random.normal(fov_mean, fov_std)
    # fov = np.clip(fov, camera.fov[0], camera.fov[1])
    fov = np.random.uniform(camera.fov[0], camera.fov[1])
    # print(f"FOV: {fov}")

    return CameraConfig(
        position=position.tolist(),
        look_at=target.tolist(),
        up=up.tolist(),
        fov=fov
    )

def generate_lighting_config(lighting: Lighting, num_lights: int, old_lighting_mode: bool=False) -> List[ObjectConfig]:
    """Generate lighting configuration"""
    # Handle case when num_lights is 0
    if num_lights == 0:
        return []
    
    lights = []
    shell_mesh: trimesh.Trimesh = trimesh.load(lighting.shell_path)  # type: ignore
    shell_mesh.apply_scale(0.5)
    
    # Change uniform sampling to poisson for number of lights
    if num_lights > 1:
        # Use Poisson distribution for light spacing
        points = []
        while len(points) < num_lights:
            position = trimesh.sample.sample_surface(shell_mesh, 1)[0][0]
            if not points:  # First point always accepted
                points.append(position)
            else:
                # Check minimum distance from existing points
                min_dist = min(np.linalg.norm(position - p) for p in points)
                if min_dist > 0.3:  # Minimum spacing between lights
                    points.append(position)
    else:
        points = [trimesh.sample.sample_surface(shell_mesh, 1)[0][0]]
    
    # Store original num_lights for power calculation
    original_num_lights = num_lights
    if old_lighting_mode:
        num_lights = 1  # do not anneal the power according to the number of lights in old mode
    
    for position in points:
        # Add deviation to look_at target
        target = np.array(lighting.look_at.target).astype(float)
        deviation = np.random.uniform(
            lighting.look_at.deviation_range[0],
            lighting.look_at.deviation_range[1],
            3
        )
        target += deviation
        
        # Adjust position based on distance range
        direction = position - target
        direction = direction / np.linalg.norm(direction)
        distance = np.random.uniform(lighting.distance_range[0], lighting.distance_range[1])
        position = target + direction * distance
        
        # Generate random rotation and scale
        rotation = {axis: np.random.uniform(lighting.rotation[axis][0], lighting.rotation[axis][1]) for axis in ['x', 'y', 'z']}
        scale = {axis: np.random.uniform(lighting.scale[axis][0], lighting.scale[axis][1]) for axis in ['x', 'y', 'z']}
        
        # Calculate power: use original_num_lights to avoid division by zero
        # If original_num_lights is 0, we already returned early, so this is safe
        power = np.random.uniform(lighting.power_range[0], lighting.power_range[1]) / max(original_num_lights, 1)  # anneal the power according to the number of lights
        if lighting.white_light:
            normalized_lighting_rgb = [1.0, 1.0, 1.0]
        else:
            lighting_rgb = [random.uniform(0., 1.) for _ in range(3)]
            max_val = max(lighting_rgb)
            if max_val > 0.0:
                normalized_lighting_rgb = [p / max_val for p in lighting_rgb]
            else:  # extreme case, all lighting_rgb are 0.0
                normalized_lighting_rgb = [1.0, 1.0, 1.0]
        power = [power * p for p in normalized_lighting_rgb]
        
        transform = TransformConfig(
            translation_x=position[0],
            translation_y=position[1],
            translation_z=position[2],
            rotation_x=rotation['x'],
            rotation_y=rotation['y'],
            rotation_z=rotation['z'],
            scale_x=scale['x'],
            scale_y=scale['y'],
            scale_z=scale['z']
        )
        from renderformer.data.blender.material_types import MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR
        material_config = MaterialConfig(
            material_type=MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
            texture_map_path=None,
            texture_map_aug_random_seed=None,
            diffuse_specular_params={
                'diffuse': [1.0, 1.0, 1.0],
                'specular': [0.0, 0.0, 0.0],
                'roughness': 1.0,
            },
            metallic_roughness_params=None,
            metallic_roughness_transmission_params=None,
            emissive=power,
            smooth_shading=False,
            rand_tri_diffuse_seed=None
        )
        lights.append(ObjectConfig(
            mesh_path=lighting.mesh_path,
            transform=transform,
            is_background=False,
            is_lighting=True,
            material=material_config
        ))
    
    return lights

def normalize_to_unit_sphere(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Normalize mesh to fit in a unit sphere centered at origin"""
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    bounding_sphere_radius = np.linalg.norm(mesh.vertices, ord=2, axis=-1).max() * 2.
    mesh.vertices = mesh.vertices / bounding_sphere_radius

    return mesh

def place_object_in_bounds(
    bounding_mesh: trimesh.Trimesh,
    constraints: ObjectConstraints,
    rotation_ranges: Dict[str, List[float]]
) -> TransformConfig:
    """Place an object within bounds using bounding sphere approach"""
    # Get bounding box extents
    bounds = bounding_mesh.bounds
    box_extents = bounds[1] - bounds[0]
    box_center = (bounds[1] + bounds[0]) / 2
    
    # Normalize object to unit sphere
    # normalized_mesh = normalize_to_unit_sphere(obj_mesh)
    
    # Random sphere scale
    sphere_scale = np.random.uniform(constraints.sphere_scale_range[0], constraints.sphere_scale_range[1])
    
    # Calculate valid range for sphere center
    # Subtract sphere radius from box extents to ensure sphere fits
    valid_extents = box_extents # - sphere_scale  # change to allow the sphere to be touching or outside the box
    
    # Random position within valid range
    position = np.random.uniform(-valid_extents/2, valid_extents/2) + box_center
    
    # Random rotation
    rotation = {
        axis: np.random.uniform(rotation_ranges[axis][0], rotation_ranges[axis][1])
        for axis in ['x', 'y', 'z']
    }
    
    return TransformConfig(
        translation_x=position[0],
        translation_y=position[1],
        translation_z=position[2],
        rotation_x=rotation['x'],
        rotation_y=rotation['y'],
        rotation_z=rotation['z'],
        scale_x=sphere_scale,
        scale_y=sphere_scale,
        scale_z=sphere_scale
    )


def generate_volume_config(
    volume_placement: VolumePlacement,
    bounding_mesh: trimesh.Trimesh,
    cache_dir: str,
    volume_idx: int
) -> VolumeConfig:
    """
    Generate a volume configuration with VDB file.
    
    Args:
        volume_placement: Volume placement configuration from template
        bounding_mesh: Bounding mesh for placement
        cache_dir: Cache directory to save VDB file
        volume_idx: Index of this volume
    
    Returns:
        VolumeConfig with vdb_path, transform, scales, and seed
    """
    # Generate random seed for this volume
    volume_seed = random.randint(0, 2**31 - 1)
    
    # Get volume generation parameters (with defaults)
    gen_params = volume_placement.volume_generation_params or {}
    freq = gen_params.get('freq', 2.0)
    tau0 = gen_params.get('tau0', 0.35)
    tau1 = gen_params.get('tau1', 0.65)
    alpha = gen_params.get('alpha', 1.0)
    use_sphere_falloff = gen_params.get('use_sphere_falloff', True)
    transpose = gen_params.get('transpose', True)
    
    # Generate random scattering and absorption scales as RGB values (before generating VDB so we can include in metadata)
    scattering_scale = [
        np.random.uniform(
            volume_placement.scattering_scale_range[0],
            volume_placement.scattering_scale_range[1]
        ) for _ in range(3)  # Generate 3 values for RGB
    ]
    absorption_scale = [
        np.random.uniform(
            volume_placement.absorption_scale_range[0],
            volume_placement.absorption_scale_range[1]
        ) for _ in range(3)  # Generate 3 values for RGB
    ]
    
    # Generate VDB file (with scales in metadata)
    vdb_path, vdb_meta = generate_volume_vdb(
        output_dir=cache_dir,
        seed=volume_seed,
        freq=freq,
        tau0=tau0,
        tau1=tau1,
        alpha=alpha,
        use_sphere_falloff=use_sphere_falloff,
        transpose=transpose,
        scattering_scale=scattering_scale,
        absorption_scale=absorption_scale
    )
    
    # Generate random transform - volume scale and position are configured separately from object scale
    # Get bounding box extents for position calculation
    bounds = bounding_mesh.bounds
    box_extents = bounds[1] - bounds[0]
    box_center = (bounds[1] + bounds[0]) / 2
    
    # Volume's canonical bbox is [-0.5, 0.5]^3, so its size is 1.0 in each dimension
    # Scale is directly from config (not multiplied by box_extents) - allows manual control
    scale_x = random.uniform(volume_placement.transform.scale['x'][0], volume_placement.transform.scale['x'][1])
    scale_y = random.uniform(volume_placement.transform.scale['y'][0], volume_placement.transform.scale['y'][1])
    scale_z = random.uniform(volume_placement.transform.scale['z'][0], volume_placement.transform.scale['z'][1])
    
    # Position: translation values from config are relative to box_center
    # Config values are offsets from box_center (in world units)
    translation_x = box_center[0] + random.uniform(volume_placement.transform.translation['x'][0], volume_placement.transform.translation['x'][1])
    translation_y = box_center[1] + random.uniform(volume_placement.transform.translation['y'][0], volume_placement.transform.translation['y'][1])
    translation_z = box_center[2] + random.uniform(volume_placement.transform.translation['z'][0], volume_placement.transform.translation['z'][1])
    
    # Random rotation
    rotation_x = random.uniform(volume_placement.transform.rotation['x'][0], volume_placement.transform.rotation['x'][1])
    rotation_y = random.uniform(volume_placement.transform.rotation['y'][0], volume_placement.transform.rotation['y'][1])
    rotation_z = random.uniform(volume_placement.transform.rotation['z'][0], volume_placement.transform.rotation['z'][1])
    
    transform = TransformConfig(
        translation_x=translation_x,
        translation_y=translation_y,
        translation_z=translation_z,
        rotation_x=rotation_x,
        rotation_y=rotation_y,
        rotation_z=rotation_z,
        scale_x=scale_x,
        scale_y=scale_y,
        scale_z=scale_z,
        normalize=False  # Volumes don't need normalization
    )
    
    # Generate occupancy mesh for debugging
    occupancy_mesh = create_volume_occupancy_mesh(
        vdb_path=vdb_path,
        volume_transform={
            'translation_x': translation_x,
            'translation_y': translation_y,
            'translation_z': translation_z,
            'rotation_x': rotation_x,
            'rotation_y': rotation_y,
            'rotation_z': rotation_z,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'scale_z': scale_z
        },
        threshold=0.5,
        transpose_written=transpose
    )
    
    # Save occupancy mesh to metadata folder
    if occupancy_mesh is not None:
        occ_mesh_path = Path(cache_dir) / f"volume_{volume_seed}_occupancy.obj"
        occupancy_mesh.export(str(occ_mesh_path))
        print(f"Saved occupancy mesh to {occ_mesh_path}")
    
    return VolumeConfig(
        vdb_path=vdb_path,
        transform=transform,
        scattering_scale=[float(x) for x in scattering_scale],  # RGB values
        absorption_scale=[float(x) for x in absorption_scale],  # RGB values
        seed=volume_seed,
        transpose_written=transpose
    )

def generate_scene_from_template(template: SceneTemplate, output_dir: str, object_list: List[str], texture_list: List[str], scene_config_path: str, mesh_path: str, num_cam: int, old_lighting_mode: bool = False, texture_crop_res: int = 192, texture_target_res: int = 128, env_map_list: List[str] = None) -> None:
    """Generate scene configuration and meshes from template"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate scene configuration
    scene_config = GeneratedConfig(
        scene_name=template.scene_name,
        version=template.version,
        objects={},
        cameras=[],  # Changed to a list of cameras
        lighting=[],
        env_map=None,
        volumes=None  # Will be initialized as empty list if volumes are generated
    )

    # Process background objects
    for i, obj in enumerate(template.background_scene):
        obj_key = f'background_{i}'
        scene_config.objects[obj_key] = generate_background_object_config(obj, template.material, texture_list)
    # Load and process random objects
    num_objects = random.randint(
        template.object_placement.objects.count['min'],
        template.object_placement.objects.count['max']
    )
    
    # Load bounding mesh
    bounding_mesh: trimesh.Trimesh = trimesh.load(template.object_placement.bounding_mesh)  # type: ignore
    
    # Place random objects
    base_path = template.object_placement.objects.base_path
    for i in range(num_objects):
        # Select random object from provided list
        obj_path = str(Path(base_path) / random.choice(object_list))
        
        # Place object
        transform = place_object_in_bounds(
            bounding_mesh,
            template.object_placement.object_constraints,
            template.object_placement.objects.rotation
        )
        
        obj_key = f'random_object_{i}'
        material_config = sample_material(template.material, texture_list)
        scene_config.objects[obj_key] = ObjectConfig(
            mesh_path=obj_path,
            transform=transform,
            is_background=False,
            is_lighting=False,
            material=material_config
        )

    # Generate multiple camera configurations
    for i in range(num_cam):
        camera_config = generate_camera_config(template.camera)
        scene_config.cameras.append(camera_config)

    # Generate lighting configuration - use truncated normal distribution for num_lights
    # mean_lights = (template.lighting.min_num + template.lighting.max_num) / 2
    # std_lights = (template.lighting.max_num - template.lighting.min_num) / 4  # 2 sigma covers 95% of range
    # num_lights = int(round(np.clip(
    #     np.random.normal(mean_lights, std_lights),
    #     template.lighting.min_num,
    #     template.lighting.max_num
    # )))
    num_lights = random.randint(template.lighting.min_num, template.lighting.max_num)
    lighting_config = generate_lighting_config(template.lighting, num_lights, old_lighting_mode)
    
    # Add lighting objects to scene config
    for i, light in enumerate(lighting_config):
        obj_key = f'light_{i}'
        scene_config.objects[obj_key] = light

    # Generate environment map configuration if template has env_map
    if template.env_map is not None:
        if env_map_list is None:
            env_map_list = []
        if len(env_map_list) > 0:
            scene_config.env_map = sample_env_map(template.env_map, env_map_list)

    # Generate volumes if template has volume configuration
    if template.volume is not None:
        # Check probability
        if random.random() < template.volume.probability:
            # Determine number of volumes
            num_volumes = random.randint(
                template.volume.count['min'],
                template.volume.count['max']
            )
            
            # Initialize volumes list
            scene_config.volumes = []
            
            # Get cache directory from mesh_path (same directory as scene_config.json and scene.obj)
            cache_dir = str(Path(mesh_path).parent)
            
            # Generate each volume
            for i in range(num_volumes):
                volume_config = generate_volume_config(
                    template.volume,
                    bounding_mesh,
                    cache_dir,  # Use same directory as other metadata files
                    volume_idx=i
                )
                scene_config.volumes.append(volume_config)
        else:
            # No volumes generated
            scene_config.volumes = None

    # Save scene configuration
    config_path = scene_config_path
    with open(config_path, 'w') as f:
        json.dump(asdict(scene_config), f, indent=2)
    print(f"Scene configuration saved to {config_path}")
    print(scene_config)

    # Generate and save combined mesh
    generate_scene_mesh(scene_config, str(mesh_path), texture_crop_res=texture_crop_res, texture_target_res=texture_target_res)

def rotate_normal_map(normal_map, angle_deg, axis='z'):
    angle_rad = angle_deg * (torch.pi / 180.0)

    normal_map  = normal_map * 2.0 - 1.0 # Convert to [-1, 1]
    normal_map = normal_map.unsqueeze(0) # Add batch dimension

    # Rotate the Vectors
    rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad), 0],
                                    [math.sin(angle_rad), math.cos(angle_rad), 0],
                                    [0, 0, 1]], device=normal_map.device)

    # Reshape for batch matrix multiplication
    reshaped_normal_map = normal_map.view(1, 3, -1)  # Reshape to [1, 3, H*W]
    rotation_matrix = rotation_matrix.view(1, 3, 3)  # Add batch dimension

    # Rotate the vectors
    rotated_vectors = torch.bmm(rotation_matrix, reshaped_normal_map)
    rotated_vectors = rotated_vectors.view(1, 3, normal_map.size(2), normal_map.size(3))

    rotated_vectors = rotated_vectors / 2.0 + 0.5 # Convert back to [0, 1]

    return rotated_vectors[0]

def _copy_material_no_crop(
        orig_path: str,
        crop_path: str,
        target_res: int,
        material_type: str = None,
        use_heightmap: bool = False
    ) -> None:
    """
    Copy and resize texture maps without random crop/rotation augmentation.
    Used for Blender animation export where textures are already final.
    """
    from renderformer.data.blender.material_types import get_brdf_type

    brdf_type = get_brdf_type(material_type) if material_type else "diffuse_specular"
    os.makedirs(crop_path, exist_ok=True)

    # Determine which maps to copy based on material type
    if brdf_type == "diffuse_specular":
        map_names = ['diffuse.png', 'specular.png']
    elif brdf_type == "metallic_roughness":
        if os.path.exists(os.path.join(orig_path, 'basecolor.png')):
            map_names = ['basecolor.png']
        else:
            map_names = ['base_color.png']
        map_names.append('metallic.png')
    else:
        raise ValueError(f"Unsupported BRDF type for texture copy: {brdf_type}")

    map_names += ['roughness.png', 'normal.png']
    if use_heightmap:
        map_names.append('height.png')

    for name in map_names:
        src = os.path.join(orig_path, name)
        dst = os.path.join(crop_path, name)
        if name == 'normal.png':
            # normal.png needs BGR→RGB→BGR handling to match existing pipeline
            img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            if h != target_res or w != target_res:
                img = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        elif name == 'height.png':
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {src}, skipping heightmap")
                continue
            h, w = img.shape[:2]
            if h != target_res or w != target_res:
                img = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, img)
        else:
            img = cv2.imread(src)
            if img is None:
                print(f"Warning: Could not read {src}, skipping")
                continue
            h, w = img.shape[:2]
            if h != target_res or w != target_res:
                img = cv2.resize(img, (target_res, target_res), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, img)


def random_crop_material(
        orig_path: str,
        crop_path: str,
        crop_seed: int = 0,
        crop_res: int = 192,
        target_res: int = 128,
        material_type: str = None,
        use_heightmap: bool = False
    ) -> None:
    """
    Random crop and augment texture maps.

    Args:
        orig_path: Path to original texture folder
        crop_path: Path to save cropped textures
        crop_seed: Random seed for augmentation. If None, bypass augmentation
            and just copy/resize the original textures (no-crop mode).
        crop_res: Crop resolution
        target_res: Target resolution
        material_type: Material type to determine which maps to crop
            - If None or "diffuse_specular": crop diffuse, specular, roughness, normal
            - If "metallic_roughness": crop basecolor, metallic, roughness, normal
        use_heightmap: Whether to process heightmap (height.png) for displacement mapping
    """
    # No-crop bypass: just copy and resize without augmentation
    if crop_seed is None:
        _copy_material_no_crop(orig_path, crop_path, target_res, material_type, use_heightmap)
        return

    from renderformer.data.blender.material_types import get_brdf_type
    
    # Determine which maps to load based on material_type
    if material_type is not None:
        brdf_type = get_brdf_type(material_type)
    else:
        brdf_type = "diffuse_specular"  # Default
    
    if brdf_type == "diffuse_specular":
        # Load diffuse-specular maps
        diffuse_map = cv2.imread(orig_path + '/diffuse.png')
        specular_map = cv2.imread(orig_path + '/specular.png')
        basecolor_map = None
        metallic_map = None
    elif brdf_type == "metallic_roughness":
        # Load metallic-roughness maps
        # Try basecolor.png first, fallback to base_color.png
        basecolor_path = orig_path + '/basecolor.png'
        base_color_path = orig_path + '/base_color.png'
        if os.path.exists(basecolor_path):
            basecolor_map = cv2.imread(basecolor_path)
        elif os.path.exists(base_color_path):
            basecolor_map = cv2.imread(base_color_path)
        else:
            raise FileNotFoundError(f"Neither basecolor.png nor base_color.png found in {orig_path}/")
        metallic_map = cv2.imread(orig_path + '/metallic.png')
        diffuse_map = None
        specular_map = None
    else:
        raise ValueError(f"Unsupported BRDF type for texture cropping: {brdf_type}")
    
    # Load common maps
    roughness_map = cv2.imread(orig_path + '/roughness.png')
    normal_map = cv2.cvtColor(cv2.imread(orig_path + '/normal.png'), cv2.COLOR_BGR2RGB)
    
    # Load heightmap if needed
    height_map = None
    if use_heightmap:
        height_path = orig_path + '/height.png'
        if os.path.exists(height_path):
            # Heightmap is typically grayscale, read as single channel
            height_map = cv2.imread(height_path, cv2.IMREAD_GRAYSCALE)
            if height_map is None:
                print(f"Warning: Failed to load heightmap from {height_path}, skipping heightmap processing")
                use_heightmap = False
        else:
            print(f"Warning: Heightmap file not found at {height_path}, skipping heightmap processing")
            use_heightmap = False

    random.seed(crop_seed)

    torch.set_num_threads(4)

    # random rotate the maps
    rot_angle = random.random() * 360

    # device = 'cuda'
    device = 'cpu'
    
    # Convert maps to tensors
    if brdf_type == "diffuse_specular":
        diffuse_img = TF.to_tensor(diffuse_map).to(device)
        specular_img = TF.to_tensor(specular_map).to(device)
        basecolor_img = None
        metallic_img = None
    else:  # metallic_roughness
        basecolor_img = TF.to_tensor(basecolor_map).to(device)
        metallic_img = TF.to_tensor(metallic_map).to(device)
        diffuse_img = None
        specular_img = None
    
    roughness_img = TF.to_tensor(roughness_map).to(device)
    normal_img = TF.to_tensor(normal_map).to(device)
    
    # Convert heightmap to tensor if needed
    height_img = None
    if use_heightmap and height_map is not None:
        # Convert grayscale to 3-channel for consistency with other maps
        height_map_3ch = cv2.cvtColor(height_map, cv2.COLOR_GRAY2BGR)
        height_img = TF.to_tensor(height_map_3ch).to(device)

    # Get reference dimensions (all maps should have same size)
    ref_map = diffuse_map if brdf_type == "diffuse_specular" else basecolor_map
    h, w = ref_map.shape[:2]

    # First crop a small map out of original
    first_crop_res = crop_res * 4
    first_crop_h = random.randint(0, h - first_crop_res)
    first_crop_w = random.randint(0, w - first_crop_res)
    
    if brdf_type == "diffuse_specular":
        diffuse_img = TF.crop(diffuse_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
        specular_img = TF.crop(specular_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
    else:  # metallic_roughness
        basecolor_img = TF.crop(basecolor_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
        metallic_img = TF.crop(metallic_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
    
    roughness_img = TF.crop(roughness_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
    normal_img = TF.crop(normal_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)
    
    if use_heightmap and height_img is not None:
        height_img = TF.crop(height_img, first_crop_h, first_crop_w, first_crop_res, first_crop_res)

    # Downsample to half
    middle_res = first_crop_res // 2
    if brdf_type == "diffuse_specular":
        diffuse_img = TF.resize(diffuse_img, (middle_res, middle_res), antialias=True)
        specular_img = TF.resize(specular_img, (middle_res, middle_res), antialias=True)
    else:  # metallic_roughness
        basecolor_img = TF.resize(basecolor_img, (middle_res, middle_res), antialias=True)
        metallic_img = TF.resize(metallic_img, (middle_res, middle_res), antialias=True)
    
    roughness_img = TF.resize(roughness_img, (middle_res, middle_res), antialias=True)
    normal_img = TF.resize(normal_img, (middle_res, middle_res), antialias=True)
    
    if use_heightmap and height_img is not None:
        height_img = TF.resize(height_img, (middle_res, middle_res), antialias=True)

    # Rotate the maps
    rot_res = math.floor(middle_res * math.sqrt(2))
    if brdf_type == "diffuse_specular":
        diffuse_img = diffuse_img.repeat(1, 3, 3)
        diffuse_img = TF.center_crop(diffuse_img, (rot_res, rot_res))
        specular_img = specular_img.repeat(1, 3, 3)
        specular_img = TF.center_crop(specular_img, (rot_res, rot_res))
    else:  # metallic_roughness
        basecolor_img = basecolor_img.repeat(1, 3, 3)
        basecolor_img = TF.center_crop(basecolor_img, (rot_res, rot_res))
        metallic_img = metallic_img.repeat(1, 3, 3)
        metallic_img = TF.center_crop(metallic_img, (rot_res, rot_res))
    
    roughness_img = roughness_img.repeat(1, 3, 3)
    roughness_img = TF.center_crop(roughness_img, (rot_res, rot_res))
    normal_img = normal_img.repeat(1, 3, 3)
    normal_img = TF.center_crop(normal_img, (rot_res, rot_res))
    
    if use_heightmap and height_img is not None:
        height_img = height_img.repeat(1, 3, 3)
        height_img = TF.center_crop(height_img, (rot_res, rot_res))

    # Apply rotation
    if brdf_type == "diffuse_specular":
        diffuse_img = TF.rotate(diffuse_img, rot_angle, TF.InterpolationMode.BILINEAR)
        diffuse_img = TF.center_crop(diffuse_img, (middle_res, middle_res))
        diffuse_map = (diffuse_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)

        specular_img = TF.rotate(specular_img, rot_angle, TF.InterpolationMode.BILINEAR)
        specular_img = TF.center_crop(specular_img, (middle_res, middle_res))
        specular_map = (specular_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    else:  # metallic_roughness
        basecolor_img = TF.rotate(basecolor_img, rot_angle, TF.InterpolationMode.BILINEAR)
        basecolor_img = TF.center_crop(basecolor_img, (middle_res, middle_res))
        basecolor_map = (basecolor_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)

        metallic_img = TF.rotate(metallic_img, rot_angle, TF.InterpolationMode.BILINEAR)
        metallic_img = TF.center_crop(metallic_img, (middle_res, middle_res))
        metallic_map = (metallic_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)

    roughness_img = TF.rotate(roughness_img, rot_angle, TF.InterpolationMode.BILINEAR)
    roughness_img = TF.center_crop(roughness_img, (middle_res, middle_res))
    roughness_map = (roughness_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)

    normal_img = rotate_normal_map(normal_img, axis='z', angle_deg=rot_angle)
    normal_img = TF.rotate(normal_img, rot_angle, TF.InterpolationMode.BILINEAR)
    normal_img = TF.center_crop(normal_img, (middle_res, middle_res))
    normal_map = (normal_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    if use_heightmap and height_img is not None:
        # Heightmap doesn't need special rotation like normal map, just regular rotation
        height_img = TF.rotate(height_img, rot_angle, TF.InterpolationMode.BILINEAR)
        height_img = TF.center_crop(height_img, (middle_res, middle_res))
        height_map = (height_img * 255.).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)

    # Final small crop in the rotated maps
    ref_final_map = diffuse_map if brdf_type == "diffuse_specular" else basecolor_map
    h, w, _ = ref_final_map.shape
    crop_res_final = random.randint(target_res, crop_res)
    print(crop_res_final)
    crop_h = random.randint(0, h - crop_res_final)
    crop_w = random.randint(0, w - crop_res_final)
    
    if brdf_type == "diffuse_specular":
        diffuse_map = cv2.resize(diffuse_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
        specular_map = cv2.resize(specular_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
    else:  # metallic_roughness
        basecolor_map = cv2.resize(basecolor_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
        metallic_map = cv2.resize(metallic_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
    
    roughness_map = cv2.resize(roughness_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
    normal_map = cv2.resize(normal_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)
    
    if use_heightmap and height_map is not None:
        height_map = cv2.resize(height_map[crop_h:crop_h + crop_res_final, crop_w:crop_w + crop_res_final], (target_res, target_res), interpolation=cv2.INTER_AREA)

    # Save the maps
    if brdf_type == "diffuse_specular":
        cv2.imwrite(crop_path + '/diffuse.png', diffuse_map)
        cv2.imwrite(crop_path + '/specular.png', specular_map)
    else:  # metallic_roughness
        cv2.imwrite(crop_path + '/basecolor.png', basecolor_map)
        cv2.imwrite(crop_path + '/metallic.png', metallic_map)
    
    cv2.imwrite(crop_path + '/roughness.png', roughness_map)
    cv2.imwrite(crop_path + '/normal.png', cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))
    
    # Save heightmap if processed
    if use_heightmap and height_map is not None:
        # Convert back to grayscale for saving (heightmap is typically single channel)
        if len(height_map.shape) == 3:
            height_map_gray = cv2.cvtColor(height_map, cv2.COLOR_BGR2GRAY)
        else:
            height_map_gray = height_map
        cv2.imwrite(crop_path + '/height.png', height_map_gray)
    # normal_map = np.zeros_like(diffuse_map)
    # normal_map[:, :, 0] = 128
    # normal_map[:, :, 1] = 128
    # normal_map[:, :, 2] = 255
    # cv2.imwrite(crop_path + '/normal.png', cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR))

def generate_scene_mesh(scene_config: GeneratedConfig, output_path: str, texture_crop_res: int = 192, texture_target_res: int = 128, force_heightmap: bool = False, skip_transform: bool = False) -> None:
    """
    Generate combined mesh from scene configuration using trimesh.Scene

    Args:
        scene_config: Scene configuration
        output_path: Path to save the combined mesh
        texture_crop_res: Resolution to crop textures
        texture_target_res: Target resolution for textures
        force_heightmap: If True, override use_heightmap in material config to True (if supported)
        skip_transform: If True, skip normalize/rotation/scale/translation transforms.
            Used for Blender animation export where meshes are already in world space.
    """
    scene = trimesh.Scene()
    split_mesh_folder_path = os.path.dirname(output_path) + '/split'
    os.makedirs(split_mesh_folder_path, exist_ok=True)

    for obj_key, obj_config in scene_config.objects.items():
        if skip_transform:
            # When skip_transform=True, the OBJ files are already in split/ from the
            # Blender plugin export with correct vertex normals and UV data.
            # Do NOT load/process/re-export the mesh — trimesh's load→export cycle
            # can destroy vertex normals and texture/UV data.
            # Only process textures (no-crop copy + BRDF latent encoding) below.
            print(f'skip_transform: skipping mesh load/export for {obj_key}')
        else:
            mesh: trimesh.Trimesh = trimesh.load(obj_config.mesh_path, process=False)  # type: ignore

            if not obj_config.is_background and not obj_config.is_lighting and obj_config.transform.normalize:
                mesh = normalize_to_unit_sphere(mesh)

            # Apply transformations
            transform = obj_config.transform

            # first apply rotation, then scale, then translation
            for axis in ['x', 'y', 'z']:
                axis_array = [1, 0, 0] if axis == 'x' else [0, 1, 0] if axis == 'y' else [0, 0, 1]
                axis_array = np.array(axis_array).astype(float)
                angle = getattr(transform, f'rotation_{axis}')
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.deg2rad(angle), axis_array
                )
                mesh.apply_transform(rotation_matrix)

            scale = np.array([getattr(transform, f'scale_{axis}') for axis in ['x', 'y', 'z']])  # direct scale half
            if not obj_config.is_lighting:
                scale = scale / 2.
            mesh.apply_scale(scale)
            translation = np.array([getattr(transform, f'translation_{axis}') for axis in ['x', 'y', 'z']])  # direct translation half
            if not obj_config.is_lighting:
                translation = translation / 2.
            mesh.apply_translation(translation)

            if obj_config.material.smooth_shading:
                mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(30))
            elif obj_config.material.texture_map_path is not None:
                # have to handle the uv mapping when converting to flat shading mesh
                uv = mesh.visual.uv[mesh.faces].reshape(-1, 2)
                assert uv.shape[0] == mesh.faces.shape[0] * 3, "UV shape does not match faces"
                mesh = trimesh.Trimesh(
                    vertices=mesh.triangles.reshape(-1, 3),
                    faces=np.arange(len(mesh.triangles) * 3).reshape(-1, 3),
                    process=False
                )
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv,
                    image=None
                )
            else:  # ordinary vertex color
                mesh = trimesh.Trimesh(
                    vertices=mesh.triangles.reshape(-1, 3),
                    faces=np.arange(len(mesh.triangles) * 3).reshape(-1, 3),
                    process=False
                )

        if obj_config.material.texture_map_path is not None:
            # crop the texture maps
            crop_path = f"{split_mesh_folder_path}/{obj_key}"
            os.makedirs(crop_path, exist_ok=True)
            # Determine effective use_heightmap flag
            # If force_heightmap is True, use it. Otherwise use config value.
            effective_use_heightmap = obj_config.material.use_heightmap
            if force_heightmap:
                effective_use_heightmap = True
                
            random_crop_material(
                orig_path=obj_config.material.texture_map_path,
                crop_path=crop_path,
                crop_seed=obj_config.material.texture_map_aug_random_seed,
                crop_res=texture_crop_res,
                target_res=texture_target_res,
                material_type=obj_config.material.material_type,
                use_heightmap=effective_use_heightmap
            )
            
            # Extract BRDF parameters and map to latent space
            from renderformer.data.blender.brdf_utils import (
                extract_brdf_params_from_texture,
                map_brdf_to_latent,
                save_latent_map,
            )
            from renderformer.data.blender.material_types import supports_svbrdf
            
            if supports_svbrdf(obj_config.material.material_type):
                # Extract BRDF parameters from augmented textures
                brdf_params = extract_brdf_params_from_texture(
                    crop_path,
                    obj_config.material.material_type
                )
                
                # Map to latent space
                latent_map = map_brdf_to_latent(
                    brdf_params,
                    obj_config.material.material_type,
                )
                
                # Save as 3 PNG images
                save_latent_map(latent_map, crop_path)
                print(f'Generated and saved latent map for {obj_key}')
            
            if not skip_transform:
                print('object uv:', mesh.visual.uv.shape)
        elif skip_transform:
            # skip_transform: no mesh loaded, skip all mesh-based material processing
            pass
        elif obj_config.material.rand_tri_diffuse_seed is not None:
            from renderformer.data.blender.material_types import (
                MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
                MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
                MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
            )
            # to make the random color consistent
            random.seed(obj_config.material.rand_tri_diffuse_seed)
            np.random.seed(obj_config.material.rand_tri_diffuse_seed)

            # apply diffuse properties
            mesh_split = []
            kwarg: Dict = {'only_watertight': False}
            if obj_config.is_background:
                kwarg['adjacency'] = np.array([])
            # Get max channel value from params to bound random base color
            if obj_config.material.material_type == MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR:
                params = obj_config.material.diffuse_specular_params
                base_color = params.get('diffuse', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            elif obj_config.material.material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS:
                params = obj_config.material.metallic_roughness_params
                base_color = params.get('base_color', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            elif obj_config.material.material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION:
                params = obj_config.material.metallic_roughness_transmission_params
                base_color = params.get('base_color', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            else:
                base_color = [0.5, 0.5, 0.5]
            color_max = max(base_color) if isinstance(base_color, list) else 1.0
            for small_mesh in mesh.split(**kwarg):
                shared_color = np.random.randint(0, math.ceil(256 * color_max), (1, 3)).repeat(small_mesh.faces.shape[0], axis=0)
                # print('shared_color:', shared_color.shape, shared_color.dtype)
                new_small_mesh = trimesh.Trimesh(
                    vertices=small_mesh.vertices,
                    faces=small_mesh.faces,
                    vertex_normals=small_mesh.vertex_normals,
                    face_colors=shared_color,
                    process=False
                )
                mesh_split.append(new_small_mesh)
            mesh = trimesh.util.concatenate(mesh_split)
        else:
            from renderformer.data.blender.material_types import (
                MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
                MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
                MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
            )
            # Get base color from params
            if obj_config.material.material_type == MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR:
                params = obj_config.material.diffuse_specular_params
                base_color = params.get('diffuse', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            elif obj_config.material.material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS:
                params = obj_config.material.metallic_roughness_params
                base_color = params.get('base_color', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            elif obj_config.material.material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION:
                params = obj_config.material.metallic_roughness_transmission_params
                base_color = params.get('base_color', [0.5, 0.5, 0.5]) if params else [0.5, 0.5, 0.5]
            else:
                base_color = [0.5, 0.5, 0.5]
            vertex_colors = (np.array(base_color) * 255.).clip(0, 255).astype(int)
            vertex_colors = np.tile(vertex_colors, mesh.vertices.shape[0]).reshape(-1, 3)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=vertex_colors,
            )

        if not skip_transform:
            print('object vn:', mesh.vertex_normals.shape)  # must have this line to trigger the calculation of vertex normals

            # Add mesh to scene with key and material
            scene.add_geometry(mesh, geom_name=obj_key)

            # Save individual meshes
            mesh.export(f"{split_mesh_folder_path}/{obj_key}.obj", include_normals=True, include_texture=True)
    
    # Save combined mesh  # no longer used
    # scene.export(output_path, include_normals=True, include_texture=True)
