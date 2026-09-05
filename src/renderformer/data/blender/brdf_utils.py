"""
BRDF utility functions for parameter extraction, latent mapping, and file I/O.
"""

import numpy as np
import torch
import torch.nn as nn
import imageio
import os
from typing import Optional, Dict
from renderformer.data.blender.material_types import (
    get_brdf_type, get_mapper_checkpoint, get_brdf_param_dim,
    MATERIAL_TYPE_SVBRDF_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR,
    MATERIAL_TYPE_SVBRDF_METALLIC_ROUGHNESS,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS,
    MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION,
)
from renderformer.models.material import (
    DiffuseSpecularToLatent,
    PrincipledBRDFToLatent,
    PrincipledBRDFToLatentWithTransmission,
)

# Global mapper cache to avoid repeated loading. The full resolved identity is
# part of the key so environment overrides cannot accidentally reuse a mapper
# loaded from a different source.
_mapper_cache: Dict[tuple[str, str | None, type[nn.Module], str], nn.Module] = {}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_mapper(
    material_type: str,
    device: str | torch.device | None = None,
) -> nn.Module:
    """
    Load BRDF mapper from huggingface path.
    Uses caching to avoid repeated loading.
    
    Args:
        material_type: Material type string
        device: Device to load mapper on. Defaults to CUDA when available,
            otherwise CPU.
    
    Returns:
        Loaded mapper module
    """
    resolved_device = _resolve_device(device)

    # Get mapper path and class
    mapper_path, mapper_subfolder = get_mapper_checkpoint(material_type)
    brdf_type = get_brdf_type(material_type)
    
    # Map BRDF type to mapper class
    if brdf_type == "diffuse_specular":
        mapper_class = DiffuseSpecularToLatent
    elif brdf_type == "metallic_roughness":
        mapper_class = PrincipledBRDFToLatent
    elif brdf_type == "metallic_roughness_transmission":
        mapper_class = PrincipledBRDFToLatentWithTransmission
    else:
        raise ValueError(f"Unknown BRDF type: {brdf_type}")

    cache_key = (
        mapper_path,
        mapper_subfolder,
        mapper_class,
        str(resolved_device),
    )
    if cache_key in _mapper_cache:
        return _mapper_cache[cache_key]
    
    mapper = mapper_class.from_pretrained(
        mapper_path,
        subfolder=mapper_subfolder,
        strict=True,
    )
    mapper = mapper.to(resolved_device)
    mapper.eval()
    
    # Cache it
    _mapper_cache[cache_key] = mapper
    
    return mapper


def extract_brdf_params_from_texture(texture_folder: str, material_type: str) -> np.ndarray:
    """
    Extract BRDF parameters from texture files (SVBRDF mode).
    
    Args:
        texture_folder: Path to texture folder containing texture maps
        material_type: Material type string
    
    Returns:
        BRDF parameters as numpy array of shape (H, W, param_dim)
    """
    brdf_type = get_brdf_type(material_type)
    
    if brdf_type == "diffuse_specular":
        # Read diffuse, specular, roughness
        diffuse = imageio.v3.imread(f'{texture_folder}/diffuse.png')[..., :3] / 255.
        specular = imageio.v3.imread(f'{texture_folder}/specular.png')[..., :3] / 255.
        roughness = imageio.v3.imread(f'{texture_folder}/roughness.png')[..., :1] / 255.
        # Concatenate: (H, W, 7)
        return np.concatenate([diffuse, specular, roughness], axis=-1)
    
    elif brdf_type == "metallic_roughness":
        # Read base_color, metallic, roughness
        base_color = imageio.v3.imread(f'{texture_folder}/basecolor.png')[..., :3] / 255.
        metallic = imageio.v3.imread(f'{texture_folder}/metallic.png')[..., :1] / 255.
        roughness = imageio.v3.imread(f'{texture_folder}/roughness.png')[..., :1] / 255.
        # Concatenate: (H, W, 5)
        return np.concatenate([base_color, metallic, roughness], axis=-1)
    
    else:
        raise ValueError(f"BRDF type {brdf_type} does not support SVBRDF (texture extraction)")


def extract_brdf_params_from_material_config(material_config) -> np.ndarray:
    """
    Extract BRDF parameters from MaterialConfig (homogeneous mode).
    
    Args:
        material_config: MaterialConfig object
    
    Returns:
        BRDF parameters as numpy array of shape (param_dim,)
    """
    material_type = material_config.material_type
    
    if material_type in [MATERIAL_TYPE_HOMO_DIFFUSE_SPECULAR]:
        params = material_config.diffuse_specular_params
        diffuse = np.array(params['diffuse'])
        specular = np.array(params['specular'])
        roughness = np.array([params['roughness']])
        return np.concatenate([diffuse, specular, roughness])
    
    elif material_type in [MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS]:
        params = material_config.metallic_roughness_params
        base_color = np.array(params['base_color'])
        metallic = np.array([params['metallic']])
        roughness = np.array([params['roughness']])
        return np.concatenate([base_color, metallic, roughness])
    
    elif material_type == MATERIAL_TYPE_HOMO_METALLIC_ROUGHNESS_TRANSMISSION:
        params = material_config.metallic_roughness_transmission_params
        base_color = np.array(params['base_color'])
        metallic = np.array([params['metallic']])
        roughness = np.array([params['roughness']])
        ior = np.array([params['ior']])
        transmission_weight = np.array([params['transmission_weight']])
        return np.concatenate([base_color, metallic, roughness, ior, transmission_weight])
    
    else:
        raise ValueError(f"Material type {material_type} does not support homogeneous extraction")


def map_brdf_to_latent(
    brdf_params: np.ndarray,
    material_type: str,
    mapper: Optional[nn.Module] = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Map BRDF parameters to latent space using mapper.
    
    Args:
        brdf_params: BRDF parameters as numpy array
            - For SVBRDF: shape (H, W, param_dim)
            - For homogeneous: shape (param_dim,) or (N, param_dim) for batch
        material_type: Material type string
        mapper: Optional pre-loaded mapper (if None, will load)
        device: Device to run mapper on. Defaults to CUDA when available,
            otherwise CPU.
    
    Returns:
        Latent vectors as numpy array
            - For SVBRDF: shape (H, W, 9)
            - For homogeneous: shape (9,) or (N, 9) for batch
    """
    resolved_device = _resolve_device(device)
    if mapper is None:
        mapper = load_mapper(material_type, device=resolved_device)
    
    # Convert to tensor
    brdf_tensor = torch.tensor(brdf_params, dtype=torch.float32, device=resolved_device)
    
    # Handle different input shapes
    original_shape = brdf_tensor.shape
    is_3d_image = False
    is_1d = False
    
    if brdf_tensor.ndim == 3:
        # (H, W, param_dim) - This is an image (SVBRDF case)
        H, W, param_dim = original_shape
        brdf_tensor = brdf_tensor.view(-1, param_dim)  # (H*W, param_dim)
        is_3d_image = True
    elif brdf_tensor.ndim == 2:
        # (N, param_dim) - This is batch data (homogeneous with multiple groups)
        # Keep as is, no reshaping needed
        pass
    elif brdf_tensor.ndim == 1:
        # (param_dim,) - Single homogeneous material
        brdf_tensor = brdf_tensor.unsqueeze(0)  # (1, param_dim)
        is_1d = True
    
    # Map to latent
    with torch.no_grad():
        latent_tensor = mapper(brdf_tensor)  # (N, 9)
    
    # Reshape back to original structure
    if is_3d_image:
        # (H*W, 9) -> (H, W, 9)
        H, W = original_shape[:2]
        latent_tensor = latent_tensor.view(H, W, 9)
    elif is_1d:
        # (1, 9) -> (9,)
        latent_tensor = latent_tensor.squeeze(0)
    # For 2D case (N, param_dim), keep as (N, 9) - no reshaping needed
    
    # Convert back to numpy
    return latent_tensor.cpu().numpy()


def save_latent_map(latent_map: np.ndarray, output_dir: str) -> None:
    """
    Save latent map as 3 PNG images (9 channels total).
    
    Args:
        latent_map: Latent map as numpy array of shape (H, W, 9)
        output_dir: Directory to save PNG files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert from [-1, 1] to [0, 1]
    latent_map_01 = (latent_map + 1.0) * 0.5
    latent_map_01 = np.clip(latent_map_01, 0.0, 1.0)
    
    # Split into 3 images
    latent_0 = latent_map_01[..., 0:3]  # channels 0-2
    latent_1 = latent_map_01[..., 3:6]  # channels 3-5
    latent_2 = latent_map_01[..., 6:9]  # channels 6-8
    
    # Save as PNG (range [0, 1] -> [0, 255])
    imageio.v3.imwrite(
        os.path.join(output_dir, 'latent_0.png'),
        (latent_0 * 255).astype(np.uint8)
    )
    imageio.v3.imwrite(
        os.path.join(output_dir, 'latent_1.png'),
        (latent_1 * 255).astype(np.uint8)
    )
    imageio.v3.imwrite(
        os.path.join(output_dir, 'latent_2.png'),
        (latent_2 * 255).astype(np.uint8)
    )


def load_latent_map(latent_dir: str) -> np.ndarray:
    """
    Load latent map from 3 PNG images.
    
    Args:
        latent_dir: Directory containing latent_0.png, latent_1.png, latent_2.png
    
    Returns:
        Latent map as numpy array of shape (H, W, 9) in range [-1, 1]
    """
    # Load 3 images
    latent_0 = imageio.v3.imread(os.path.join(latent_dir, 'latent_0.png'))[..., :3] / 255.
    latent_1 = imageio.v3.imread(os.path.join(latent_dir, 'latent_1.png'))[..., :3] / 255.
    latent_2 = imageio.v3.imread(os.path.join(latent_dir, 'latent_2.png'))[..., :3] / 255.
    
    # Concatenate to 9 channels
    latent_map_01 = np.concatenate([latent_0, latent_1, latent_2], axis=-1)
    
    # Convert from [0, 1] to [-1, 1]
    latent_map = (latent_map_01 - 0.5) * 2.0
    
    return latent_map
