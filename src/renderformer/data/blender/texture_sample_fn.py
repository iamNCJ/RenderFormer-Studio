import numpy as np
import torch
from torch.nn.functional import grid_sample
import trimesh
import imageio


def calculate_barycentric_coordinates(p):
    # p: (N, 2)
    # p0, p1, p2: (N, 2)
    p0 = torch.tensor([[1, 0]], dtype=torch.float32)
    p1 = torch.tensor([[0, 1]], dtype=torch.float32)
    p2 = torch.tensor([[0, 0]], dtype=torch.float32)
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return torch.stack([u, v, w], dim=-1)


def texture_sample(
    mesh: trimesh.Trimesh, 
    texture_folder: str, 
    size: int = 64,
    material_type: str = None,
    use_latent: bool = True,
    emissive: list = None,
    use_heightmap: bool = False
):
    """
    Sample texture from texture folder and map to barycentric coordinates.
    
    Args:
        mesh: Trimesh object
        texture_folder: Path to texture folder
        size: Texture size (default: 64)
        material_type: Material type string (required if use_latent=True)
        use_latent: Whether to use latent maps (default: True)
        emissive: Emissive color [r, g, b], non-negative, no upper bound (for area lights)
        use_heightmap: Whether to load and process heightmap for displacement mapping
    
    Returns:
        Barycentric texture array of shape (num_face, 15 or 16, size, size)
        - 15 channels: 9 latent + 3 normal + 3 emissive (when use_heightmap=False)
        - 16 channels: 9 latent + 3 normal + 3 emissive + 1 heightmap (when use_heightmap=True)
    """
    if emissive is None:
        emissive = [0.0, 0.0, 0.0]
    empty_triangle_mask = torch.ones((1, size, size), dtype=torch.bool)
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    _, x, y = torch.where(empty_triangle_mask)

    # Generate barycentric coordinates
    p = (torch.stack([x, y], dim=-1).float() + 0.5) / size
    bary_coord = calculate_barycentric_coordinates(p)

    if use_latent:
        # Load latent map from 3 PNG images
        from renderformer.data.blender.brdf_utils import load_latent_map
        
        if material_type is None:
            raise ValueError("material_type is required when use_latent=True")
        
        try:
            latent_map = load_latent_map(texture_folder)  # (H, W, 9)
        except Exception as e:
            raise ValueError(f"Failed to load latent map from {texture_folder}: {e}")
        
        normal_01 = imageio.v3.imread(f'{texture_folder}/normal.png')[..., :3] / 255.
        # Convert normal from [0, 1] to [-1, 1] to match latent range
        normal = normal_01 * 2. - 1.
        
        # Load and process heightmap if needed
        heightmap = None
        if use_heightmap:
            try:
                # Heightmap is typically grayscale, read as single channel
                heightmap_01 = imageio.v3.imread(f'{texture_folder}/height.png')
                if len(heightmap_01.shape) == 3:
                    # If RGB, take first channel or convert to grayscale
                    heightmap_01 = heightmap_01[..., 0] / 255.
                else:
                    heightmap_01 = heightmap_01 / 255.
                # Convert heightmap from [0, 1] to [-1, 1] to match normal map range
                heightmap = heightmap_01 * 2. - 1.
                # Expand to (H, W, 1) for concatenation
                if heightmap.ndim == 2:
                    heightmap = heightmap[..., None]  # (H, W, 1)
            except Exception as e:
                print(f"Warning: Failed to load heightmap from {texture_folder}/height.png: {e}, skipping heightmap")
                use_heightmap = False
        
        # Emissive should be non-negative, no upper bound (for area lights)
        # Keep original values, do not convert to [-1, 1]
        emissive_map = np.array(emissive, dtype=np.float32)  # (3,)
        # Create uniform map with same shape as latent_map
        H, W = latent_map.shape[:2]
        emissive_map = np.broadcast_to(emissive_map[None, None, :], (H, W, 3))  # (H, W, 3)
        
        # Assemble texture: [latent(9), normal(3), emissive(3)] = 15 channels
        # Or with heightmap: [latent(9), normal(3), emissive(3), heightmap(1)] = 16 channels
        if use_heightmap and heightmap is not None:
            texture = np.concatenate([latent_map, normal, emissive_map, heightmap], axis=-1)
        else:
            texture = np.concatenate([latent_map, normal, emissive_map], axis=-1)
    else:
        # Legacy mode: load from original BRDF textures
        diffuse = imageio.v3.imread(f'{texture_folder}/diffuse.png')[..., :3] / 255.
        specular = imageio.v3.imread(f'{texture_folder}/specular.png')[..., :3] / 255.
        roughness = imageio.v3.imread(f'{texture_folder}/roughness.png')[..., :1] / 255.
        normal_01 = imageio.v3.imread(f'{texture_folder}/normal.png')[..., :3] / 255.
        # Convert normal from [0, 1] to [-1, 1] for consistency
        normal = normal_01 * 2. - 1.
        
        # Load and process heightmap if needed
        heightmap = None
        if use_heightmap:
            try:
                # Heightmap is typically grayscale, read as single channel
                heightmap_01 = imageio.v3.imread(f'{texture_folder}/height.png')
                if len(heightmap_01.shape) == 3:
                    # If RGB, take first channel or convert to grayscale
                    heightmap_01 = heightmap_01[..., 0] / 255.
                else:
                    heightmap_01 = heightmap_01 / 255.
                # Convert heightmap from [0, 1] to [-1, 1] to match normal map range
                heightmap = heightmap_01 * 2. - 1.
                # Expand to (H, W, 1) for concatenation
                if heightmap.ndim == 2:
                    heightmap = heightmap[..., None]  # (H, W, 1)
            except Exception as e:
                print(f"Warning: Failed to load heightmap from {texture_folder}/height.png: {e}, skipping heightmap")
                use_heightmap = False
        
        # Emissive should be non-negative, no upper bound (for area lights)
        # Keep original values, do not convert to [-1, 1]
        emissive_map = np.array(emissive, dtype=np.float32)  # (3,)
        H, W = diffuse.shape[:2]
        emissive_map = np.broadcast_to(emissive_map[None, None, :], (H, W, 3))  # (H, W, 3)
        
        # For backward compatibility, pad to 16 channels (13 original + 3 emissive)
        # Or with heightmap: 17 channels (13 original + 3 emissive + 1 heightmap)
        if use_heightmap and heightmap is not None:
            texture = np.concatenate([diffuse, specular, roughness, normal, np.zeros_like(diffuse), emissive_map, heightmap], axis=-1)
        else:
            texture = np.concatenate([diffuse, specular, roughness, normal, np.zeros_like(diffuse), emissive_map], axis=-1)
    # print(normal.max(), normal.min())
    texture = torch.tensor(texture, dtype=torch.float32).permute(2, 0, 1)[None]

    num_face = len(mesh.faces)
    num_channels = texture.shape[1]  # 15 for latent mode, 16 for legacy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    barycentric_texture = torch.zeros((num_face, num_channels, size, size))
    face_vert_ids = mesh.faces
    
    with torch.no_grad():
        # Check if mesh has UV coordinates
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = torch.from_numpy(mesh.visual.uv[face_vert_ids].astype(np.float32))
        else:
            raise ValueError(f"Mesh does not have UV coordinates for texture sampling")
        bary_uv = bary_coord.to(device) @ uvs.to(device)
        grid_coord = bary_uv[None] * 2. - 1.
        grid_coord[..., 1] *= -1
        sampled_color = grid_sample(
            texture.to(device),
            grid_coord,
            align_corners=False,
            padding_mode='border',
        )
        barycentric_texture[:, :, y, x] = sampled_color[0].permute(1, 0, 2).cpu()

    print(f'Barycentric texture shape: {barycentric_texture.shape}')
    barycentric_texture_np = barycentric_texture.numpy()
    return barycentric_texture_np
