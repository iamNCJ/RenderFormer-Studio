import torch


def compute_triangle_area(vertices):
    """
    Compute triangle areas from 3D vertex positions.
    
    Args:
        vertices: tensor of shape (..., 3, 3) for N triangles, each with 3 vertices in 3D
    
    Returns:
        areas: tensor of shape (N,) for N triangles or a single scalar for one triangle
    """

    # Get vectors for two edges of each triangle
    v1 = vertices[..., 1] - vertices[..., 0]  # First edge vector
    v2 = vertices[..., 2] - vertices[..., 0]  # Second edge vector

    # Compute cross product
    cross = torch.cross(v1, v2, dim=-1)

    # Area = 1/2 * |cross product|
    areas = 0.5 * torch.norm(cross, dim=-1)

    return areas
