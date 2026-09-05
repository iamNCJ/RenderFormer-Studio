"""Lat-long environment-map geometry, sampling, and rotation helpers.

This module intentionally depends only on PyTorch.  It is not imported from
``renderformer.data.__init__`` so importing the package itself stays light.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "latlong_ray_direction",
    "prepare_raw_env_map",
    "rotate_env_map",
    "sample_env_map",
    "view_direction_to_latlong",
]


def prepare_raw_env_map(
    env_map: torch.Tensor,
    rotation_degrees: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply legacy raw-H5 rotation/strength and return RGB ``(C,H,W)``.

    Raw V2 exports store environment maps channel-last together with XYZ Euler
    angles in degrees. Processed files are already channel-first and must not
    pass through this function a second time.
    """

    if env_map.dim() != 3 or env_map.shape[-1] not in (3, 4):
        raise ValueError(
            "raw env_map must have shape (H,W,3) or (H,W,4), "
            f"got {tuple(env_map.shape)}"
        )
    rotation_degrees = torch.as_tensor(
        rotation_degrees,
        dtype=torch.float32,
        device=env_map.device,
    ).reshape(1, 3)
    try:
        import roma
    except ImportError as exc:
        raise RuntimeError("raw environment-map rotation requires roma") from exc

    rotation_matrix = roma.euler_to_rotmat(
        convention="xyz",
        angles=rotation_degrees * (torch.pi / 180.0),
    )
    rgb = env_map[..., :3].to(torch.float32)
    rotated = rotate_env_map(rgb, rotation_matrix)
    return (rotated * float(strength)).permute(2, 0, 1).contiguous()


def latlong_ray_direction(
    h: int = 256,
    w: int = 512,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return the Z-up unit direction at every lat-long pixel center.

    The horizontal coordinate starts at ``+X`` and advances toward ``-Y``;
    the vertical coordinate runs from ``+Z`` at the top to ``-Z`` at the
    bottom.  The returned tensor has shape ``(h, w, 3)``.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"h and w must be positive, got h={h}, w={w}")

    v = (torch.arange(h, device=device) + 0.5) / h
    u = (torch.arange(w, device=device) + 0.5) / w
    grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

    theta = grid_v * torch.pi
    phi = (0.5 - grid_u) * (2.0 * torch.pi)
    sin_theta = torch.sin(theta)
    directions = torch.stack(
        [
            sin_theta * torch.cos(phi),
            sin_theta * torch.sin(phi),
            torch.cos(theta),
        ],
        dim=-1,
    )
    return F.normalize(directions, dim=-1)


def view_direction_to_latlong(view_direction: torch.Tensor) -> torch.Tensor:
    """Convert Z-up Cartesian directions to ``(theta, phi)`` coordinates.

    ``theta`` is in ``[0, pi]`` and ``phi`` is in ``[-pi, pi]``.  Input
    vectors do not need to be normalized, but callers should normally pass
    unit directions because only the Z component is clamped here for legacy
    numerical compatibility.
    """
    if view_direction.shape[-1] != 3:
        raise ValueError(
            "view_direction must have a final dimension of 3, "
            f"got shape {tuple(view_direction.shape)}"
        )

    x = view_direction[..., 0]
    y = view_direction[..., 1]
    z = view_direction[..., 2]
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)
    return torch.stack([theta, phi], dim=-1)


def sample_env_map(env_map: torch.Tensor, theta_phi: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a lat-long environment map.

    Args:
        env_map: ``(N, C, H, W)``, ``(N, H, W, C)``, or ``(H, W, C)``.
        theta_phi: Sampling coordinates whose last dimension is two.  The
            common forms are ``(M, 2)``, ``(N, M, 2)``, and
            ``(N, H_out, W_out, 2)``.

    Returns:
        Channel-first output for batched inputs.  An unbatched ``HWC`` input
        produces a channel-last output.

    Horizontal coordinates wrap periodically; vertical samples use border
    padding at the poles.
    """
    if theta_phi.shape[-1] != 2:
        raise ValueError(
            "theta_phi must have a final dimension of 2, "
            f"got shape {tuple(theta_phi.shape)}"
        )

    restore_hwc = False
    if env_map.dim() == 3:
        restore_hwc = True
        env_map = env_map.permute(2, 0, 1).unsqueeze(0)
    elif env_map.dim() == 4:
        if env_map.shape[1] not in (1, 2, 3, 4):
            env_map = env_map.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"env_map must be 3D or 4D, got shape {tuple(env_map.shape)}"
        )

    batch_size, channels, _, _ = env_map.shape
    theta = theta_phi[..., 0]
    phi = theta_phi[..., 1]
    v = 2.0 * (theta / torch.pi) - 1.0
    u = torch.remainder(0.5 - phi / (2.0 * torch.pi), 1.0) * 2.0 - 1.0
    grid = torch.stack([u, v], dim=-1).to(
        dtype=env_map.dtype, device=env_map.device
    )

    if grid.dim() == 2:
        spatial_shape = (grid.shape[0],)
        grid_4d = grid.unsqueeze(0).unsqueeze(1)
        grid_batch_size = 1
    elif grid.dim() == 3:
        spatial_shape = (grid.shape[1],)
        grid_4d = grid.unsqueeze(1)
        grid_batch_size = grid.shape[0]
    else:
        if grid.shape[0] == batch_size:
            spatial_shape = tuple(grid.shape[1:-1])
            grid_4d = grid.reshape(batch_size, 1, -1, 2)
            grid_batch_size = batch_size
        else:
            spatial_shape = tuple(grid.shape[:-1])
            grid_4d = grid.reshape(1, 1, -1, 2)
            grid_batch_size = 1

    sampled = F.grid_sample(
        env_map if grid_batch_size == batch_size else env_map[:1],
        grid_4d,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(2)

    if len(spatial_shape) > 1:
        sampled = sampled.reshape(grid_batch_size, channels, *spatial_shape)

    if restore_hwc:
        sampled = sampled.squeeze(0)
        sampled = sampled.permute(*range(1, sampled.dim()), 0)
    return sampled


def rotate_env_map(
    env_map: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Rotate a lat-long environment map with one matrix per batch item.

    ``env_map`` accepts ``NCHW``, ``NHWC``, or unbatched ``HWC`` layout and
    is returned in the same layout.  ``rotation_matrix`` accepts ``(3, 3)``
    or ``(N, 3, 3)``; a single matrix is broadcast across a batched map.
    """
    original_layout: str
    if env_map.dim() == 3:
        original_layout = "hwc"
        env_map_nchw = env_map.permute(2, 0, 1).unsqueeze(0)
    elif env_map.dim() == 4 and env_map.shape[1] in (1, 2, 3, 4):
        original_layout = "nchw"
        env_map_nchw = env_map
    elif env_map.dim() == 4:
        original_layout = "nhwc"
        env_map_nchw = env_map.permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"env_map must be 3D or 4D, got shape {tuple(env_map.shape)}"
        )

    if rotation_matrix.shape == (3, 3):
        rotation_matrix = rotation_matrix.unsqueeze(0)
    if rotation_matrix.dim() != 3 or rotation_matrix.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrix must have shape (3, 3) or (N, 3, 3), "
            f"got {tuple(rotation_matrix.shape)}"
        )

    batch_size, _, h, w = env_map_nchw.shape
    if rotation_matrix.shape[0] == 1 and batch_size != 1:
        rotation_matrix = rotation_matrix.expand(batch_size, -1, -1)
    elif rotation_matrix.shape[0] != batch_size:
        raise ValueError(
            "rotation batch size must be one or match env_map; "
            f"got {rotation_matrix.shape[0]} and {batch_size}"
        )

    rotation_matrix = rotation_matrix.to(device=env_map_nchw.device)
    ray_directions = latlong_ray_direction(
        h, w, device=env_map_nchw.device
    ).to(dtype=rotation_matrix.dtype)
    rotated_directions = torch.matmul(
        rotation_matrix[:, None, None], ray_directions[None, ..., None]
    ).squeeze(-1)
    theta_phi = view_direction_to_latlong(rotated_directions)
    rotated = sample_env_map(env_map_nchw, theta_phi)

    if original_layout == "hwc":
        return rotated.squeeze(0).permute(1, 2, 0)
    if original_layout == "nhwc":
        return rotated.permute(0, 2, 3, 1)
    return rotated
