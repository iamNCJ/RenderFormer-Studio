import torch
from torch.amp import autocast
import roma
from typing import Optional, Tuple, List, Union

from renderformer.data.envmap import rotate_env_map


def generate_random_rotation_and_translation(
    batch_shape: Tuple[int, ...],
    rot_scale: Union[float, List[float]] = 1.0,
    trans_scale: Union[float, List[float]] = 0.25,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> roma.Rigid:
    # rot_scale: if float, use for all axes; if list, must be length 3.
    # trans_scale: if float, use for all axes; if list, must be length 3.
    if isinstance(rot_scale, (float, int)):
        rot_scale_tensor = torch.tensor(rot_scale, dtype=dtype, device=device).expand(3)
    elif isinstance(rot_scale, list) or isinstance(rot_scale, tuple):
        assert len(rot_scale) == 3, "rot_scale as a list/tuple must have 3 elements (x, y, z)"
        rot_scale_tensor = torch.tensor(rot_scale, dtype=dtype, device=device)
    else:
        raise TypeError("rot_scale must be a float, int, list, or tuple")

    if isinstance(trans_scale, (float, int)):
        trans_scale_tensor = torch.tensor(trans_scale, dtype=dtype, device=device).expand(3)
    elif isinstance(trans_scale, list) or isinstance(trans_scale, tuple):
        assert len(trans_scale) == 3, "trans_scale as a list/tuple must have 3 elements (x, y, z)"
        trans_scale_tensor = torch.tensor(trans_scale, dtype=dtype, device=device)
    else:
        raise TypeError("trans_scale must be a float, int, list, or tuple")

    # Sample random vectors and apply per-axis scale
    rotvec = torch.randn(batch_shape + (3,), device=device, dtype=dtype) * rot_scale_tensor
    # print(rotvec.shape)
    # print(rotvec)
    t = torch.randn(batch_shape + (3,), device=device, dtype=dtype) * trans_scale_tensor
    R = roma.rotvec_to_rotmat(rotvec)
    T = roma.Rigid(R, t)
    return T


@torch.no_grad()
@autocast("cuda", enabled=False)
def aug_triangles(triangles: torch.Tensor, vns: Optional[torch.Tensor] = None, rot_scale: float = 1.0, trans_scale: float = 0.25) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Augment triangles by rotation and translation.

    Args:
        triangles: (batch_size, n_tris, 3, 3) tensor of triangles.
        vns: (batch_size, n_tris, 3, 3) tensor of vertex normals.
        rot_scale: rotation scale. Default is 1.0.
        trans_scale: translation scale. Default is 0.25.
    """
    device = triangles.device
    dtype = triangles.dtype
    batch_shape = triangles.shape[:1]
    T = generate_random_rotation_and_translation(batch_shape, rot_scale, trans_scale, device, dtype)
    return T[:, None, None].apply(triangles), T[:, None, None].linear_apply(vns) if vns is not None else None


@torch.no_grad()
@autocast("cuda", enabled=False)
def aug_triangles_and_camera(
    c2w: torch.Tensor,
    triangles: torch.Tensor,
    vns: Optional[torch.Tensor] = None,
    rot_scale: Union[float, List[float]] = 1.0,
    trans_scale: Union[float, List[float]] = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Augment triangles and camera by rotation and translation.

    Args:
        c2w: (batch_size, nv, 4, 4) tensor of camera to world matrices.
        triangles: (batch_size, n_tris, 3, 3) tensor of triangles.
        vns: (batch_size, n_tris, 3, 3) tensor of vertex normals.
        rot_scale: rotation scale (float or 3-list for xyz). Default is 1.0.
        trans_scale: translation scale (float or 3-list for xyz). Default is 0.25.

    Returns:
        triangles_aug: (batch_size, n_tris, 3, 3) tensor of augmented triangles.
        c2w_aug: (batch_size, nv, 4, 4) tensor of augmented camera to world matrices.
        vns_aug: (batch_size, n_tris, 3, 3) tensor of augmented vertex normals.
    """
    device = triangles.device
    dtype = triangles.dtype
    batch_shape = triangles.shape[:1]
    nv = c2w.shape[1]
    T = generate_random_rotation_and_translation(batch_shape, rot_scale, trans_scale, device, dtype)
    T_for_c2w = roma.Rigid(linear=T.linear.repeat_interleave(nv, dim=0), translation=T.translation.repeat_interleave(nv, dim=0))
    c2w_T = roma.Rigid.from_homogeneous(c2w.view(-1, 4, 4))
    return (
        T[:, None, None].apply(triangles),
        T_for_c2w.compose(c2w_T).to_homogeneous().view(-1, nv, 4, 4),
        T[:, None, None].linear_apply(vns) if vns is not None else None
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def apply_rotation_to_env_map(env_map: torch.Tensor, R: roma.Rotation) -> torch.Tensor:
    """
    Rotate environment map using rotation matrix.
    
    Args:
        env_map: (batch_size, 3, h, w) or (batch_size, h, w, 3) tensor
        R: roma.Rotation object with shape (batch_size, ...)
    
    Returns:
        Rotated environment map with same shape as input
    """
    return rotate_env_map(env_map, R.linear)


@torch.no_grad()
@autocast("cuda", enabled=False)
def aug_triangles_and_camera_and_env(
    c2w: torch.Tensor,
    triangles: torch.Tensor,
    vns: Optional[torch.Tensor] = None,
    env_map: Optional[torch.Tensor] = None,
    rot_scale: Union[float, List[float]] = 1.0,
    trans_scale: Union[float, List[float]] = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Augment triangles, camera, and environment map by rotation and translation.

    Args:
        c2w: (batch_size, nv, 4, 4) tensor of camera to world matrices.
        triangles: (batch_size, n_tris, 3, 3) tensor of triangles.
        vns: (batch_size, n_tris, 3, 3) tensor of vertex normals.
        env_map: (batch_size, 3, h, w) or (batch_size, h, w, 3) tensor of environment maps.
        rot_scale: rotation scale (float or 3-list for xyz). Default is 1.0.
        trans_scale: translation scale (float or 3-list for xyz). Default is 0.25.

    Returns:
        triangles_aug: (batch_size, n_tris, 3, 3) tensor of augmented triangles.
        c2w_aug: (batch_size, nv, 4, 4) tensor of augmented camera to world matrices.
        vns_aug: (batch_size, n_tris, 3, 3) tensor of augmented vertex normals.
        env_map_aug: (batch_size, 3, h, w) or (batch_size, h, w, 3) tensor of augmented environment maps.
    """
    device = triangles.device
    dtype = triangles.dtype
    batch_shape = triangles.shape[:1]
    nv = c2w.shape[1]
    T = generate_random_rotation_and_translation(batch_shape, rot_scale, trans_scale, device, dtype)
    T_for_c2w = roma.Rigid(linear=T.linear.repeat_interleave(nv, dim=0), translation=T.translation.repeat_interleave(nv, dim=0))
    c2w_T = roma.Rigid.from_homogeneous(c2w.view(-1, 4, 4))
    
    # Apply rotation to env_map if provided
    env_map_aug = None
    if env_map is not None:
        # Convert R to roma.Rotation for env_map rotation
        R = roma.Rotation(T.linear).inverse()
        env_map_aug = apply_rotation_to_env_map(env_map, R)
    
    return (
        T[:, None, None].apply(triangles),
        T_for_c2w.compose(c2w_T).to_homogeneous().view(-1, nv, 4, 4),
        T[:, None, None].linear_apply(vns) if vns is not None else None,
        env_map_aug
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def aug_triangles_and_camera_and_env_and_vol(
    c2w: torch.Tensor,
    triangles: torch.Tensor,
    vns: Optional[torch.Tensor] = None,
    env_map: Optional[torch.Tensor] = None,
    volume_position: Optional[torch.Tensor] = None,
    volume_rotation: Optional[torch.Tensor] = None,
    volume_mask: Optional[torch.Tensor] = None,
    rot_scale: Union[float, List[float]] = 1.0,
    trans_scale: Union[float, List[float]] = 0.25
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    Augment triangles, camera, environment map, and volumes by rotation and translation.
    """
    device = triangles.device
    dtype = triangles.dtype
    batch_shape = triangles.shape[:1]
    nv = c2w.shape[1]
    T = generate_random_rotation_and_translation(batch_shape, rot_scale, trans_scale, device, dtype)
    T_for_c2w = roma.Rigid(linear=T.linear.repeat_interleave(nv, dim=0), translation=T.translation.repeat_interleave(nv, dim=0))
    c2w_T = roma.Rigid.from_homogeneous(c2w.view(-1, 4, 4))

    env_map_aug = None
    if env_map is not None:
        R = roma.Rotation(T.linear).inverse()
        env_map_aug = apply_rotation_to_env_map(env_map, R)

    volume_position_aug = volume_position
    if volume_position is not None:
        volume_position_aug = T[:, None].apply(volume_position)

    volume_rotation_aug = volume_rotation
    if volume_rotation is not None:
        rotation_dtype = volume_rotation.dtype
        volume_rotmat = roma.euler_to_rotmat(convention='xyz', angles=volume_rotation.float())
        R_aug = T.linear
        volume_rotmat = torch.matmul(R_aug[:, None], volume_rotmat)
        volume_rotation_aug = roma.rotmat_to_euler(convention='xyz', rotmat=volume_rotmat).to(rotation_dtype)

    if volume_mask is not None:
        mask = volume_mask[..., None]
        if volume_position_aug is not None:
            volume_position_aug = torch.where(mask, volume_position_aug, volume_position)
        if volume_rotation_aug is not None:
            volume_rotation_aug = torch.where(mask, volume_rotation_aug, volume_rotation)

    return (
        T[:, None, None].apply(triangles),
        T_for_c2w.compose(c2w_T).to_homogeneous().view(-1, nv, 4, 4),
        T[:, None, None].linear_apply(vns) if vns is not None else None,
        env_map_aug,
        volume_position_aug,
        volume_rotation_aug,
    )


@torch.no_grad()
@autocast("cuda", enabled=False)
def trans_to_cam_coord(c2w: torch.Tensor, triangles: torch.Tensor, vns: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Transform triangles to camera coordinate system.

    Args:
        c2w: (batch_size, 4, 4) tensor of camera to world matrices.
        triangles: (batch_size, n_tris, 3, 3) tensor of triangles.
        vns: (batch_size, n_tris, 3, 3) tensor of vertex

    Returns:
        triangles_cam: (batch_size, n_tris, 3, 3) tensor of triangles in camera coordinate system.
        c2w_cam: (batch_size, 4, 4) tensor of camera to world matrices in camera coordinate system, should always be identity.
        vns_cam: (batch_size, n_tris, 3, 3) tensor of vertex normals in camera coordinate system.
    """
    device = triangles.device
    dtype = triangles.dtype
    T = roma.Rigid.from_homogeneous(c2w)
    T_inv = T.inverse()
    return T_inv[:, None, None].apply(triangles), torch.eye(4, device=device, dtype=dtype).repeat(c2w.shape[0], 1, 1), T_inv[:, None, None].linear_apply(vns) if vns is not None else None
