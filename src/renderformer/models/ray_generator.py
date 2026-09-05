import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union


class RayGenerator(nn.Module):
    """
    Generate rays from camera model.

    Two modes:

    * Legacy: only ``c2w``, ``fov``, ``img_res`` provided — generates a full
      ``img_res x img_res`` ray map identical to the original implementation.
    * Multi-resolution / cropped: ``crop_size``, ``crop_origin`` and/or
      ``source_res`` provided — generates a ``crop_size x crop_size`` ray map
      that corresponds to the patch ``[y0:y0+crop_size, x0:x0+crop_size]`` of
      a virtual ``source_res x source_res`` pixel grid. Focal length is derived
      from ``source_res`` (which may be per-sample), so different samples in
      a batch can come from H5 files rendered at different resolutions while
      sharing the same cropped patch size.
    """
    def __init__(self):
        super().__init__()

    def forward(
            self,
            c2w: torch.Tensor,
            fov: torch.Tensor,
            img_res: int = 256,
            crop_origin: Optional[torch.Tensor] = None,
            crop_size: Optional[int] = None,
            source_res: Optional[Union[int, float, torch.Tensor]] = None,
    ):
        """
        Args:
            c2w: ``(*BATCH_SHAPE, 4, 4)``
            fov: ``(*BATCH_SHAPE, 1)`` — full vertical field of view in radians
            img_res: legacy full-image size. Used as the default for both
                ``crop_size`` and ``source_res`` when those are not provided.
            crop_origin: ``(*BATCH_SHAPE, 2)`` integer / float tensor giving
                ``(y0, x0)`` in source-pixel coordinates (top-left of the patch).
                Defaults to all zeros.
            crop_size: side of the square ray patch in pixels. Defaults to
                ``img_res``.
            source_res: scalar or ``(*BATCH_SHAPE,)`` tensor giving the source
                image resolution (assumed square) used to derive focal length
                and principal point. Defaults to ``img_res``.

        Returns:
            rays_o: ``(*BATCH_SHAPE, 3)`` — perspective camera origin
            rays_d: ``(*BATCH_SHAPE, crop_size, crop_size, 3)``
        """
        batch_shape = c2w.shape[:-2]
        n_batch = len(batch_shape)
        device, dtype = c2w.device, c2w.dtype

        if crop_size is None:
            crop_size = img_res
        if source_res is None:
            source_res = img_res

        # Pixel-center coordinates within the patch: 0.5, 1.5, ..., crop_size - 0.5
        crop_idx = torch.arange(crop_size, device=device, dtype=dtype) + 0.5

        if crop_origin is None:
            y_pix_local = crop_idx.view(*([1] * n_batch), crop_size, 1)
            x_pix_local = crop_idx.view(*([1] * n_batch), 1, crop_size)
            y_pix = y_pix_local.expand(*batch_shape, crop_size, crop_size)
            x_pix = x_pix_local.expand(*batch_shape, crop_size, crop_size)
        else:
            crop_origin = crop_origin.to(device=device, dtype=dtype)
            y0 = crop_origin[..., 0].unsqueeze(-1).unsqueeze(-1)  # (*BATCH, 1, 1)
            x0 = crop_origin[..., 1].unsqueeze(-1).unsqueeze(-1)
            yi = crop_idx.view(*([1] * n_batch), crop_size, 1)
            xi = crop_idx.view(*([1] * n_batch), 1, crop_size)
            y_pix = y0 + yi
            x_pix = x0 + xi
            y_pix = y_pix.expand(*batch_shape, crop_size, crop_size)
            x_pix = x_pix.expand(*batch_shape, crop_size, crop_size)

        # source_res broadcast to (*BATCH, 1, 1) so cx/cy/fx/fy broadcast cleanly
        if torch.is_tensor(source_res):
            sr = source_res.to(device=device, dtype=dtype).view(*batch_shape, 1, 1)
        else:
            sr = float(source_res)

        cx = cy = sr / 2.0
        # fov[..., 0]: (*BATCH,) -> (*BATCH, 1, 1)
        tan_half_fov = torch.tan(0.5 * fov[..., 0, None, None])
        fx = fy = sr / 2.0 / tan_half_fov

        dirs = torch.stack([
            (x_pix - cx) / fx,
            -(y_pix - cy) / fy,
            -torch.ones_like(x_pix),
        ], dim=-1)  # (*BATCH, H, W, 3)

        R = c2w[..., :3, :3]  # (*BATCH, 3, 3)
        t = c2w[..., :3, 3]   # (*BATCH, 3)

        rays_d = torch.sum(dirs[..., None, :] * R[..., None, None, :, :], dim=-1)
        rays_d = F.normalize(rays_d, dim=-1, p=2)

        return t, rays_d


if __name__ == '__main__':
    ray_generator = RayGenerator()
    bs, V, X = 7, 5, 3
    c2w = torch.randn(bs, V, X, 4, 4)
    fov = torch.randn(bs, V, X, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)

    bs, V = 15, 9
    c2w = torch.randn(bs, V, 4, 4)
    fov = torch.randn(bs, V, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)

    bs = 11
    c2w = torch.randn(bs, 4, 4)
    fov = torch.randn(bs, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)

    # multires / cropped path
    bs, V = 4, 2
    c2w = torch.randn(bs, V, 4, 4)
    fov = torch.full((bs, V, 1), 0.6)
    crop_origin = torch.zeros(bs, V, 2)
    source_res = torch.tensor([256., 256., 512., 512.])[:, None].expand(bs, V)
    rays_o, rays_d = ray_generator(
        c2w, fov, crop_origin=crop_origin, crop_size=128, source_res=source_res,
    )
    print(rays_o.shape, rays_d.shape)
