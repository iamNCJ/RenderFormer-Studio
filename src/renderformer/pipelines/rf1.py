"""RF1 inference pipeline and legacy tensor compatibility API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from renderformer.config import CheckpointVersion
from renderformer.data import SceneData
from renderformer.pipelines.base import (
    RenderOutput,
    _precision_dtype,
    _RenderFormerPipelineBase,
)


def _log10_v1_emission_channels_(texture: torch.Tensor) -> None:
    """Apply the legacy HDR transform to the final three texture channels.

    RF1 Base keeps patch textures as ``(B,N,C,H,W)`` while Large collapses
    single-value textures to ``(B,N,C)``.  In both layouts the channel axis is
    dimension 2; an ellipsis-only slice would incorrectly select image columns
    for the Base layout.
    """

    if texture.ndim not in {3, 5} or texture.shape[2] < 3:
        raise ValueError(
            "V1 texture must have shape (B,N,C) or (B,N,C,H,W) with at "
            f"least three channels, got {tuple(texture.shape)}"
        )
    emission = texture[:, :, -3:, ...]
    texture[:, :, -3:, ...] = torch.log10(emission + 1.0)


class RenderFormerPipeline(_RenderFormerPipelineBase):
    """V1/RF1 inference pipeline.

    The tensor ``render``/``__call__`` contract is kept compatible with the
    original public V1 package while using the shared training model.
    """

    checkpoint_version = CheckpointVersion.V1

    def _render_v1_batch(
        self,
        *,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor,
        vn: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
        dtype: torch.dtype,
    ) -> RenderOutput:
        low_precision = dtype in {torch.float16, torch.bfloat16}
        with self._cuda_tf32_backend(low_precision):
            return self._render_v1_batch_impl(
                triangles=triangles,
                texture=texture,
                mask=mask,
                vn=vn,
                c2w=c2w,
                fov=fov,
                resolution=resolution,
                dtype=dtype,
            )

    def _render_v1_batch_impl(
        self,
        *,
        triangles: torch.Tensor,
        texture: torch.Tensor,
        mask: torch.Tensor,
        vn: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
        dtype: torch.dtype,
    ) -> RenderOutput:
        if self.config.runtime.version is not CheckpointVersion.V1:
            raise ValueError("the tensor compatibility API is only defined for V1")
        triangles = triangles.to(self.device, dtype=torch.float32)
        texture = texture.to(self.device, dtype=torch.float32).clone()
        mask = mask.to(self.device, dtype=torch.bool)
        vn = vn.to(self.device, dtype=torch.float32)
        c2w = c2w.to(self.device, dtype=torch.float32)
        fov = fov.to(self.device, dtype=torch.float32)
        if self.config.model_config.texture_encode_patch_size == 1 and texture.ndim == 5:
            texture = texture[..., 0, 0]
        if not self.config.runtime.learn_ldr:
            _log10_v1_emission_channels_(texture)

        view_triangles, rays_o, rays_d = self._camera_inputs(
            triangles, c2w, fov, resolution
        )
        batch_size, num_views = c2w.shape[:2]
        mvps = torch.zeros(
            (batch_size, 16), device=self.device, dtype=torch.float32
        )
        low_precision = dtype in {torch.float16, torch.bfloat16}
        with torch.inference_mode(), self._autocast(dtype):
            raw = self.model(
                triangles.reshape(batch_size, -1, 9),
                texture,
                mvps,
                mask,
                vn.reshape(batch_size, -1, 9),
                rays_o=rays_o,
                rays_d=rays_d,
                tri_vpos_view_tf=view_triangles.reshape(
                    batch_size, num_views, -1, 9
                ),
                tf32_view_tf=low_precision,
                use_packed_sequence=False,
            )
        hdr = raw.permute(0, 1, 3, 4, 2)
        if not self.config.runtime.learn_ldr:
            hdr = torch.pow(10.0, hdr) - 1.0
        return RenderOutput(raw=raw, hdr=hdr)

    def render(
        self,
        triangles,
        texture,
        mask,
        vn,
        c2w,
        fov,
        resolution: int = 512,
        torch_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """Compatibility API for the released V1 rendering pipeline."""

        if torch_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"unsupported torch dtype: {torch_dtype}")
        if self.device.type != "cuda" and torch_dtype is not torch.float32:
            raise ValueError("non-CUDA V1 inference requires torch.float32")
        return self._render_v1_batch(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            dtype=torch_dtype,
        ).hdr

    __call__ = render

    def render_scene(
        self,
        scene: SceneData,
        *,
        resolution: int = 512,
        precision: str = "fp16",
        first_view_only: bool = False,
        execution_profile: Literal[
            "checkpoint", "legacy_v2", "legacy_template_v2"
        ] = "checkpoint",
        view_transformer_tf32: bool | None = None,
        use_packed_sequence: bool | None = None,
    ) -> RenderOutput:
        dtype = _precision_dtype(precision, self.device)
        if execution_profile != "checkpoint":
            raise ValueError("legacy V2 execution profiles require a V2 checkpoint")
        if view_transformer_tf32 is not None or use_packed_sequence is not None:
            raise ValueError("V2 execution overrides cannot be used with a V1 checkpoint")
        num_views = 1 if first_view_only else scene.num_views
        mask = torch.ones((1, scene.num_triangles), dtype=torch.bool)
        return self._render_v1_batch(
            triangles=scene.triangles[None],
            texture=scene.texture[None],
            mask=mask,
            vn=scene.vn[None],
            c2w=scene.c2w[None, :num_views],
            fov=scene.fov[None, :num_views],
            resolution=resolution,
            dtype=dtype,
        )

    def render_v1_scenes(
        self,
        scenes: list[SceneData],
        *,
        resolution: int = 512,
        precision: str = "fp16",
        padding_length: int | None = None,
        first_view_only: bool = False,
    ) -> list[RenderOutput]:
        """Pad and render an RF1 batch while preserving the legacy mask contract."""

        if self.config.runtime.version is not CheckpointVersion.V1:
            raise ValueError("render_v1_scenes requires a V1 checkpoint")
        if not scenes:
            return []
        dtype = _precision_dtype(precision, self.device)
        view_counts = {1 if first_view_only else scene.num_views for scene in scenes}
        if len(view_counts) != 1:
            raise ValueError("all RF1 scenes in one batch must have the same view count")
        max_triangles = max(scene.num_triangles for scene in scenes)
        target_length = max_triangles if padding_length is None else padding_length
        if target_length < max_triangles:
            raise ValueError(
                f"padding_length={target_length} is smaller than a scene with "
                f"{max_triangles} triangles"
            )

        def pad(tensor: torch.Tensor):
            padding = torch.zeros(
                (target_length - tensor.shape[0], *tensor.shape[1:]),
                dtype=tensor.dtype,
            )
            return torch.cat([tensor, padding], dim=0)

        triangles = torch.stack([pad(scene.triangles) for scene in scenes])
        texture = torch.stack([pad(scene.texture) for scene in scenes])
        vn = torch.stack([pad(scene.vn) for scene in scenes])
        mask = torch.zeros((len(scenes), target_length), dtype=torch.bool)
        for index, scene in enumerate(scenes):
            mask[index, : scene.num_triangles] = True
        num_views = next(iter(view_counts))
        c2w = torch.stack([scene.c2w[:num_views] for scene in scenes])
        fov = torch.stack([scene.fov[:num_views] for scene in scenes])
        result = self._render_v1_batch(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=resolution,
            dtype=dtype,
        )
        return [
            RenderOutput(raw=result.raw[index : index + 1], hdr=result.hdr[index : index + 1])
            for index in range(len(scenes))
        ]

    def __call__(self, scene_or_triangles=None, *args, **kwargs):
        """Render an H5/SceneData input, or retain the legacy tensor call."""

        if isinstance(scene_or_triangles, (SceneData, str, Path)):
            if args:
                raise TypeError("H5/SceneData pipeline calls accept keyword arguments only")
            allow_legacy_rf1_dtypes = kwargs.pop(
                "allow_legacy_rf1_dtypes",
                False,
            )
            if isinstance(scene_or_triangles, SceneData) and allow_legacy_rf1_dtypes:
                raise ValueError(
                    "allow_legacy_rf1_dtypes applies while loading an H5 path, "
                    "not to an existing SceneData object"
                )
            scene = (
                scene_or_triangles
                if isinstance(scene_or_triangles, SceneData)
                else self.load_h5(
                    scene_or_triangles,
                    allow_legacy_rf1_dtypes=allow_legacy_rf1_dtypes,
                )
            )
            return self.render_scene(scene, **kwargs)
        if scene_or_triangles is None:
            if "triangles" not in kwargs:
                raise TypeError("missing scene/H5 input or triangles tensor")
            return self.render(*args, **kwargs)
        return self.render(scene_or_triangles, *args, **kwargs)


# Public compatibility name used by the original V1 release.
RenderFormerRenderingPipeline = RenderFormerPipeline
