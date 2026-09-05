"""Shared infrastructure for RenderFormer inference pipelines."""

from __future__ import annotations

import contextlib
import dataclasses
import math
from pathlib import Path

import torch

from renderformer.checkpoints import (
    load_shared_model,
    resolve_checkpoint,
    select_v2_transformer_subfolder,
)
from renderformer.config import CheckpointVersion, InferenceConfig
from renderformer.data import SceneData, load_h5_scene
from renderformer.models.aug import trans_to_cam_coord
from renderformer.models.ray_generator import RayGenerator


@dataclasses.dataclass(frozen=True)
class RenderOutput:
    """Unmodified model output and decoded linear HDR output."""

    raw: torch.Tensor
    hdr: torch.Tensor


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    choices = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in choices:
        raise ValueError(f"precision must be one of {sorted(choices)}; got {precision}")
    dtype = choices[precision]
    if device.type != "cuda" and dtype is not torch.float32:
        raise ValueError(
            f"{precision} inference is only supported on CUDA; "
            f"use fp32 on {device.type}"
        )
    return dtype


class _RenderFormerPipelineBase:
    checkpoint_version: CheckpointVersion
    default_pretrained_source: str | None = None

    def __init__(
        self,
        model,
        config: InferenceConfig,
        checkpoint_dir: str | Path,
        *,
        fused_kernels_requested: bool = True,
        kernels_applied: bool = False,
        transformer_subfolder: str | None = None,
    ):
        if config.runtime.version is not self.checkpoint_version:
            raise ValueError(
                f"{type(self).__name__} requires a {self.checkpoint_version.value} "
                f"checkpoint, got {config.runtime.version.value}"
            )
        self.model = model
        self.transformer = model
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.transformer_subfolder = transformer_subfolder
        self.ray_generator = RayGenerator().to(self.device)
        self._fused_kernels_requested = fused_kernels_requested
        self._kernels_applied = kernels_applied
        self._env_vae = None
        self._env_vae_is_video = False
        self._texture_vae = None
        self._texture_vae_is_video = False
        self._component_cache_dir = None
        self._component_local_files_only = False
        self._component_revisions = {}
        self._component_registry = {}
        self._component_records = {}

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        token: str | bool | None = None,
        local_files_only: bool = False,
        subfolder: str | None = None,
        transformer_subfolder: str | None = None,
        resolution: int | None = None,
        device: str | torch.device = "cpu",
        apply_fused_kernels: bool = True,
        torch_dtype: torch.dtype | None = None,
        load_auxiliary_models: bool = True,
        environment_encoder=None,
        texture_encoder=None,
        environment_encoder_is_video: bool | None = None,
        texture_encoder_is_video: bool | None = None,
        environment_encoder_dtype: torch.dtype | None = None,
        texture_encoder_dtype: torch.dtype | None = None,
        component_revisions: dict[str, str] | None = None,
    ):
        if source is None:
            source = cls.default_pretrained_source
        if source is None:
            raise TypeError(
                f"{cls.__name__}.from_pretrained() requires a checkpoint source"
            )
        if subfolder is not None and transformer_subfolder is not None:
            raise ValueError(
                "pass only one of subfolder or transformer_subfolder; they refer "
                "to the same transformer component"
            )
        if subfolder is not None:
            transformer_subfolder = subfolder
        if cls.checkpoint_version is CheckpointVersion.V2:
            transformer_subfolder = select_v2_transformer_subfolder(
                source,
                resolution=512 if resolution is None else resolution,
                subfolder=transformer_subfolder,
            )
        elif resolution is not None:
            raise ValueError(
                "resolution-based component selection is only valid for "
                "RenderFormerV2Pipeline"
            )
        checkpoint_dir = resolve_checkpoint(
            source,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            subfolder=transformer_subfolder,
        )
        device = torch.device(device)
        model, config = load_shared_model(
            checkpoint_dir,
            device=device,
            apply_fused_kernels=apply_fused_kernels,
            subfolder=transformer_subfolder,
        )
        if config.runtime.version is not cls.checkpoint_version:
            raise ValueError(
                f"{cls.__name__} requires a {cls.checkpoint_version.value} "
                f"checkpoint, got {config.runtime.version.value}"
            )
        pipeline_kwargs = {}
        if cls.checkpoint_version is CheckpointVersion.V2:
            pipeline_kwargs = {
                "torch_dtype": torch_dtype,
                "load_auxiliary_models": load_auxiliary_models,
                "environment_encoder": environment_encoder,
                "texture_encoder": texture_encoder,
                "environment_encoder_is_video": environment_encoder_is_video,
                "texture_encoder_is_video": texture_encoder_is_video,
                "environment_encoder_dtype": environment_encoder_dtype,
                "texture_encoder_dtype": texture_encoder_dtype,
                "component_cache_dir": cache_dir,
                "component_force_download": force_download,
                "component_token": token,
                "component_local_files_only": local_files_only,
                "component_revisions": component_revisions,
            }
        elif any(
            value is not None
            for value in (
                torch_dtype,
                environment_encoder,
                texture_encoder,
                environment_encoder_is_video,
                texture_encoder_is_video,
                environment_encoder_dtype,
                texture_encoder_dtype,
                component_revisions,
            )
        ):
            raise ValueError(
                "torch_dtype and auxiliary encoder options are only valid for "
                "RenderFormerV2Pipeline; select V1 render precision per call"
            )
        return cls(
            model,
            config,
            checkpoint_dir,
            fused_kernels_requested=apply_fused_kernels,
            kernels_applied=apply_fused_kernels and device.type == "cuda",
            transformer_subfolder=transformer_subfolder,
            **pipeline_kwargs,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def fused_kernels_requested(self) -> bool:
        """Whether fused CUDA kernels were requested for this pipeline."""

        return self._fused_kernels_requested

    @property
    def fused_kernels_applied(self) -> bool:
        """Whether fused CUDA kernels are currently installed on the model."""

        return self._kernels_applied

    def to(self, device: str | torch.device):
        device = torch.device(device)
        if (
            device.type == "cuda"
            and self._fused_kernels_requested
            and not self._kernels_applied
        ):
            from renderformer.ops.apply_kernels import apply_kernels

            self.model = apply_kernels(self.model, verbose=False)
            self.transformer = self.model
            self._kernels_applied = True
        self.model.to(device)
        self.ray_generator.to(device)
        if self._env_vae is not None:
            self._env_vae.to(device)
        if self._texture_vae is not None and self._texture_vae is not self._env_vae:
            self._texture_vae.to(device)
        self.model.eval().requires_grad_(False)
        return self

    def optimize(self, level: str = "compile"):
        from renderformer.optimizations import apply_inference_optimizations

        self.model = apply_inference_optimizations(self.model, level=level)
        self.transformer = self.model
        return self

    def load_h5(
        self,
        path: str | Path,
        *,
        allow_legacy_rf1_dtypes: bool = False,
    ) -> SceneData:
        return load_h5_scene(
            path,
            self.config.runtime.version,
            image_key=self.config.runtime.image_key,
            allow_legacy_rf1_dtypes=allow_legacy_rf1_dtypes,
        )

    def _autocast(self, dtype: torch.dtype):
        if dtype is torch.float32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    @contextlib.contextmanager
    def _cuda_tf32_backend(self, enabled: bool):
        """Own and restore CUDA TF32 backend state for one render call."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if self.device.type != "cuda":
            yield
            return

        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul
            torch.backends.cudnn.allow_tf32 = previous_cudnn

    def _camera_inputs(
        self,
        triangles: torch.Tensor,
        c2w: torch.Tensor,
        fov: torch.Tensor,
        resolution: int,
    ):
        batch_size, num_views = c2w.shape[:2]
        if self.config.runtime.turn_to_cam_coord:
            view_triangles, ray_c2w, _ = trans_to_cam_coord(
                c2w.reshape(-1, 4, 4),
                triangles.repeat_interleave(num_views, dim=0),
            )
            view_triangles = view_triangles.reshape(
                batch_size, num_views, -1, 3, 3
            )
            ray_c2w = ray_c2w.reshape(batch_size, num_views, 4, 4)
        else:
            view_triangles = triangles[:, None].expand(
                -1, num_views, -1, -1, -1
            )
            ray_c2w = c2w
        if fov.ndim == 2:
            fov = fov[..., None]
        rays_o, rays_d = self.ray_generator(
            ray_c2w,
            fov / 180.0 * math.pi,
            resolution,
        )
        return view_triangles, rays_o, rays_d
