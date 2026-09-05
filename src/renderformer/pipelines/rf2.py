"""RF2 inference pipeline with auxiliary encoder and scene handling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import torch

from renderformer.config import CheckpointVersion, InferenceConfig
from renderformer.checkpoints import DEFAULT_V2_REPOSITORY
from renderformer.data import SceneData
from renderformer.models.tri_utils import compute_triangle_area
from renderformer.pipelines.base import (
    RenderOutput,
    _precision_dtype,
    _RenderFormerPipelineBase,
)


class RenderFormerV2Pipeline(_RenderFormerPipelineBase):
    """V2 inference pipeline with environment/volume component handling."""

    checkpoint_version = CheckpointVersion.V2
    default_pretrained_source = DEFAULT_V2_REPOSITORY
    AUTO_SINGLE_VIEW_MIN_RESOLUTION = 2048

    def __init__(
        self,
        model,
        config: InferenceConfig,
        checkpoint_dir: str | Path,
        *,
        fused_kernels_requested: bool = True,
        kernels_applied: bool = False,
        transformer_subfolder: str | None = None,
        torch_dtype: torch.dtype | None = None,
        load_auxiliary_models: bool = True,
        environment_encoder=None,
        texture_encoder=None,
        environment_encoder_is_video: bool | None = None,
        texture_encoder_is_video: bool | None = None,
        environment_encoder_dtype: torch.dtype | None = None,
        texture_encoder_dtype: torch.dtype | None = None,
        component_cache_dir: str | Path | None = None,
        component_force_download: bool = False,
        component_token: str | bool | None = None,
        component_local_files_only: bool = False,
        component_revisions: dict[str, str] | None = None,
    ):
        super().__init__(
            model,
            config,
            checkpoint_dir,
            fused_kernels_requested=fused_kernels_requested,
            kernels_applied=kernels_applied,
            transformer_subfolder=transformer_subfolder,
        )
        self._component_cache_dir = component_cache_dir
        self._component_force_download = component_force_download
        self._component_token = component_token
        self._component_local_files_only = component_local_files_only
        if component_revisions is not None:
            self._component_revisions.update(component_revisions)
        for name, value in (
            ("torch_dtype", torch_dtype),
            ("environment_encoder_dtype", environment_encoder_dtype),
            ("texture_encoder_dtype", texture_encoder_dtype),
        ):
            if value is not None:
                self._validate_auxiliary_dtype(name, value)
        for name, value in (
            ("load_auxiliary_models", load_auxiliary_models),
            ("environment_encoder_is_video", environment_encoder_is_video),
            ("texture_encoder_is_video", texture_encoder_is_video),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if environment_encoder_is_video is not None and environment_encoder is None:
            raise ValueError(
                "environment_encoder_is_video is only valid with an injected "
                "environment_encoder"
            )
        if texture_encoder_is_video is not None and texture_encoder is None:
            raise ValueError(
                "texture_encoder_is_video is only valid with an injected "
                "texture_encoder"
            )
        default_environment_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._environment_encoder_dtype = (
            environment_encoder_dtype
            if environment_encoder_dtype is not None
            else (
                torch_dtype
                if torch_dtype is not None
                else default_environment_dtype
            )
        )
        self._texture_encoder_dtype = (
            texture_encoder_dtype
            if texture_encoder_dtype is not None
            else torch_dtype if torch_dtype is not None else torch.float32
        )
        self._env_vae = (
            None
            if environment_encoder is None
            else self._validate_injected_component(
                "environment_encoder", environment_encoder
            )
        )
        self._texture_vae = (
            None
            if texture_encoder is None
            else self._validate_injected_component("texture_encoder", texture_encoder)
        )
        if (
            self._env_vae is not None
            and self._env_vae is self._texture_vae
            and self._environment_encoder_dtype is not self._texture_encoder_dtype
        ):
            raise ValueError(
                "one injected encoder cannot serve environment and texture roles "
                "with different dtypes"
            )
        resolved_environment_is_video = (
            type(self._env_vae).__name__ == "AutoencoderKLQwenImage"
            if self._env_vae is not None and environment_encoder_is_video is None
            else environment_encoder_is_video
        )
        resolved_texture_is_video = (
            type(self._texture_vae).__name__ == "AutoencoderKLQwenImage"
            if self._texture_vae is not None and texture_encoder_is_video is None
            else texture_encoder_is_video
        )
        if (
            self._env_vae is not None
            and self._env_vae is self._texture_vae
            and resolved_environment_is_video is not resolved_texture_is_video
        ):
            raise ValueError(
                "one injected encoder cannot serve environment and texture roles "
                "with different is_video conventions"
            )
        if self._env_vae is not None:
            assert resolved_environment_is_video is not None
            self._env_vae_is_video = resolved_environment_is_video
            self._env_vae = self._prepare_auxiliary_component(
                "environment_encoder",
                self._env_vae,
                device=self.device,
                dtype=self._environment_encoder_dtype,
            )
        if self._texture_vae is not None:
            assert resolved_texture_is_video is not None
            self._texture_vae_is_video = resolved_texture_is_video
            self._texture_vae = self._prepare_auxiliary_component(
                "texture_encoder",
                self._texture_vae,
                device=self.device,
                dtype=self._texture_encoder_dtype,
            )
        if self._env_vae is not None:
            self._record_injected_component("environment_encoder", self._env_vae)
        if self._texture_vae is not None:
            self._record_injected_component("texture_encoder", self._texture_vae)
        if load_auxiliary_models:
            if self.config.model_config.use_env_lighting:
                self._ensure_env_vae()
        elif (
            self.config.model_config.use_env_lighting
            and not self.config.model_config.envmap_vae_model_id
            and self._env_vae is None
        ):
            raise ValueError(
                "checkpoint enables environment lighting without "
                "envmap_vae_model_id or an injected environment_encoder"
            )

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        return str(dtype).removeprefix("torch.")

    @staticmethod
    def _validate_auxiliary_dtype(name: str, dtype) -> torch.dtype:
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype, got {type(dtype).__name__}")
        if not dtype.is_floating_point:
            raise ValueError(f"{name} must be floating point, got {dtype}")
        return dtype

    @staticmethod
    def _validate_injected_component(name: str, component) -> torch.nn.Module:
        if not isinstance(component, torch.nn.Module):
            raise TypeError(
                f"{name} must be a torch.nn.Module, got {type(component).__name__}"
            )
        if not callable(getattr(component, "encode", None)):
            raise TypeError(f"{name} must provide a callable encode method")
        return component

    @classmethod
    def _component_dtype(cls, name: str, component: torch.nn.Module) -> torch.dtype:
        dtypes = {
            parameter.dtype
            for parameter in component.parameters()
            if parameter.is_floating_point()
        }
        if not dtypes:
            raise ValueError(f"{name} must expose floating-point parameters")
        if len(dtypes) != 1:
            names = sorted(cls._dtype_name(dtype) for dtype in dtypes)
            raise ValueError(f"{name} has mixed floating-point parameter dtypes: {names}")
        return next(iter(dtypes))

    @classmethod
    def _prepare_auxiliary_component(
        cls,
        name: str,
        component,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        component = cls._validate_injected_component(name, component)
        component.eval().requires_grad_(False).to(device=device, dtype=dtype)
        actual_dtype = cls._component_dtype(name, component)
        if actual_dtype is not dtype:
            raise RuntimeError(
                f"{name} did not honor requested dtype {dtype}; got {actual_dtype}"
            )
        return component

    def _load_auxiliary_component(
        self,
        *,
        role: str,
        model_id: str,
        loader_kind: Literal["qwen", "dc", "kl"],
        subfolder: str | None,
        dtype: torch.dtype,
    ):
        revision = self._component_revisions.get(model_id)
        key = (model_id, revision, subfolder, self._dtype_name(dtype), loader_kind)
        if key in self._component_registry:
            component = self._component_registry[key]
            reused = True
        else:
            try:
                import diffusers
            except ImportError as exc:
                raise RuntimeError(
                    "V2 auxiliary encoders require the inference extra"
                ) from exc
            loader_name = {
                "qwen": "AutoencoderKLQwenImage",
                "dc": "AutoencoderDC",
                "kl": "AutoencoderKL",
            }[loader_kind]
            loader_class = getattr(diffusers, loader_name)
            kwargs = {
                "torch_dtype": dtype,
                "revision": revision,
                "cache_dir": self._component_cache_dir,
                "force_download": self._component_force_download,
                "token": self._component_token,
                "local_files_only": self._component_local_files_only,
            }
            if subfolder is not None:
                kwargs["subfolder"] = subfolder
            component = loader_class.from_pretrained(model_id, **kwargs)
            component = self._prepare_auxiliary_component(
                role,
                component,
                device=self.device,
                dtype=dtype,
            )
            self._component_registry[key] = component
            reused = False
        actual_dtype = self._component_dtype(role, component)
        if actual_dtype is not dtype:
            raise RuntimeError(
                f"{role} dtype changed after loading: expected {dtype}, got {actual_dtype}"
            )
        self._component_records[role] = {
            "loaded": True,
            "source": "pretrained",
            "model_id": model_id,
            "revision": revision,
            "subfolder": subfolder,
            "dtype": self._dtype_name(actual_dtype),
            "class": type(component).__name__,
            "is_video": loader_kind == "qwen",
            "reused": reused,
        }
        return component

    def _record_injected_component(self, role: str, component) -> None:
        dtype = self._component_dtype(role, component)
        self._component_records[role] = {
            "loaded": True,
            "source": "injected",
            "model_id": None,
            "revision": None,
            "subfolder": None,
            "dtype": self._dtype_name(dtype),
            "class": type(component).__name__,
            "is_video": (
                self._env_vae_is_video
                if role == "environment_encoder"
                else self._texture_vae_is_video
            ),
            "reused": self._env_vae is self._texture_vae,
        }

    def _ensure_env_vae(self):
        if self._env_vae is not None:
            return
        model_id = self.config.model_config.envmap_vae_model_id
        if not model_id:
            raise ValueError(
                "checkpoint enables environment lighting without "
                "envmap_vae_model_id or an injected environment_encoder"
            )
        if model_id == "Qwen/Qwen-Image":
            self._env_vae = self._load_auxiliary_component(
                role="environment_encoder",
                model_id=model_id,
                loader_kind="qwen",
                subfolder="vae",
                dtype=self._environment_encoder_dtype,
            )
            self._env_vae_is_video = True
        else:
            self._env_vae = self._load_auxiliary_component(
                role="environment_encoder",
                model_id=model_id,
                loader_kind="kl",
                subfolder="vae",
                dtype=self._environment_encoder_dtype,
            )
            self._env_vae_is_video = False

    def _texture_encoder_spec(self) -> tuple[str, bool]:
        channels = self.config.model_config.texture_channels
        if channels in {64, 80}:
            return "Qwen/Qwen-Image", True
        if channels in {128, 160}:
            return "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", False
        raise ValueError(
            "this checkpoint does not declare a known learned texture encoder; "
            f"provide pre-encoded {channels}-channel textures"
        )

    def _ensure_texture_vae(self):
        if self._texture_vae is not None:
            return
        model_id, is_video = self._texture_encoder_spec()
        self._texture_vae = self._load_auxiliary_component(
            role="texture_encoder",
            model_id=model_id,
            loader_kind="qwen" if is_video else "dc",
            subfolder="vae" if is_video else None,
            dtype=self._texture_encoder_dtype,
        )
        self._texture_vae_is_video = is_video

    def _validate_texture_input(self, texture: torch.Tensor) -> bool:
        """Validate one V2 texture and return whether it is pre-encoded."""

        expected_channels = self.config.model_config.texture_channels
        if texture.ndim in {2, 4} and texture.shape[1] == expected_channels:
            patch_size = self.config.model_config.texture_encode_patch_size
            if texture.ndim == 2 and patch_size != 1:
                raise ValueError(
                    "2D pre-encoded textures are only valid when "
                    f"texture_encode_patch_size=1, got {patch_size}"
                )
            if texture.ndim == 4:
                if texture.shape[2] != texture.shape[3]:
                    raise ValueError(
                        f"texture patches must be square, got {tuple(texture.shape[2:])}"
                    )
                if (
                    not self.config.model_config.auto_interpolate_texture
                    and texture.shape[2] != patch_size
                ):
                    raise ValueError(
                        "pre-encoded texture resolution must match "
                        f"texture_encode_patch_size={patch_size}, got {texture.shape[2]}"
                    )
            return True

        if texture.ndim != 4 or texture.shape[1] < 12:
            raise ValueError(
                "V2 texture must already match the checkpoint channels or be raw "
                f"(N, >=12, H, W); got {tuple(texture.shape)}"
            )
        self._texture_encoder_spec()
        if (
            expected_channels in {80, 160}
            and texture.shape[1] not in {15}
            and texture.shape[1] < 16
        ):
            raise ValueError(
                f"{expected_channels}-channel checkpoints require raw heightmap "
                "channel 15 or the historical 15-channel zero-heightmap format"
            )
        return False

    def encode_texture(
        self,
        texture: torch.Tensor,
        *,
        dtype: torch.dtype,
        triangle_batch_size: int = 256,
    ) -> torch.Tensor:
        """Return a checkpoint-ready V2 texture, encoding raw maps if needed."""

        expected_channels = self.config.model_config.texture_channels
        if self._validate_texture_input(texture):
            return texture.to(self.device, dtype=torch.float32).clone()
        if triangle_batch_size <= 0:
            raise ValueError("triangle_batch_size must be positive")

        self._ensure_texture_vae()
        vae = self._texture_vae
        assert vae is not None
        vae_dtype = next(vae.parameters()).dtype
        encoded_chunks = []
        for start in range(0, texture.shape[0], triangle_batch_size):
            batch = texture[start : start + triangle_batch_size].to(
                self.device, dtype=vae_dtype
            )
            groups = []
            with torch.inference_mode():
                for channel_start in (0, 3, 6, 9):
                    component = batch[:, channel_start : channel_start + 3]
                    if self._texture_vae_is_video:
                        latent = vae.encode(component[:, :, None]).latent_dist.mean[:, :, 0]
                    else:
                        latent = vae.encode(component).latent
                    groups.append(latent)
                if expected_channels in {80, 160}:
                    if batch.shape[1] == 15:
                        heightmap = torch.zeros(
                            (batch.shape[0], 1, *batch.shape[2:]),
                            device=batch.device,
                            dtype=batch.dtype,
                        )
                    elif batch.shape[1] >= 16:
                        heightmap = batch[:, 15:16]
                    else:
                        raise ValueError(
                            f"{expected_channels}-channel checkpoints require raw heightmap "
                            "channel 15 or the historical 15-channel zero-heightmap format"
                        )
                    heightmap = heightmap.repeat(1, 3, 1, 1)
                    if self._texture_vae_is_video:
                        latent = vae.encode(heightmap[:, :, None]).latent_dist.mean[:, :, 0]
                    else:
                        latent = vae.encode(heightmap).latent
                    groups.append(latent)
            encoded_chunks.append(torch.cat(groups, dim=1))
        encoded = torch.cat(encoded_chunks, dim=0)
        if encoded.shape[1] != expected_channels:
            raise ValueError(
                f"texture encoder produced {encoded.shape[1]} channels; "
                f"checkpoint expects {expected_channels}"
            )
        return encoded.to(torch.float32)

    def _encode_env_map(
        self,
        env_map: torch.Tensor | None,
        dtype: torch.dtype,
        *,
        missing_environment: Literal["black", "none"],
    ):
        if not self.config.model_config.use_env_lighting:
            return None, None, None, None
        if env_map is None:
            if missing_environment == "none":
                return None, None, None, None
            if missing_environment != "black":
                raise ValueError(
                    "missing_environment must be 'black' or 'none', "
                    f"got {missing_environment!r}"
                )
            env_map = torch.zeros((3, 256, 512), dtype=torch.float32)
        self._ensure_env_vae()
        from renderformer.data.envmap import latlong_ray_direction

        env_map = env_map[None].to(self.device, dtype=torch.float32)
        if env_map.shape != (1, 3, 256, 512):
            raise ValueError(
                "V2 environment maps must have shape (3,256,512), "
                f"got {tuple(env_map.shape[1:])}"
            )
        env_hwc = env_map.permute(0, 2, 3, 1)
        clamped = env_hwc.clamp(0.0, 1.0)
        log_env = torch.log10(env_hwc + 1.0)
        log_max = log_env.amax(dim=(1, 2, 3), keepdim=True)
        normalized = log_env / (log_max + 1e-8)
        env_images = torch.cat(
            [clamped.permute(0, 3, 1, 2), normalized.permute(0, 3, 1, 2)],
            dim=0,
        ) * 2.0 - 1.0
        if self._env_vae_is_video:
            env_images = env_images[:, :, None]
        vae_dtype = next(self._env_vae.parameters()).dtype
        with torch.inference_mode():
            latent = self._env_vae.encode(env_images.to(vae_dtype)).latent_dist.mean
        if self._env_vae_is_video:
            latent = latent[:, :, 0]
        env_latent_ldr, env_latent_hdr = latent.chunk(2, dim=0)
        ray_direction = latlong_ray_direction(
            env_map.shape[2], env_map.shape[3], device=self.device
        )[None]
        return (
            env_latent_ldr.to(dtype),
            env_latent_hdr.to(dtype),
            log_max.flatten().to(dtype),
            ray_direction.to(dtype),
        )

    def _prepare_volumes(
        self,
        volume: dict[str, torch.Tensor] | None,
        dtype: torch.dtype,
        *,
        missing_volume: Literal["masked", "legacy_zero", "none"],
        preserve_legacy_length: bool = False,
    ):
        if volume is None and missing_volume == "none":
            return (None, None, None, None, None, None, None)
        if missing_volume not in {"masked", "legacy_zero", "none"}:
            raise ValueError(f"invalid missing-volume behavior: {missing_volume}")
        length = self.config.runtime.volume_padding_length
        shapes = {
            "density": (64,),
            "position": (3,),
            "rotation": (3,),
            "scale": (3,),
            "scattering": (3,),
            "absorption": (3,),
        }
        count = (
            1
            if volume is None and missing_volume == "legacy_zero"
            else 0
            if volume is None
            else self._validate_volume_input(volume)
        )
        if count > length:
            raise ValueError(
                f"scene has {count} volumes but volume_padding_length is {length}"
            )
        output_length = count if preserve_legacy_length else length
        prepared = {}
        for name, trailing_shape in shapes.items():
            source = (
                torch.zeros((count, *trailing_shape), dtype=torch.float32)
                if volume is None
                else volume[name].to(torch.float32)
            )
            if source.shape != (count, *trailing_shape):
                raise ValueError(f"invalid volume {name} shape: {tuple(source.shape)}")
            padding = torch.zeros(
                (output_length - count, *trailing_shape), dtype=source.dtype
            )
            prepared[name] = torch.cat([source, padding], dim=0)[None].to(self.device)
        mask = torch.zeros((1, output_length), dtype=torch.bool, device=self.device)
        mask[:, :count] = True
        position_dtype = dtype if preserve_legacy_length else torch.float32
        return (
            prepared["density"].to(dtype),
            prepared["position"].to(position_dtype),
            prepared["rotation"].to(dtype),
            prepared["scale"].to(dtype),
            prepared["scattering"].to(dtype),
            prepared["absorption"].to(dtype),
            mask,
        )

    def _validate_volume_input(
        self,
        volume: dict[str, torch.Tensor] | None,
    ) -> int:
        if volume is None:
            return 0
        shapes = {
            "density": (64,),
            "position": (3,),
            "rotation": (3,),
            "scale": (3,),
            "scattering": (3,),
            "absorption": (3,),
        }
        missing = sorted(set(shapes) - set(volume))
        if missing:
            raise ValueError(f"volume input is missing fields: {missing}")
        count = int(volume["density"].shape[0])
        for name, trailing_shape in shapes.items():
            expected = (count, *trailing_shape)
            if tuple(volume[name].shape) != expected:
                raise ValueError(
                    f"invalid volume {name} shape: {tuple(volume[name].shape)}; "
                    f"expected {expected}"
                )
        if count > self.config.runtime.volume_padding_length:
            raise ValueError(
                f"scene has {count} volumes but volume_padding_length is "
                f"{self.config.runtime.volume_padding_length}"
            )
        return count

    @staticmethod
    def _require_packed_sequence_backend(enabled: bool) -> None:
        if not enabled:
            return
        from renderformer.models.attention import ATTN

        if ATTN != "flash_attn":
            raise RuntimeError(
                "V2 packed-sequence inference requires flash-attn; "
                "unset USE_SDPA/ATTN_IMPL=sdpa before importing RenderFormer, "
                "or explicitly use_packed_sequence=False"
            )

    @classmethod
    def resolve_view_batch_size(
        cls,
        *,
        resolution: int,
        num_views: int,
        first_view_only: bool = False,
        view_batch_size: int | None = None,
    ) -> int:
        """Resolve the maximum number of camera views in one V2 model call.

        The unchunked path is retained below 2048 so existing 512 inference is
        unchanged. At 2048 and above, automatic mode renders one view per call.
        This avoids PyTorch's 32-bit indexing limit in high-resolution decoder
        upsampling while preserving every input view and its original order.
        """

        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if num_views <= 0:
            raise ValueError("num_views must be positive")
        if view_batch_size is not None:
            if isinstance(view_batch_size, bool) or not isinstance(
                view_batch_size, int
            ):
                raise TypeError("view_batch_size must be an int or None")
            if view_batch_size <= 0:
                raise ValueError("view_batch_size must be positive")

        rendered_views = 1 if first_view_only else num_views
        if view_batch_size is not None:
            return min(view_batch_size, rendered_views)
        if resolution >= cls.AUTO_SINGLE_VIEW_MIN_RESOLUTION:
            return 1
        return rendered_views

    def _render_v2_scene(
        self,
        scene: SceneData,
        *,
        resolution: int,
        dtype: torch.dtype,
        first_view_only: bool,
        execution_profile: Literal[
            "release", "checkpoint", "legacy_v2", "legacy_template_v2"
        ],
        view_transformer_tf32: bool | None,
        use_packed_sequence: bool | None,
        view_batch_size: int | None = None,
    ) -> RenderOutput:
        execution_settings = self.execution_settings(
            execution_profile,
            view_transformer_tf32=view_transformer_tf32,
            use_packed_sequence=use_packed_sequence,
        )
        self._require_packed_sequence_backend(
            bool(execution_settings["use_packed_sequence"])
        )

        triangles = scene.triangles[None].to(self.device, dtype=torch.float32)
        texture = self.encode_texture(scene.texture, dtype=dtype)[None]
        vn = scene.vn[None].to(self.device, dtype=torch.float32)
        c2w = scene.c2w[None].to(self.device, dtype=torch.float32)
        fov = scene.fov[None].to(self.device, dtype=torch.float32)
        if first_view_only:
            c2w = c2w[:, :1]
            fov = fov[:, :1]
        mask = torch.ones(
            (1, scene.num_triangles), device=self.device, dtype=torch.bool
        )
        if (
            self.config.model_config.separate_light_strength
            and scene.input_metadata.get("light_strength") == "missing"
        ):
            raise ValueError(
                "this checkpoint requires light_strength, but the pre-encoded H5 "
                "does not contain it"
            )
        source_light_strength = (
            torch.zeros((scene.num_triangles, 3), dtype=torch.float32)
            if scene.light_strength is None
            else scene.light_strength
        )
        light_strength = source_light_strength[None].to(
            self.device, dtype=torch.float32
        ).clamp(min=0.0)

        if execution_settings["preprocessing_dtype"] == "render":
            texture = texture.to(dtype)
            light_strength = light_strength.to(dtype)

        if self.config.runtime.emission_area_encoding:
            areas = compute_triangle_area(triangles)
            area_shape = (1, scene.num_triangles, 1) + (1,) * (texture.ndim - 3)
            texture[:, :, -3:] *= areas.reshape(area_shape)
        if not self.config.runtime.learn_ldr:
            light_strength = torch.log10(light_strength + 1.0)
        if self.config.model_config.log_roughness_encoding:
            log_bias = 0.05
            compensation = math.log10(0.5 + log_bias)
            texture[:, :, 6] = torch.log10(texture[:, :, 6] + log_bias) - compensation
        if self.config.model_config.texture_encode_patch_size == 1 and texture.ndim == 5:
            texture = texture[..., 0, 0]

        env_values = self._encode_env_map(
            scene.env_map,
            dtype,
            missing_environment=execution_settings["missing_environment"],
        )
        volume_values = self._prepare_volumes(
            scene.volume,
            dtype,
            missing_volume=execution_settings["missing_volume"],
            preserve_legacy_length=(
                execution_settings["volume_length_policy"] == "explicit"
            ),
        )
        geometry_is_float32 = execution_settings["geometry_dtype"] == "float32"
        model_triangles = triangles if geometry_is_float32 else triangles.to(dtype)
        model_vn = vn if geometry_is_float32 else vn.to(dtype)
        texture = texture.to(dtype)
        light_strength = light_strength.to(dtype)
        mvps = torch.zeros(
            (1, 16),
            device=self.device,
            dtype=torch.float32 if geometry_is_float32 else dtype,
        )
        effective_view_batch_size = self.resolve_view_batch_size(
            resolution=resolution,
            num_views=scene.num_views,
            first_view_only=first_view_only,
            view_batch_size=view_batch_size,
        )
        raw_chunks = []
        for view_start in range(0, c2w.shape[1], effective_view_batch_size):
            view_stop = min(view_start + effective_view_batch_size, c2w.shape[1])
            chunk_c2w = c2w[:, view_start:view_stop]
            chunk_fov = fov[:, view_start:view_stop]
            view_triangles, rays_o, rays_d = self._camera_inputs(
                triangles, chunk_c2w, chunk_fov, resolution
            )
            with torch.inference_mode(), self._autocast(dtype):
                raw_chunks.append(
                    self.model(
                        model_triangles.reshape(1, -1, 9),
                        texture,
                        mvps,
                        mask,
                        model_vn.reshape(1, -1, 9),
                        light_strength=light_strength,
                        env_latent_ldr=env_values[0],
                        env_latent_hdr=env_values[1],
                        env_lighting_strength=env_values[2],
                        env_ray_dir=env_values[3],
                        rays_o=rays_o,
                        rays_d=rays_d,
                        tri_vpos_view_tf=view_triangles.reshape(
                            1, chunk_c2w.shape[1], -1, 9
                        ),
                        tf32_view_tf=execution_settings["view_transformer_tf32"],
                        use_packed_sequence=execution_settings["use_packed_sequence"],
                        volume_density=volume_values[0],
                        volume_position=volume_values[1],
                        volume_rotation=volume_values[2],
                        volume_scale=volume_values[3],
                        volume_scattering=volume_values[4],
                        volume_absorption=volume_values[5],
                        volume_mask=volume_values[6],
                    )
                )
        raw = raw_chunks[0] if len(raw_chunks) == 1 else torch.cat(raw_chunks, dim=1)
        hdr = raw.permute(0, 1, 3, 4, 2)
        if not self.config.runtime.learn_ldr:
            hdr = torch.pow(10.0, hdr) - 1.0
        return RenderOutput(raw=raw, hdr=hdr)

    def render_scene(
        self,
        scene: SceneData,
        *,
        resolution: int = 512,
        precision: str = "fp16",
        first_view_only: bool = False,
        view_batch_size: int | None = None,
        execution_profile: Literal[
            "release", "checkpoint", "legacy_v2", "legacy_template_v2"
        ] = "release",
        view_transformer_tf32: bool | None = None,
        use_packed_sequence: bool | None = None,
    ) -> RenderOutput:
        dtype = _precision_dtype(precision, self.device)
        execution_settings = self.execution_settings(
            execution_profile,
            view_transformer_tf32=view_transformer_tf32,
            use_packed_sequence=use_packed_sequence,
        )
        with self._cuda_tf32_backend(
            bool(execution_settings["view_transformer_tf32"])
        ):
            return self._render_v2_scene(
                scene,
                resolution=resolution,
                dtype=dtype,
                first_view_only=first_view_only,
                view_batch_size=view_batch_size,
                execution_profile=execution_profile,
                view_transformer_tf32=view_transformer_tf32,
                use_packed_sequence=use_packed_sequence,
            )

    @property
    def environment_encoder(self):
        return self._env_vae

    @property
    def texture_encoder(self):
        return self._texture_vae

    def execution_settings(
        self,
        profile: Literal[
            "release", "checkpoint", "legacy_v2", "legacy_template_v2"
        ] = "release",
        *,
        view_transformer_tf32: bool | None = None,
        use_packed_sequence: bool | None = None,
    ) -> dict[str, bool | str]:
        """Resolve the auditable V2 execution switches for one render."""

        for name, value in (
            ("view_transformer_tf32", view_transformer_tf32),
            ("use_packed_sequence", use_packed_sequence),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool or None")

        if profile == "release":
            settings: dict[str, bool | str] = {
                "view_transformer_tf32": True,
                "use_packed_sequence": False,
                "missing_environment": "black",
                "missing_volume": "masked",
                "preprocessing_dtype": "float32",
                "geometry_dtype": "render",
                "volume_length_policy": "padded",
            }
        elif profile == "checkpoint":
            settings: dict[str, bool | str] = {
                "view_transformer_tf32": self.config.runtime.view_transformer_tf32,
                "use_packed_sequence": self.config.runtime.use_packed_sequence,
                "missing_environment": "black",
                "missing_volume": "masked",
                "preprocessing_dtype": "float32",
                "geometry_dtype": "render",
                "volume_length_policy": "padded",
            }
        elif profile == "legacy_v2":
            settings = {
                "view_transformer_tf32": True,
                "use_packed_sequence": False,
                "missing_environment": "none",
                "missing_volume": "legacy_zero",
                "preprocessing_dtype": "float32",
                "geometry_dtype": "float32",
                "volume_length_policy": "explicit",
            }
        elif profile == "legacy_template_v2":
            settings = {
                "view_transformer_tf32": True,
                "use_packed_sequence": False,
                "missing_environment": "none",
                "missing_volume": "none",
                "preprocessing_dtype": "render",
                "geometry_dtype": "float32",
                "volume_length_policy": "padded",
            }
        else:
            raise ValueError(
                "profile must be 'release', 'checkpoint', 'legacy_v2', or "
                "'legacy_template_v2', "
                f"got {profile!r}"
            )
        if view_transformer_tf32 is not None:
            settings["view_transformer_tf32"] = view_transformer_tf32
        if use_packed_sequence is not None:
            settings["use_packed_sequence"] = use_packed_sequence
        backend_tf32 = bool(settings["view_transformer_tf32"]) and (
            self.device.type == "cuda"
        )
        settings["cuda_matmul_allow_tf32"] = backend_tf32
        settings["cudnn_allow_tf32"] = backend_tf32
        return settings

    def validate_scene(
        self,
        scene: SceneData,
        *,
        execution_profile: Literal[
            "release", "checkpoint", "legacy_v2", "legacy_template_v2"
        ] = "release",
        view_transformer_tf32: bool | None = None,
        use_packed_sequence: bool | None = None,
    ) -> dict[str, bool | str | int]:
        """Validate an H5 scene against this checkpoint without model execution."""

        settings = self.execution_settings(
            execution_profile,
            view_transformer_tf32=view_transformer_tf32,
            use_packed_sequence=use_packed_sequence,
        )
        self._require_packed_sequence_backend(
            bool(settings["use_packed_sequence"])
        )
        preencoded = self._validate_texture_input(scene.texture)
        if (
            self.config.model_config.separate_light_strength
            and scene.input_metadata.get("light_strength") == "missing"
        ):
            raise ValueError(
                "this checkpoint requires light_strength, but the pre-encoded H5 "
                "does not contain it"
            )
        if self.config.model_config.use_env_lighting:
            if (
                not self.config.model_config.envmap_vae_model_id
                and self._env_vae is None
            ):
                raise ValueError(
                    "checkpoint enables environment lighting without "
                    "envmap_vae_model_id or an injected environment_encoder"
                )
            if scene.env_map is not None and tuple(scene.env_map.shape) != (3, 256, 512):
                raise ValueError(
                    "V2 environment maps must have shape (3,256,512), "
                    f"got {tuple(scene.env_map.shape)}"
                )
        volume_count = self._validate_volume_input(scene.volume)
        return {
            **settings,
            "texture_input": "preencoded" if preencoded else "raw",
            "volume_count": volume_count,
        }

    @property
    def component_info(self) -> dict[str, dict]:
        """Serializable provenance for loaded and lazily configured components."""

        records = {name: dict(value) for name, value in self._component_records.items()}
        if "environment_encoder" not in records and self.config.model_config.use_env_lighting:
            model_id = self.config.model_config.envmap_vae_model_id
            records["environment_encoder"] = {
                "loaded": False,
                "source": "pretrained",
                "model_id": model_id,
                "revision": self._component_revisions.get(model_id),
                "subfolder": "vae",
                "dtype": self._dtype_name(self._environment_encoder_dtype),
                "is_video": model_id == "Qwen/Qwen-Image",
            }
        if "texture_encoder" not in records and self.config.model_config.texture_channels in {
            64,
            80,
            128,
            160,
        }:
            model_id, is_video = self._texture_encoder_spec()
            records["texture_encoder"] = {
                "loaded": False,
                "source": "pretrained",
                "model_id": model_id,
                "revision": self._component_revisions.get(model_id),
                "subfolder": "vae" if is_video else None,
                "dtype": self._dtype_name(self._texture_encoder_dtype),
                "is_video": is_video,
            }
        return records

    def __call__(
        self,
        scene: SceneData | str | Path,
        *,
        resolution: int = 512,
        precision: str = "fp16",
        first_view_only: bool = False,
        view_batch_size: int | None = None,
        execution_profile: Literal[
            "release", "checkpoint", "legacy_v2", "legacy_template_v2"
        ] = "release",
        view_transformer_tf32: bool | None = None,
        use_packed_sequence: bool | None = None,
    ) -> RenderOutput:
        if not isinstance(scene, SceneData):
            scene = self.load_h5(scene)
        return self.render_scene(
            scene,
            resolution=resolution,
            precision=precision,
            first_view_only=first_view_only,
            view_batch_size=view_batch_size,
            execution_profile=execution_profile,
            view_transformer_tf32=view_transformer_tf32,
            use_packed_sequence=use_packed_sequence,
        )
