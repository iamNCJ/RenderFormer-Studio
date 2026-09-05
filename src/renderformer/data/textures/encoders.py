from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class TextureEncoder(Protocol):
    name: str

    def encode(self, texture: np.ndarray, triangle_batch_size: int) -> np.ndarray:
        ...


@dataclass(frozen=True)
class PassthroughTextureEncoder:
    name: str = "raw"

    def encode(self, texture: np.ndarray, triangle_batch_size: int) -> np.ndarray:
        return texture


@dataclass
class QwenTextureVAEEncoder:
    config: dict[str, Any] = field(default_factory=dict)
    name: str = "qwen_vae"
    _torch: Any = field(init=False, default=None, repr=False)
    _vae: Any = field(init=False, default=None, repr=False)
    _device: str | None = field(init=False, default=None, repr=False)

    def _load_backend(self) -> None:
        if self._vae is not None:
            return
        import torch
        from diffusers import AutoencoderKLQwenImage

        torch_dtype = getattr(torch, str(self.config.get("torch_dtype", "float32")))
        device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_id = str(self.config.get("model_id", "Qwen/Qwen-Image"))
        subfolder = self.config.get("subfolder", "vae")

        self._torch = torch
        self._device = device
        self._vae = AutoencoderKLQwenImage.from_pretrained(
            model_id,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            revision=self.config.get("revision"),
            cache_dir=self.config.get("cache_dir"),
            local_files_only=bool(self.config.get("local_files_only", False)),
        ).to(device)
        self._vae.eval()

    def encode(self, texture: np.ndarray, triangle_batch_size: int) -> np.ndarray:
        if texture.ndim != 4 or texture.shape[1] < 12:
            raise ValueError(f"qwen texture encoder expects texture shape (N, >=12, H, W), got {texture.shape}")
        if texture.shape[1] == 15 and bool(self.config.get("encode_heightmap", True)):
            texture = np.concatenate(
                [texture, np.zeros((texture.shape[0], 1, *texture.shape[2:]), dtype=texture.dtype)],
                axis=1,
            )
        self._load_backend()
        torch = self._torch
        assert torch is not None
        assert self._vae is not None
        assert self._device is not None

        # The VAE may be loaded in float16 / bfloat16 to save memory; cast input
        # to whatever dtype the model uses so we don't hit "Input type (float)
        # and bias type (c10::Half) should be the same" from the conv layers.
        vae_dtype = next(self._vae.parameters()).dtype

        chunks = []
        with torch.no_grad():
            for start in range(0, texture.shape[0], triangle_batch_size):
                batch_np = texture[start : start + triangle_batch_size].astype(np.float32, copy=False)
                batch = torch.from_numpy(batch_np).to(self._device).to(vae_dtype)
                groups = []
                for channel_start in (0, 3, 6, 9):
                    component = batch[:, channel_start : channel_start + 3, None]
                    groups.append(self._vae.encode(component).latent_dist.mean[:, :, 0])
                if batch.shape[1] >= 16 and bool(self.config.get("encode_heightmap", True)):
                    heightmap = batch[:, 15:16].repeat(1, 3, 1, 1)
                    groups.append(self._vae.encode(heightmap[:, :, None]).latent_dist.mean[:, :, 0])
                chunks.append(torch.cat(groups, dim=1).float().detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float16)


@dataclass
class DCAETextureEncoder:
    config: dict[str, Any] = field(default_factory=dict)
    name: str = "dc_ae"
    _torch: Any = field(init=False, default=None, repr=False)
    _vae: Any = field(init=False, default=None, repr=False)
    _device: str | None = field(init=False, default=None, repr=False)

    def _load_backend(self) -> None:
        if self._vae is not None:
            return
        import torch
        from diffusers import AutoencoderDC

        torch_dtype = getattr(torch, str(self.config.get("torch_dtype", "float32")))
        device = str(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model_id = str(self.config.get("model_id", "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"))

        self._torch = torch
        self._device = device
        self._vae = AutoencoderDC.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=self.config.get("revision"),
            cache_dir=self.config.get("cache_dir"),
            local_files_only=bool(self.config.get("local_files_only", False)),
        ).to(device)
        self._vae.eval()

    def encode(self, texture: np.ndarray, triangle_batch_size: int) -> np.ndarray:
        if texture.ndim != 4 or texture.shape[1] < 12:
            raise ValueError(f"dc-ae texture encoder expects texture shape (N, >=12, H, W), got {texture.shape}")
        if texture.shape[1] == 15 and bool(self.config.get("encode_heightmap", True)):
            texture = np.concatenate(
                [texture, np.zeros((texture.shape[0], 1, *texture.shape[2:]), dtype=texture.dtype)],
                axis=1,
            )
        self._load_backend()
        torch = self._torch
        assert torch is not None
        assert self._vae is not None
        assert self._device is not None

        vae_dtype = next(self._vae.parameters()).dtype

        chunks = []
        with torch.no_grad():
            for start in range(0, texture.shape[0], triangle_batch_size):
                batch_np = texture[start : start + triangle_batch_size].astype(np.float32, copy=False)
                batch = torch.from_numpy(batch_np).to(self._device).to(vae_dtype)
                groups = []
                for channel_start in (0, 3, 6, 9):
                    groups.append(self._vae.encode(batch[:, channel_start : channel_start + 3]).latent)
                if batch.shape[1] >= 16 and bool(self.config.get("encode_heightmap", True)):
                    heightmap = batch[:, 15:16].repeat(1, 3, 1, 1)
                    groups.append(self._vae.encode(heightmap).latent)
                chunks.append(torch.cat(groups, dim=1).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float16)


def build_texture_encoder(config: dict | None) -> TextureEncoder:
    if config is None:
        return PassthroughTextureEncoder()

    encoder_type = str(config.get("type", "raw")).lower()
    if encoder_type in {"raw", "passthrough"}:
        return PassthroughTextureEncoder()
    if encoder_type in {"qwen", "qwen_vae", "qwen_texture_vae", "qwen-vae"}:
        return QwenTextureVAEEncoder(config=dict(config))
    if encoder_type in {"dcae", "dc_ae", "dc-ae"}:
        return DCAETextureEncoder(config=dict(config))

    raise ValueError(f"Unsupported texture encoder: {encoder_type}")
