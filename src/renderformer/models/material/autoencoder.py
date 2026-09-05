"""Material latent autoencoder used by the RenderFormer V2 data path.

The module names intentionally match the historical ``VAE`` checkpoint. In
particular, the combined model keeps a flat state dict (``encoder_initial.*``,
``latent_proj.*``, ``decoder_proj.*``, ...), so checkpoints containing a
``model_state_dict`` can be loaded without an architecture-specific rename.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from renderformer.models.material.hub import ComponentModelHubMixin


_ENCODER_PREFIXES = (
    "encoder_initial.",
    "encoder_mid.",
    "encoder_residual.",
    "encoder_final.",
    "latent_proj.",
)
_DECODER_PREFIXES = (
    "decoder_proj.",
    "decoder_initial.",
    "decoder_residual.",
    "decoder_final.",
)


def _validate_architecture(
    *,
    latent_dim: int,
    image_channels: int,
    image_size: int,
) -> None:
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if image_channels <= 0:
        raise ValueError(f"image_channels must be positive, got {image_channels}")
    if image_size != 256:
        raise ValueError(
            "The historical material architecture only supports "
            f"image_size=256, got {image_size}"
        )


class MaterialResidualBlock(nn.Module):
    """The GroupNorm/SiLU residual block used by the historical model."""

    def __init__(self, channels: int, num_groups: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.silu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        out = out + residual
        return F.silu(out, inplace=True)


def _install_encoder(
    module: nn.Module,
    *,
    latent_dim: int,
    image_channels: int,
) -> None:
    module.encoder_initial = nn.Sequential(  # type: ignore[attr-defined]
        nn.Conv2d(image_channels, 1024, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 1024),
        nn.SiLU(inplace=True),
    )
    module.encoder_mid = nn.Sequential(  # type: ignore[attr-defined]
        nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 512),
        nn.SiLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
    )
    module.encoder_residual = nn.Sequential(  # type: ignore[attr-defined]
        *(MaterialResidualBlock(256, num_groups=32) for _ in range(8))
    )
    module.encoder_final = nn.Sequential(  # type: ignore[attr-defined]
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
    )
    module.latent_proj = nn.Linear(256, latent_dim)  # type: ignore[attr-defined]


def _install_decoder(
    module: nn.Module,
    *,
    latent_dim: int,
    image_channels: int,
) -> None:
    module.decoder_proj = nn.Linear(latent_dim, 256)  # type: ignore[attr-defined]
    module.decoder_initial = nn.Sequential(  # type: ignore[attr-defined]
        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=1, padding=0),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 256),
        nn.SiLU(inplace=True),
    )
    module.decoder_residual = nn.Sequential(  # type: ignore[attr-defined]
        *(MaterialResidualBlock(256, num_groups=32) for _ in range(8))
    )
    module.decoder_final = nn.Sequential(  # type: ignore[attr-defined]
        nn.ConvTranspose2d(256, 512, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 512),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(512, 1024, 3, 2, 1, output_padding=1),
        nn.GroupNorm(32, 1024),
        nn.SiLU(inplace=True),
        nn.ConvTranspose2d(1024, image_channels, 3, 2, 1, output_padding=1),
        nn.Sigmoid(),
    )


def _encode(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    h = module.encoder_initial(x)  # type: ignore[attr-defined]
    h = module.encoder_mid(h)  # type: ignore[attr-defined]
    h = module.encoder_residual(h)  # type: ignore[attr-defined]
    h = module.encoder_final(h)  # type: ignore[attr-defined]
    h = h.view(h.size(0), -1)
    latent = module.latent_proj(h)  # type: ignore[attr-defined]
    return torch.tanh(latent)


def _decode(module: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    h = module.decoder_proj(latent)  # type: ignore[attr-defined]
    h = h.view(h.size(0), 256, 1, 1)
    h = module.decoder_initial(h)  # type: ignore[attr-defined]
    h = module.decoder_residual(h)  # type: ignore[attr-defined]
    return module.decoder_final(h)  # type: ignore[attr-defined]


class MaterialEncoder(nn.Module):
    """Encode a 3x256x256 network-space material image to a 9-D latent."""

    def __init__(
        self,
        latent_dim: int = 9,
        image_channels: int = 3,
        image_size: int = 256,
    ) -> None:
        _validate_architecture(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
        )
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        _install_encoder(self, latent_dim=latent_dim, image_channels=image_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _encode(self, x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

    def encode_hdr(self, x: torch.Tensor) -> torch.Tensor:
        """Encode alpha-composited, nonnegative linear RGB with ``log10(1+x)``."""

        from renderformer.data.material_preprocessing import encode_material_hdr

        return self.encode(encode_material_hdr(x))


class MaterialDecoder(nn.Module):
    """Decode a 9-D latent to a sigmoid-bounded 3x256x256 network-space image."""

    def __init__(
        self,
        latent_dim: int = 9,
        image_channels: int = 3,
        image_size: int = 256,
    ) -> None:
        _validate_architecture(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
        )
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        _install_decoder(self, latent_dim=latent_dim, image_channels=image_channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return _decode(self, latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self(latent)

    def decode_hdr(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent and invert network space with ``10**x - 1``."""

        from renderformer.data.material_preprocessing import decode_material_hdr

        return decode_material_hdr(self.decode(latent))


class MaterialAutoencoder(nn.Module, ComponentModelHubMixin):
    """Historical RenderFormer material autoencoder with checkpoint-stable keys.

    ``forward`` returns ``(decoded_network_space, latent)`` in the same order
    as the original implementation. This is a deterministic autoencoder
    despite the historical checkpoint being described as a VAE. ``encode``
    and ``decode`` retain the raw network-space contract; ``encode_hdr`` and
    ``decode_hdr`` apply the audited release ``log10(1+x)`` transform and its
    inverse without adding parameters or changing state-dict keys.

    The Hugging Face mixin stores the constructor arguments in ``config.json``
    and the flat checkpoint-compatible state dict in ``model.safetensors``.
    The standalone legacy ``best_model.pt`` helpers below remain available for
    historical checkpoints that wrap weights under ``model_state_dict``.
    """

    def __init__(
        self,
        latent_dim: int = 9,
        image_channels: int = 3,
        image_size: int = 256,
    ) -> None:
        _validate_architecture(
            latent_dim=latent_dim,
            image_channels=image_channels,
            image_size=image_size,
        )
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        _install_encoder(self, latent_dim=latent_dim, image_channels=image_channels)
        _install_decoder(self, latent_dim=latent_dim, image_channels=image_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return _encode(self, x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return _decode(self, latent)

    def encode_hdr(self, x: torch.Tensor) -> torch.Tensor:
        """Encode alpha-composited, nonnegative linear RGB material renders."""

        from renderformer.data.material_preprocessing import encode_material_hdr

        return self.encode(encode_material_hdr(x))

    def decode_hdr(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent to nonnegative linear RGB with ``10**x - 1``."""

        from renderformer.data.material_preprocessing import decode_material_hdr

        return decode_material_hdr(self.decode(latent))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        decoded_network_space = self.decode(latent)
        return decoded_network_space, latent

    def save_pretrained(self, save_directory: str | Path, **kwargs):
        """Save weights, architecture, and the fixed physical-input contract."""

        destination = Path(save_directory).expanduser()
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(
                f"model component destination is not a directory: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "preprocessor_config.json").write_text(
            json.dumps({"mode": "log10_1p"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return super().save_pretrained(destination, **kwargs)


CheckpointSource = Mapping[str, Any] | str | PathLike[str]
MaterialModule = TypeVar(
    "MaterialModule",
    MaterialAutoencoder,
    MaterialEncoder,
    MaterialDecoder,
)


def material_state_dict_from_checkpoint(
    checkpoint: CheckpointSource,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, torch.Tensor]:
    """Return the strict historical ``model_state_dict`` payload.

    Path inputs are deserialized with ``weights_only=True``. A preloaded
    mapping can be supplied when an application has its own checkpoint I/O.
    Raw state dictionaries are deliberately rejected: historical release
    checkpoints wrap the weights under ``model_state_dict``.
    """

    loaded: Mapping[str, Any]
    if isinstance(checkpoint, Mapping):
        loaded = checkpoint
    else:
        deserialized = torch.load(
            Path(checkpoint),
            map_location=map_location,
            weights_only=True,
        )
        if not isinstance(deserialized, Mapping):
            raise TypeError("Material checkpoint must deserialize to a mapping")
        loaded = deserialized

    if "model_state_dict" not in loaded:
        raise KeyError("Material checkpoint is missing 'model_state_dict'")
    state_dict = loaded["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("Material checkpoint 'model_state_dict' must be a mapping")

    invalid_keys = [key for key in state_dict if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError("Material checkpoint state-dict keys must be strings")
    invalid_values = [key for key, value in state_dict.items() if not torch.is_tensor(value)]
    if invalid_values:
        raise TypeError(
            "Material checkpoint state-dict values must be tensors; invalid keys: "
            + ", ".join(sorted(invalid_values))
        )
    return state_dict


def _load_component_checkpoint(
    module: MaterialModule,
    checkpoint: CheckpointSource,
    *,
    owned_prefixes: tuple[str, ...],
    ignored_prefixes: tuple[str, ...],
    map_location: str | torch.device,
) -> MaterialModule:
    state_dict = material_state_dict_from_checkpoint(
        checkpoint,
        map_location=map_location,
    )
    expected_keys = tuple(module.state_dict())
    expected = set(expected_keys)

    unexpected = sorted(
        key
        for key in state_dict
        if key not in expected and not key.startswith(ignored_prefixes)
    )
    unexpected_owned = sorted(
        key
        for key in state_dict
        if key.startswith(owned_prefixes) and key not in expected
    )
    unexpected = sorted(set(unexpected + unexpected_owned))
    if unexpected:
        raise RuntimeError(
            "Unexpected keys in historical material state dict: "
            + ", ".join(unexpected)
        )

    component_state = {key: state_dict[key] for key in expected_keys if key in state_dict}
    module.load_state_dict(component_state, strict=True)
    return module


def load_material_autoencoder_checkpoint(
    model: MaterialAutoencoder,
    checkpoint: CheckpointSource,
    *,
    map_location: str | torch.device = "cpu",
) -> MaterialAutoencoder:
    """Strictly load a complete historical material checkpoint."""

    state_dict = material_state_dict_from_checkpoint(
        checkpoint,
        map_location=map_location,
    )
    model.load_state_dict(state_dict, strict=True)
    return model


def load_material_encoder_checkpoint(
    encoder: MaterialEncoder,
    checkpoint: CheckpointSource,
    *,
    map_location: str | torch.device = "cpu",
) -> MaterialEncoder:
    """Strictly load the encoder portion of a historical full checkpoint."""

    return _load_component_checkpoint(
        encoder,
        checkpoint,
        owned_prefixes=_ENCODER_PREFIXES,
        ignored_prefixes=_DECODER_PREFIXES,
        map_location=map_location,
    )


def load_material_decoder_checkpoint(
    decoder: MaterialDecoder,
    checkpoint: CheckpointSource,
    *,
    map_location: str | torch.device = "cpu",
) -> MaterialDecoder:
    """Strictly load the decoder portion of a historical full checkpoint."""

    return _load_component_checkpoint(
        decoder,
        checkpoint,
        owned_prefixes=_DECODER_PREFIXES,
        ignored_prefixes=_ENCODER_PREFIXES,
        map_location=map_location,
    )


__all__ = [
    "MaterialAutoencoder",
    "MaterialDecoder",
    "MaterialEncoder",
    "MaterialResidualBlock",
    "load_material_autoencoder_checkpoint",
    "load_material_decoder_checkpoint",
    "load_material_encoder_checkpoint",
    "material_state_dict_from_checkpoint",
]
