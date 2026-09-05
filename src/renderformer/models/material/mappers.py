"""BRDF-parameter MLPs for the nine-dimensional material latent space."""

from __future__ import annotations

import torch
import torch.nn as nn

from renderformer.models.material.hub import ComponentModelHubMixin


def _install_layers(
    module: nn.Module,
    *,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    num_layers: int,
) -> None:
    if num_layers < 2:
        raise ValueError("num_layers must be at least 2 (input -> output)")

    layers = [nn.Linear(input_dim, hidden_dim)]
    layers.extend(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2))
    layers.append(nn.Linear(hidden_dim, latent_dim))
    module.layers = nn.ModuleList(layers)  # type: ignore[attr-defined]
    module.relu = nn.ReLU(inplace=True)  # type: ignore[attr-defined]
    module.tanh = nn.Tanh()  # type: ignore[attr-defined]


def _map_to_latent(module: nn.Module, parameters: torch.Tensor) -> torch.Tensor:
    x = parameters
    for layer in module.layers[:-1]:  # type: ignore[attr-defined]
        x = module.relu(layer(x))  # type: ignore[attr-defined]
    return module.tanh(module.layers[-1](x))  # type: ignore[attr-defined]


class DiffuseSpecularToLatent(nn.Module, ComponentModelHubMixin):
    """Map diffuse RGB, specular RGB, and roughness to a material latent."""

    def __init__(
        self,
        latent_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        _install_layers(
            self,
            input_dim=7,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, brdf_params: torch.Tensor) -> torch.Tensor:
        return _map_to_latent(self, brdf_params)


class PrincipledBRDFToLatent(nn.Module, ComponentModelHubMixin):
    """Map base-color RGB, metallic, and roughness to a material latent."""

    def __init__(
        self,
        latent_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        _install_layers(
            self,
            input_dim=5,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, brdf_params: torch.Tensor) -> torch.Tensor:
        return _map_to_latent(self, brdf_params)


class PrincipledBRDFToLatentWithTransmission(nn.Module, ComponentModelHubMixin):
    """Map principled BRDF parameters including IOR and transmission."""

    def __init__(
        self,
        latent_dim: int = 9,
        hidden_dim: int = 256,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        _install_layers(
            self,
            input_dim=7,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, brdf_params: torch.Tensor) -> torch.Tensor:
        return _map_to_latent(self, brdf_params)


__all__ = [
    "DiffuseSpecularToLatent",
    "PrincipledBRDFToLatent",
    "PrincipledBRDFToLatentWithTransmission",
]
