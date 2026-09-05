import torch
import torch.nn as nn


class VolumeEncoder(nn.Module):
    """
    Encode volume tokens (voxels) into embeddings.

    Input per voxel:
        - density: [64] (flattened 4x4x4)
        - rotation: [3] (radians)
        - scale: [3]
        - scattering: [3] (RGB)
        - absorption: [3] (RGB)
        - position: [3] (used by RoPE, not embedded here)

    Output:
        - embedding: [D]
        - center: [3] (the unmodified position)
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        density_input_dim: int = 64,
        use_density_mlp: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Density encoder: [64] -> [D]
        if use_density_mlp:
            self.density_encoder = nn.Sequential(
                nn.Linear(density_input_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.density_encoder = nn.Linear(density_input_dim, latent_dim)

        # Rotation encoder: [3] -> [D]
        # A direct MLP is used instead of NeRF-style positional encoding.
        self.rotation_encoder = nn.Sequential(
            nn.Linear(3, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, latent_dim),
        )

        # Scale encoder: [3] -> [D]
        self.scale_encoder = nn.Sequential(
            nn.Linear(3, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, latent_dim),
        )

        # Scattering encoder: [3] -> [D]
        self.scattering_encoder = nn.Sequential(
            nn.Linear(3, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, latent_dim),
        )

        # Absorption encoder: [3] -> [D]
        self.absorption_encoder = nn.Sequential(
            nn.Linear(3, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, latent_dim),
        )

        # Learnable volume token, analogous to the triangle token.
        self.vol_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

    def forward(
        self,
        density: torch.Tensor,      # [B, N_vol, 64]
        position: torch.Tensor,     # [B, N_vol, 3]
        rotation: torch.Tensor,     # [B, N_vol, 3]
        scale: torch.Tensor,        # [B, N_vol, 3]
        scattering: torch.Tensor,   # [B, N_vol, 3]
        absorption: torch.Tensor,   # [B, N_vol, 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embeddings: [B, N_vol, D]
            centers: [B, N_vol, 3] (the unmodified position)
        """
        B, N_vol = density.shape[:2]

        # Encode each component
        density_emb = self.density_encoder(density)           # [B, N_vol, D]
        rotation_emb = self.rotation_encoder(rotation)        # [B, N_vol, D]
        scale_emb = self.scale_encoder(scale)                 # [B, N_vol, D]
        scattering_emb = self.scattering_encoder(scattering)  # [B, N_vol, D]
        absorption_emb = self.absorption_encoder(absorption)  # [B, N_vol, D]

        # Sum all components + learnable token
        vol_token = self.vol_token.expand(B, N_vol, -1)
        embeddings = (
            density_emb +
            rotation_emb +
            scale_emb +
            scattering_emb +
            absorption_emb +
            vol_token
        )

        # RoPE consumes the original volume centers.
        centers = position

        return embeddings, centers
