"""Material latent models used by RenderFormer V2."""

from renderformer.models.material.autoencoder import (
    MaterialAutoencoder,
    MaterialDecoder,
    MaterialEncoder,
    MaterialResidualBlock,
    load_material_autoencoder_checkpoint,
    load_material_decoder_checkpoint,
    load_material_encoder_checkpoint,
    material_state_dict_from_checkpoint,
)
from renderformer.models.material.mappers import (
    DiffuseSpecularToLatent,
    PrincipledBRDFToLatent,
    PrincipledBRDFToLatentWithTransmission,
)

__all__ = [
    "DiffuseSpecularToLatent",
    "MaterialAutoencoder",
    "MaterialDecoder",
    "MaterialEncoder",
    "MaterialResidualBlock",
    "PrincipledBRDFToLatent",
    "PrincipledBRDFToLatentWithTransmission",
    "load_material_autoencoder_checkpoint",
    "load_material_decoder_checkpoint",
    "load_material_encoder_checkpoint",
    "material_state_dict_from_checkpoint",
]
