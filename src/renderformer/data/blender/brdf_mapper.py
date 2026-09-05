"""Compatibility imports for BRDF-to-latent models.

New code should import these classes from :mod:`renderformer.models.material`.
"""

from renderformer.models.material import (
    DiffuseSpecularToLatent,
    PrincipledBRDFToLatent,
    PrincipledBRDFToLatentWithTransmission,
)

__all__ = [
    "DiffuseSpecularToLatent",
    "PrincipledBRDFToLatent",
    "PrincipledBRDFToLatentWithTransmission",
]
