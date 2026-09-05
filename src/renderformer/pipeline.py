"""Compatibility imports for the legacy inference module.

New code should import pipelines from :mod:`renderformer` or
:mod:`renderformer.pipelines`.
"""

from renderformer.pipelines import (
    RenderFormerPipeline,
    RenderFormerRenderingPipeline,
    RenderFormerV2Pipeline,
    RenderOutput,
)
from renderformer.pipelines.base import (
    _precision_dtype,
    _RenderFormerPipelineBase,
)
from renderformer.pipelines.rf1 import _log10_v1_emission_channels_

__all__ = [
    "RenderFormerPipeline",
    "RenderFormerRenderingPipeline",
    "RenderFormerV2Pipeline",
    "RenderOutput",
]
