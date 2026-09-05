"""Stable public API for RenderFormer data, training, and inference.

The implementation lives entirely under this package. Training loads the same
transformer component that the inference pipelines wrap; there is no second
inference-only model implementation.
"""

from typing import TYPE_CHECKING

from renderformer.config import (
    CheckpointVersion,
    InferenceConfig,
    InferenceRuntimeConfig,
    load_inference_config,
)
from renderformer.models.config import RenderTransformerConfig as _RenderTransformerConfig

__version__ = "0.1.0"

# Public spelling; retain the serialized/internal class name for checkpoint
# compatibility without exposing it as the preferred API name.
RenderFormerConfig = _RenderTransformerConfig

if TYPE_CHECKING:
    from renderformer.modeling import RenderFormerModel
    from renderformer.modeling import RenderFormerModel as RenderFormer
    from renderformer.pipelines import (
        RenderFormerPipeline,
        RenderFormerRenderingPipeline,
        RenderFormerV2Pipeline,
    )

__all__ = [
    "CheckpointVersion",
    "InferenceConfig",
    "InferenceRuntimeConfig",
    "RenderFormer",
    "RenderFormerConfig",
    "RenderFormerModel",
    "RenderFormerPipeline",
    "RenderFormerRenderingPipeline",
    "RenderFormerV2Pipeline",
    "__version__",
    "load_inference_config",
]


def __getattr__(name):
    if name in {"RenderFormer", "RenderFormerModel"}:
        from renderformer.modeling import RenderFormerModel

        return RenderFormerModel
    if name in {
        "RenderFormerPipeline",
        "RenderFormerRenderingPipeline",
        "RenderFormerV2Pipeline",
    }:
        from renderformer.pipelines import (
            RenderFormerPipeline,
            RenderFormerRenderingPipeline,
            RenderFormerV2Pipeline,
        )

        return {
            "RenderFormerPipeline": RenderFormerPipeline,
            "RenderFormerRenderingPipeline": RenderFormerRenderingPipeline,
            "RenderFormerV2Pipeline": RenderFormerV2Pipeline,
        }[name]
    raise AttributeError(name)
