"""RenderFormer data loading and generation package.

The package root stays import-light. Inference scene helpers are resolved
lazily so importing data-generation schemas does not eagerly import PyTorch.
"""

from typing import TYPE_CHECKING


__version__ = "0.1.0"
__all__ = ["SceneData", "collect_h5_inputs", "load_h5_scene"]


if TYPE_CHECKING:
    from renderformer.data.scene import SceneData, collect_h5_inputs, load_h5_scene


def __getattr__(name: str):
    if name in __all__:
        from renderformer.data.scene import (
            SceneData,
            collect_h5_inputs,
            load_h5_scene,
        )

        return {
            "SceneData": SceneData,
            "collect_h5_inputs": collect_h5_inputs,
            "load_h5_scene": load_h5_scene,
        }[name]
    raise AttributeError(name)
