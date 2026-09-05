"""Training dataloaders for RenderFormer H5 datasets.

The public loaders intentionally share one configuration type so historical
Tyro/YAML options keep working across V1 and V2 training entrypoints.
"""

from .config import TriangleRenderH5DatasetConfig

__all__ = ["TriangleRenderH5DatasetConfig"]
