"""Optional NVIDIA DALI imports shared by the training loaders."""

from __future__ import annotations


try:
    import nvidia.dali as dali
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    dali = None
    fn = None
    DALIGenericIterator = None

    class _DummyDALITypes:
        """Allow modules with DALI type annotations to import on CPU hosts."""

        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

    types = _DummyDALITypes()


def require_dali() -> None:
    if dali is None:
        raise ImportError(
            "NVIDIA DALI is required for TriangleRenderDALIDataset. "
            "Install the CUDA-matched nvidia-dali package or use the PyTorch dataset."
        )
