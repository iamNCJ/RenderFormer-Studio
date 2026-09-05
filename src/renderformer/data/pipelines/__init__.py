from __future__ import annotations

from typing import TYPE_CHECKING, Any

from renderformer.data.pipelines.compose_rf1 import compose_rf1_h5

if TYPE_CHECKING:
    from renderformer.data.pipelines.generate import ExportJob

__all__ = ["ExportJob", "compose_rf1_h5", "run_exports"]


def __getattr__(name: str) -> Any:
    """Load Blender-backed generation helpers only when explicitly requested."""

    if name in {"ExportJob", "run_exports"}:
        from renderformer.data.pipelines.generate import ExportJob, run_exports

        return {"ExportJob": ExportJob, "run_exports": run_exports}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
