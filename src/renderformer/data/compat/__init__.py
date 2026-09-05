"""Compatibility adapters for released RenderFormer data formats."""

from .rf1_scene import LegacyRF1SceneConfig, load_legacy_rf1_scene, prepare_legacy_rf1_scene

__all__ = [
    "LegacyRF1SceneConfig",
    "load_legacy_rf1_scene",
    "prepare_legacy_rf1_scene",
]
