"""Strict checkpoint configuration adapters for V1 and V2."""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from renderformer.models.config import RenderTransformerConfig


class CheckpointVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"


@dataclasses.dataclass(frozen=True)
class InferenceRuntimeConfig:
    """Non-architectural values needed to reproduce training-time inference."""

    version: CheckpointVersion
    turn_to_cam_coord: bool
    learn_ldr: bool
    emission_area_encoding: bool = False
    clamped_hdr: bool = False
    clamped_hdr_max: float = 9.0
    use_packed_sequence: bool = False
    volume_padding_length: int = 512
    image_key: str = "img"
    view_transformer_tf32: bool = False


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    model_config: RenderTransformerConfig
    runtime: InferenceRuntimeConfig
    source_file: Path
    raw: dict[str, Any]


# These are the fields accepted by the released flat V1 config schema. Keep the
# defaults here instead of borrowing current model defaults: several current
# defaults intentionally differ from the 2025 architecture.
_V1_DEFAULTS: dict[str, Any] = {
    "latent_dim": 768,
    "num_layers": 12,
    "num_heads": 6,
    "dim_feedforward": 768 * 4,
    "num_register_tokens": 16,
    "dropout": 0.0,
    "activation": "swiglu",
    "norm_type": "rms_norm",
    "norm_first": True,
    "view_indep_qk_norm": True,
    "qk_norm": True,
    "bias": False,
    "pe_type": "rope",
    "rope_type": "triangle",
    "rope_double_max_freq": False,
    "vertex_pe_num_freqs": 12,
    "use_vn_encoder": True,
    "vn_pe_num_freqs": 6,
    "vn_encoder_norm_type": "rms_norm",
    "texture_encode_patch_size": 32,
    "texture_channels": 13,
    "texture_encoder_norm_type": "rms_norm",
    "view_transformer_latent_dim": 768,
    "view_transformer_ffn_hidden_dim": 768 * 4,
    "view_transformer_n_heads": 6,
    "view_transformer_n_layers": 6,
    "view_transformer_include_self_attn": True,
    "view_transformer_use_swin_attn": False,
    "vdir_pe_type": "nerf",
    "vdir_num_freqs": 0,
    "patch_size": 8,
    "include_alpha": False,
    "use_dpt_decoder": True,
    "dpt_features": 128,
    "dpt_out_channels": [96, 192, 384, 768],
    "dpt_out_layers": None,
    "turn_to_cam_coord": True,
    "use_ldr": False,
}

_V1_RUNTIME_FIELDS = frozenset({"turn_to_cam_coord", "use_ldr"})

# Values that used to be implicit because the standalone V1 model did not have
# these switches. They must be explicit when instantiating the shared model.
_V1_SHARED_MODEL_VALUES: dict[str, Any] = {
    "use_view_token": False,
    "decoding_method": "view_transformer",
    "num_kv_heads": None,
    "register_token_pe": "weighted_mean",
    "view_transformer_num_kv_heads": None,
    "view_transformer_self_attn_before_cross": False,
    "custom_init": False,
    "auto_interpolate_texture": False,
    "use_env_lighting": False,
    "use_volumes": False,
    "use_perceiver": False,
    "use_triangle_ordering": False,
    "view_transformer_use_triangle_ordering": False,
    "use_local_attention": False,
    "use_flex_attention": False,
    "put_light_tokens_first": False,
    "add_summary_tokens": False,
    "separate_light_strength": False,
    "deepstack_injection_map": None,
    "deepstack_decoder_injection_map": None,
    "latent_vae_model_id": None,
    "view_transformer_use_layers": None,
}

_SYSTEM_CONFIG_FIELDS = frozenset({
    "model_config",
    "dataset_config",
    "trainer_config",
})

def detect_checkpoint_version(config_dir: str | Path) -> tuple[CheckpointVersion, Path]:
    """Detect a checkpoint by schema, never by its model ID or directory name."""

    config_dir = Path(config_dir)
    if config_dir.is_file():
        if config_dir.suffix.lower() == ".json":
            return CheckpointVersion.V1, config_dir
        if config_dir.suffix.lower() in {".yaml", ".yml"}:
            return CheckpointVersion.V2, config_dir
        raise ValueError(f"unsupported config filename: {config_dir}")

    json_path = config_dir / "config.json"
    yaml_paths = [config_dir / "config.yaml", config_dir / "config.yml"]
    present_yaml = [path for path in yaml_paths if path.is_file()]
    if json_path.is_file() and present_yaml:
        raise ValueError(
            f"ambiguous checkpoint schema in {config_dir}: both V1 JSON and V2 YAML exist"
        )
    if json_path.is_file():
        return CheckpointVersion.V1, json_path
    if len(present_yaml) == 1:
        return CheckpointVersion.V2, present_yaml[0]
    if len(present_yaml) > 1:
        raise ValueError(f"ambiguous V2 configs in {config_dir}: {present_yaml}")
    raise FileNotFoundError(
        f"checkpoint config not found in {config_dir}; expected config.json or config.yaml"
    )


def _load_v1_config(config_path: Path) -> InferenceConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"V1 config must be a JSON object: {config_path}")

    unknown = sorted(set(raw) - set(_V1_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown V1 config fields in {config_path}: {unknown}")

    values = dict(_V1_DEFAULTS)
    values.update(raw)
    model_values = {
        key: value
        for key, value in values.items()
        if key not in _V1_RUNTIME_FIELDS
    }
    model_values.update(_V1_SHARED_MODEL_VALUES)
    model_config = RenderTransformerConfig(**model_values)
    if model_config.pe_type == "rope_only_center":
        raise ValueError("V1 checkpoints require the legacy full-triangle position path")

    runtime = InferenceRuntimeConfig(
        version=CheckpointVersion.V1,
        turn_to_cam_coord=bool(values["turn_to_cam_coord"]),
        learn_ldr=bool(values["use_ldr"]),
        use_packed_sequence=False,
        view_transformer_tf32=True,
    )
    return InferenceConfig(model_config, runtime, config_path, raw)


def _load_v2_config(config_path: Path) -> InferenceConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"V2 config must be a YAML mapping: {config_path}")

    unknown = sorted(set(raw) - _SYSTEM_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown top-level V2 config fields in {config_path}: {unknown}")
    model_raw = raw.get("model_config")
    if not isinstance(model_raw, dict):
        raise ValueError(f"V2 config is missing model_config: {config_path}")
    model_values = dict(model_raw)
    valid_model_fields = {
        field.name for field in dataclasses.fields(RenderTransformerConfig)
    }
    unknown_model_fields = sorted(set(model_values) - valid_model_fields)
    if unknown_model_fields:
        raise ValueError(
            f"unknown V2 model_config fields in {config_path}: {unknown_model_fields}"
        )
    try:
        model_config = RenderTransformerConfig(**model_values)
    except TypeError as exc:
        raise ValueError(f"invalid V2 model_config in {config_path}: {exc}") from exc

    trainer_raw = raw.get("trainer_config") or {}
    dataset_raw = raw.get("dataset_config") or {}
    if not isinstance(trainer_raw, dict) or not isinstance(dataset_raw, dict):
        raise TypeError("trainer_config and dataset_config must be YAML mappings")
    runtime = InferenceRuntimeConfig(
        version=CheckpointVersion.V2,
        turn_to_cam_coord=bool(trainer_raw.get("turn_to_cam_coord", False)),
        learn_ldr=bool(trainer_raw.get("learn_ldr", False)),
        emission_area_encoding=bool(trainer_raw.get("emission_area_encoding", False)),
        clamped_hdr=bool(trainer_raw.get("clamped_hdr", False)),
        clamped_hdr_max=float(trainer_raw.get("clamped_hdr_max", 9.0)),
        use_packed_sequence=bool(model_config.use_volumes),
        volume_padding_length=int(dataset_raw.get("volume_padding_length", 512)),
        image_key=str(dataset_raw.get("img_key", "img")),
        view_transformer_tf32=bool(
            trainer_raw.get("override_view_tf32_mode", False)
        ),
    )
    return InferenceConfig(model_config, runtime, config_path, raw)


def load_inference_config(config_dir: str | Path) -> InferenceConfig:
    version, config_path = detect_checkpoint_version(config_dir)
    if version is CheckpointVersion.V1:
        return _load_v1_config(config_path)
    return _load_v2_config(config_path)
