from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Dict, Literal, List, Optional


def _normalize_deepstack_map(
    value: Optional[Mapping[object, object]],
    *,
    field_name: str,
    layer_count: int,
) -> Optional[Dict[int, int]]:
    """Normalize JSON string keys and reject maps that cannot execute."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping or None")

    normalized: Dict[int, int] = {}
    for raw_layer, raw_feature in value.items():
        if isinstance(raw_layer, bool):
            raise TypeError(f"{field_name} layer keys must be integers")
        if isinstance(raw_layer, int):
            layer = raw_layer
        elif isinstance(raw_layer, str) and re.fullmatch(r"0|[1-9]\d*", raw_layer):
            layer = int(raw_layer)
        else:
            raise TypeError(
                f"{field_name} layer keys must be non-negative integers, "
                f"got {raw_layer!r}"
            )
        if layer < 0 or layer >= layer_count:
            raise ValueError(
                f"{field_name} layer {layer} is outside [0, {layer_count})"
            )
        if isinstance(raw_feature, bool) or not isinstance(raw_feature, int):
            raise TypeError(f"{field_name} feature indices must be integers")
        if raw_feature not in {0, 1, 2, 3}:
            raise ValueError(
                f"{field_name} feature {raw_feature} is outside the supported "
                "set {0, 1, 2, 3}"
            )
        if layer in normalized:
            raise ValueError(f"{field_name} contains duplicate layer {layer}")
        normalized[layer] = raw_feature
    return normalized


@dataclass(frozen=True)
class RenderTransformerConfig:
    latent_dim: int = 384
    """The latent dimension of the transformer."""
    num_layers: int = 16
    """The number of layers in the transformer."""
    num_heads: int = 8
    """The number of heads in the transformer."""
    num_kv_heads: Optional[int] = None
    """The number of key and value heads in the transformer (for GQA/MQA). If None, set to num_heads (MHA)."""
    dim_feedforward: int = 384 * 4
    """The number of frequencies in the positional encoding for vertex positions."""
    num_register_tokens: int = 16
    """The dimension of the feedforward network in the transformer."""
    register_token_pe: Literal['weighted_mean', 'maxmin_mean'] = 'weighted_mean'
    """The type of positional encoding to use for the register tokens."""
    use_view_token: bool = True
    """Whether to use the view & bos token in the transformer."""
    dropout: float = 0.1
    """The dropout rate in the transformer."""
    activation: Literal['gelu', 'swiglu'] = 'gelu'
    """The activation function in the transformer."""
    norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """The type of normalization to use in the transformer."""
    norm_first: bool = False
    """Whether to normalize the input before the transformer."""
    view_indep_qk_norm: bool = False
    """Whether to apply normalization to query and key."""
    qk_norm: bool = False
    """Whether to apply normalization to query and key."""
    bias: bool = True
    """Whether to use bias in the transformer."""

    pe_type: Literal['nerf', 'rope', 'rope_only_center'] = 'nerf'
    """The type of positional encoding to use."""
    rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed', 'triangle_center'] = 'triangle'
    """The type of RoPE to use."""
    rope_double_max_freq: bool = False
    """Whether to double the max frequency for RoPE."""
    vertex_pe_num_freqs: int = 6
    """The number of frequencies in the positional encoding for vertex positions."""
    intra_rope_pe_num_freqs: int = 20
    """The number of frequencies in the positional encoding for intra-triangle relative positions (only used for 'rope_only_center')."""

    # vertex normal encoder
    use_vn_encoder: bool = False
    """Whether to use the vertex normal encoder."""
    vn_pe_num_freqs: int = 6
    """The number of frequencies in the positional encoding for vertex normals."""
    vn_encoder_norm_type: Literal['none', 'layer_norm', 'rms_norm'] = 'none'
    """The type of normalization to use in the vertex normal encoder."""

    # texture patch encoder
    texture_encode_patch_size: int = 32
    """The size of the triangle's transformed texture patch for encoder."""
    texture_channels: int = 13  # diffuse, specular, normal, roughness, irradiance
    """The number of channels in the texture patch."""
    texture_encoder_norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """The type of normalization to use in the texture encoder."""
    log_roughness_encoding: bool = False
    """Whether to log the roughness encoding in the texture encoder."""
    separate_light_strength: bool = False
    """Whether to separate the light strength from the texture encoder."""
    auto_interpolate_texture: bool = False
    """Whether to automatically interpolate input texture to texture_encode_patch_size if resolution mismatch."""

    # Environment Lighting
    use_env_lighting: bool = False
    """Whether to use environment lighting."""
    envmap_vae_model_id: str = 'Qwen/Qwen-Image'
    """The model id of the VAE to use for the environment lighting."""
    envmap_latent_patch_size: int = 8
    """The size of the latent patch for the environment lighting."""
    envmap_latent_channels: int = 16
    """The number of channels in the latent of the environment lighting."""
    envmap_ray_dir_map_patch_size: int = 64
    """The size of the ray direction map patch for the environment lighting."""
    envmap_encoder_norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm'
    """The type of normalization to use in the environment lighting encoder."""

    # triangle ordering and windowed attention
    use_triangle_ordering: bool = False
    """Whether to use triangle ordering."""
    triangle_ordering_order: Literal['z', 'z-trans', 'hilbert', 'hilbert-trans', 'shuffle-order', 'shift-order'] = 'z'
    """The order to use for triangle ordering."""

    # view transformer triangle ordering (separate from encoder)
    view_transformer_use_triangle_ordering: bool = False
    """Whether to use triangle ordering in the view transformer decoder."""
    view_transformer_triangle_ordering_order: Literal['z', 'z-trans', 'hilbert', 'hilbert-trans', 'shuffle-order', 'shift-order'] = 'z'
    """The order to use for triangle ordering in the view transformer decoder."""
    use_local_attention: bool = False
    """Whether to use local attention."""
    local_attention_window_size_half: int = 256
    """The size of the local attention window (half of the window size)."""

    # Flex Attention (Sliding Window + Attention Sink)
    use_flex_attention: bool = False
    """Whether to use flex attention with block masks."""
    flex_attention_block_size: int = 128
    """The block size for flex attention."""
    put_light_tokens_first: bool = False
    """Whether to put light source tokens at the front of the sequence when reordering (after register tokens)."""
    put_light_tokens_in_sink: bool = True
    """Whether to put light tokens in the sink region (for varlen packed sequence). If False, light tokens go to normal region."""
    put_env_tokens_in_sink: bool = True
    """Whether to put env tokens in the sink region (for SequenceBuilder packing). If False, env tokens are mixed into normal_geo and spatially sorted."""
    add_summary_tokens: bool = False
    """Whether to add summary tokens to the sequence (for varlen packed sequence)."""
    summary_block_size: int = 64
    """The block size for summary tokens."""
    
    # Volume encoding
    use_volumes: bool = False
    """Whether to use volume tokens in the model."""
    volume_density_input_dim: int = 64
    """Input dimension for volume density (4x4x4 = 64 for 4x4x4 voxel grid)."""
    volume_use_density_mlp: bool = True
    """Whether to use MLP for encoding volume density. If False, uses a single linear layer."""

    # Perceiver
    use_perceiver: bool = False
    """Whether to use Perceiver."""
    num_perceiver_tokens: int = 64

    decoding_method: Literal['texture_patch', 'view_transformer'] = 'texture_patch'
    """The method to decode the texture patch."""

    # texture patch decoder
    texture_decode_patch_size: int = 32
    """The size of the triangle's transformed texture patch for decoder."""

    # view transformer
    view_transformer_latent_dim: int = 384
    """The latent dimension of the view transformer."""
    view_transformer_ffn_hidden_dim: int = 384 * 4
    """The hidden dimension of the feedforward network in the view transformer."""
    view_transformer_n_heads: int = 8
    """The number of heads in the view transformer."""
    view_transformer_num_kv_heads: Optional[int] = None
    """The number of key and value heads in the view transformer (for GQA/MQA). If None, set to view_transformer_n_heads (MHA)."""
    view_transformer_n_layers: int = 10
    """The number of layers in the view transformer."""
    view_transformer_include_self_attn: bool = True
    """Whether to include self-attention between ray tokens in the view transformer."""
    view_transformer_self_attn_before_cross: bool = False
    """Whether to put self-attention before cross-attention in the view transformer. Default is False (self-attn after cross-attn)."""
    view_transformer_use_swin_attn: bool = False
    """Whether to use swin self-attention in the view transformer."""
    vdir_pe_type: Literal['nerf'] = 'nerf'
    """The type of positional encoding to use for view direction."""
    vdir_num_freqs: int = 0
    """The number of frequencies in the positional encoding for view direction."""
    patch_size: int = 8
    """The size of the image patch for encoding in the view transformer."""
    output_patch_size: Optional[int] = None
    """The size of the image patch for decoding in the view transformer. If None, defaults to patch_size."""
    output_channel: Optional[int] = None
    """The number of output channels in the view transformer. If None, defaults to 4 (if include_alpha) or 3."""
    latent_vae_model_id: Optional[str] = None
    """The model id of the latent VAE to use for the view transformer. If None, no latent VAE is used."""
    include_alpha: bool = False
    """Whether to include the alpha channel in the texture patch."""
    freeze_radiosity_transformer_if_initialized: bool = False
    """Whether to freeze the radiosity transformer if it is loaded from pre-trained ckpt."""
    use_dpt_decoder: bool = False
    """Whether to use DPT decoder for rendering."""
    dpt_features: int = 128
    """The dim of internal features in the DPT decoder."""
    dpt_out_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    """The number of output channels per layer in the DPT decoder."""
    dpt_out_layers: Optional[List[int]] = None
    """The layers to use for the DPT decoder."""
    view_transformer_use_layers: Optional[List[int]] = None
    """The layers from view-independant transformer to use for the view transformer."""

    custom_init: bool = False
    """Whether to use custom initialization for the transformer."""

    # DeepStack texture injection
    deepstack_injection_map: Optional[Dict[int, int]] = None
    """
    Layer-wise texture injection map: {layer_idx: feature_idx}

    - layer_idx: which transformer layer to inject (0-indexed)
    - feature_idx: which texture feature to inject
      - 0: original texture (single-res) or global downsampled (dual-res)
      - 1: top-left patch (dual-res only, requires input texture 64x64)
      - 2: top-right patch (dual-res only, requires input texture 64x64)
      - 3: bottom-left patch (dual-res only, requires input texture 64x64)

    Examples:
      - Single-res: {10: 0, 12: 0, 14: 0}  # inject original at layers 10, 12, 14
      - Dual-res: {10: 0, 11: 1, 12: 2, 13: 3, 14: 0}  # mix different features
      - Mixed strategy: {8: 0, 10: 1, 10: 2, 12: 3, 15: 0}  # global → local → global

    Set to None to disable DeepStack injection.
    """
    deepstack_dual_input_size: int = 64
    """Input resolution for dual-resolution features (patches 1/2/3). Will downsample/upsample to this size."""

    deepstack_decoder_injection_map: Optional[Dict[int, int]] = None
    """
    Decoder layer-wise texture injection map (same format as encoder): {layer_idx: feature_idx}

    Only used when decoding_method='view_transformer'.
    Set to None to disable decoder injection.

    Example: {5: 0, 7: 1, 9: 0}  # inject at view transformer layers 5, 7, 9
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deepstack_injection_map",
            _normalize_deepstack_map(
                self.deepstack_injection_map,
                field_name="deepstack_injection_map",
                layer_count=self.num_layers,
            ),
        )
        object.__setattr__(
            self,
            "deepstack_decoder_injection_map",
            _normalize_deepstack_map(
                self.deepstack_decoder_injection_map,
                field_name="deepstack_decoder_injection_map",
                layer_count=self.view_transformer_n_layers,
            ),
        )

    def get(self, key, default=None):
        return getattr(self, key, default)
