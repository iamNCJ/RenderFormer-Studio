from dataclasses import dataclass
from typing import Literal, Optional
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange


from .attention import TransformerEncoder, TransformerDecoder
from .pe import NeRFEncoding
from .view_transformer import ViewTransformer
from .config import RenderTransformerConfig
from .utils import map_layer_indices
from .volume_encoder import VolumeEncoder
from .sequence_builder import SequenceBuilder


class TriangleRadiosityTransformer(nn.Module):
    def __init__(self, config: RenderTransformerConfig):
        super(TriangleRadiosityTransformer, self).__init__()
        self.config = config

        if self.config.pe_type == 'nerf':
            # vertex PE and projections
            self.tri_vpos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vertex_pe_num_freqs,
                include_input=True
            )
            # triangle pe projection
            self.tri_encoding_proj = nn.Linear(
                self.tri_vpos_pe.get_out_dim(),
                self.config.latent_dim
            )
            # reuse this config for ablation...
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.tri_encoding_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.tri_encoding_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.tri_encoding_norm = nn.Identity()
            self.rope_dim = None
        elif self.config.pe_type == 'rope':
            self.rope_dim = self.config.vertex_pe_num_freqs
        elif self.config.pe_type == 'rope_only_center':
            # inner-triangle position encoding
            self.intra_tri_pos_emb = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vertex_pe_num_freqs,
                include_input=True
            )
            self.intra_tri_pos_emb_proj = nn.Linear(
                self.intra_tri_pos_emb.get_out_dim(),
                self.config.latent_dim
            )
            if self.config.norm_type == 'layer_norm':
                self.intra_tri_pos_emb_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.norm_type == 'rms_norm':
                self.intra_tri_pos_emb_norm = nn.RMSNorm(self.config.latent_dim)
            else:
                raise ValueError(f"Invalid normalization type: {self.config.norm_type}")
            # intra-triangle relative position encoding
            self.rope_dim = self.config.intra_rope_pe_num_freqs
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        if self.config.use_vn_encoder:
            self.vn_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=self.config.vn_pe_num_freqs,
                include_input=True
            )
            self.vn_encoding_proj = nn.Linear(
                self.vn_pe.get_out_dim(),
                self.config.latent_dim
            )
            if self.config.vn_encoder_norm_type == 'layer_norm':
                self.vn_encoder_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'rms_norm':
                self.vn_encoder_norm = nn.RMSNorm(self.config.latent_dim)
            elif self.config.vn_encoder_norm_type == 'none':
                self.vn_encoder_norm = nn.Identity()
            else:
                raise ValueError(f"Invalid vertex normal encoder normalization type: {self.config.vn_encoder_norm_type}")

        # texture encoder
        self.texture_encoder = nn.Linear(
            self.config.texture_channels * self.config.texture_encode_patch_size * self.config.texture_encode_patch_size,
            self.config.latent_dim
        )
        if self.config.texture_encoder_norm_type == 'layer_norm':
            self.texture_encoder_norm = nn.LayerNorm(self.config.latent_dim)
        elif self.config.texture_encoder_norm_type == 'rms_norm':
            self.texture_encoder_norm = nn.RMSNorm(self.config.latent_dim)
        else:
            raise ValueError(f"Invalid texture encoder normalization type: {self.config.texture_encoder_norm_type}")
        if self.config.separate_light_strength:
            self.light_strength_encoder = nn.Linear(
                3,
                self.config.latent_dim
            )
            if self.config.texture_encoder_norm_type == 'layer_norm':
                self.light_strength_encoder_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.texture_encoder_norm_type == 'rms_norm':
                self.light_strength_encoder_norm = nn.RMSNorm(self.config.latent_dim)
            else:
                raise ValueError(f"Invalid light strength encoder normalization type: {self.config.texture_encoder_norm_type}")

        # Environment map processing components
        if self.config.use_env_lighting:
            # Calculate env_proj input dimension based on config
            # VAE latent: [B, D, H_lat, W_lat] where D = envmap_latent_channels
            # After patchify with patch_size: [B, N_patches, D * patch_size * patch_size]
            env_latent_patch_dim = self.config.envmap_latent_channels * self.config.envmap_latent_patch_size * self.config.envmap_latent_patch_size
            self.env_proj = nn.Linear(env_latent_patch_dim, self.config.latent_dim)

            # normalization for env_proj
            if self.config.envmap_encoder_norm_type == 'layer_norm':
                self.env_proj_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.envmap_encoder_norm_type == 'rms_norm':
                self.env_proj_norm = nn.RMSNorm(self.config.latent_dim)
            else:
                raise ValueError(f"Invalid envmap encoder normalization type: {self.config.envmap_encoder_norm_type}")

            # ray_dir_proj
            ray_dir_patch_dim = 3 * self.config.envmap_ray_dir_map_patch_size * self.config.envmap_ray_dir_map_patch_size
            self.ray_dir_proj = nn.Linear(ray_dir_patch_dim, self.config.latent_dim)
            
            # normalization layers for ray_dir
            if self.config.texture_encoder_norm_type == 'layer_norm':
                self.ray_dir_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.texture_encoder_norm_type == 'rms_norm':
                self.ray_dir_norm = nn.RMSNorm(self.config.latent_dim)
            else:
                raise ValueError(f"Invalid ray_dir norm type: {self.config.texture_encoder_norm_type}")
            
            # env light strength encoder (separate from triangle light strength)
            self.env_light_strength_encoder = nn.Linear(1, self.config.latent_dim)
            if self.config.texture_encoder_norm_type == 'layer_norm':
                self.env_light_strength_encoder_norm = nn.LayerNorm(self.config.latent_dim)
            elif self.config.texture_encoder_norm_type == 'rms_norm':
                self.env_light_strength_encoder_norm = nn.RMSNorm(self.config.latent_dim)
            else:
                raise ValueError(f"Invalid env light strength encoder normalization type: {self.config.texture_encoder_norm_type}")

        # learnable tokens
        self.tri_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
        self.reg_tokens = nn.Parameter(torch.randn(1, self.config.num_register_tokens, self.config.latent_dim))
        
        # Volume encoder (only if volumes are enabled)
        if self.config.use_volumes:
            self.volume_encoder = VolumeEncoder(
                latent_dim=self.config.latent_dim,
                density_input_dim=self.config.volume_density_input_dim,
                use_density_mlp=self.config.volume_use_density_mlp,
            )
        else:
            self.volume_encoder = None
        
        # Sequence builder for packed sequences
        # Note: shuffle-order and shift-order are per-layer strategies, not spatial ordering
        # For SequenceBuilder, we only use spatial ordering types
        ordering_order = None
        if self.config.use_triangle_ordering:
            if self.config.triangle_ordering_order in {'z', 'z-trans', 'hilbert', 'hilbert-trans'}:
                ordering_order = self.config.triangle_ordering_order
            else:
                # shuffle-order and shift-order are handled per-layer in TransformerEncoder
                # Use default 'z' for SequenceBuilder
                ordering_order = 'z'
        
        self.sequence_builder = SequenceBuilder(
            latent_dim=self.config.latent_dim,
            summary_block_size=self.config.summary_block_size,
            z_order_depth=16,
            add_summary_tokens=self.config.add_summary_tokens,
            use_z_order=self.config.use_triangle_ordering,
            put_light_tokens_in_sink=self.config.put_light_tokens_in_sink,
            put_env_tokens_in_sink=self.config.put_env_tokens_in_sink,
            ordering_order=ordering_order,
        )
        if self.config.use_view_token:
            self.view_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
            self.bos_token = nn.Parameter(torch.randn(1, 1, self.config.latent_dim))
            # view info (mvp 4x4 mat) projection
            self.mvp_proj = nn.Linear(4 * 4, self.config.latent_dim)
        self.skip_token_num = self.config.num_register_tokens + 2 if self.config.use_view_token else self.config.num_register_tokens
        if self.config.use_perceiver:
            self.perceiver_query_num = self.config.num_perceiver_tokens
            self.perceiver_query = nn.Parameter(torch.randn(1, self.perceiver_query_num, self.config.latent_dim))
            self.skip_token_num += self.perceiver_query_num

        # core radiosity transformer
        if not self.config.use_perceiver:
            self.transformer = TransformerEncoder(
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                hidden_dim=self.config.latent_dim,
                ffn_hidden_dim=self.config.dim_feedforward,
                dropout=self.config.dropout,
                activation=self.config.activation,
                norm_type=self.config.norm_type,
                norm_first=self.config.norm_first,
                rope_dim=self.rope_dim,
                rope_type=self.config.rope_type,
                bias=self.config.bias,
                qk_norm=self.config.view_indep_qk_norm,
                rope_double_max_freq=self.config.rope_double_max_freq,
                use_triangle_ordering=self.config.use_triangle_ordering,
                triangle_ordering_order=self.config.triangle_ordering_order,
                use_local_attention=self.config.use_local_attention,
                local_attention_window_size_half=self.config.local_attention_window_size_half,
                use_flex_attention=self.config.use_flex_attention,
                flex_attention_block_size=self.config.flex_attention_block_size,
                add_summary_tokens=self.config.add_summary_tokens,
                summary_block_size=self.config.summary_block_size,
                input_tri_center_pos=self.config.pe_type == 'rope_only_center',
                # DeepStack config - let transformer create its own projectors internally
                deepstack_injection_map=self.config.deepstack_injection_map,
                deepstack_feature_dim=(
                    self.config.texture_channels *
                    self.config.texture_encode_patch_size *
                    self.config.texture_encode_patch_size
                ) if self.config.deepstack_injection_map is not None else None,
            )
        else:
            self.transformer = TransformerDecoder(
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                hidden_dim=self.config.latent_dim,
                ffn_hidden_dim=self.config.dim_feedforward,
                num_kv_heads=self.config.num_kv_heads,
                ctx_dim=self.config.latent_dim,
                dropout=self.config.dropout,
                include_self_attn=True,
                qk_norm=self.config.view_indep_qk_norm,
                norm_type=self.config.norm_type,
                norm_first=self.config.norm_first,
                bias=self.config.bias,
                rope_dim=self.rope_dim,
                rope_type=self.config.rope_type,
                rope_double_max_freq=self.config.rope_double_max_freq,
                use_triangle_ordering=self.config.use_triangle_ordering,
                triangle_ordering_order=self.config.triangle_ordering_order,
            )

        # output projection
        if config.decoding_method == 'texture_patch':
            out_tex_patch_size = self.config.texture_decode_patch_size
            self.out_proj = nn.Linear(self.config.latent_dim, 3 * out_tex_patch_size * out_tex_patch_size)
            self.act = nn.ELU(alpha=1e-3)  # 0.5/255
        elif config.decoding_method == 'view_transformer':
            self.view_transformer = ViewTransformer(config)
        else:
            raise ValueError(f"Invalid decoding method: {config.decoding_method}")

        if self.config.custom_init:
            self.initialize_weights()

        # Initialize DeepStack projectors to zero AFTER custom_init
        # Transformers create their own projectors internally, we just need to zero-init them
        if hasattr(self.transformer, 'deepstack_projectors') and self.transformer.deepstack_projectors is not None:
            for proj in self.transformer.deepstack_projectors:
                nn.init.constant_(proj.weight, 0)
                nn.init.constant_(proj.bias, 0)

        # Decoder projectors are inside view_transformer.transformer (TransformerDecoder)
        if hasattr(self, 'view_transformer') and hasattr(self.view_transformer.transformer, 'deepstack_projectors') and self.view_transformer.transformer.deepstack_projectors is not None:
            for proj in self.view_transformer.transformer.deepstack_projectors:
                nn.init.constant_(proj.weight, 0)
                nn.init.constant_(proj.bias, 0)

        self.use_layers = []
        self.use_indices = []
        if self.config.view_transformer_use_layers is not None:
            self.use_layers, self.use_indices = map_layer_indices(self.config.view_transformer_use_layers)
            print(f"Using view-independant transformer layers: {self.use_layers}")
            print(f"Using view-dependant transformer per-layer feature indices: {self.use_indices}")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # texture decoder
        if self.config.decoding_method == 'texture_patch':
            nn.init.constant_(self.out_proj.weight, 0)
            nn.init.constant_(self.out_proj.bias, 0)
        elif self.config.decoding_method == 'view_transformer':
            if self.config.use_dpt_decoder:
                # only zero out the last conv layer in dpt
                nn.init.constant_(self.view_transformer.out_dpt.scratch.output_conv2[2].weight, 0)
                nn.init.constant_(self.view_transformer.out_dpt.scratch.output_conv2[2].bias, 0)
            else:
                nn.init.constant_(self.view_transformer.out_proj.weight, 0)
                nn.init.constant_(self.view_transformer.out_proj.bias, 0)
 
    @property
    def device(self):
        return next(self.parameters()).device

    def process_tri_vpos_list(self, tri_vpos_list, valid_mask, env_token_num=0):
        """
        Process tri_vpos_list for RoPE positional encoding.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param valid_mask: [batch_size, max_num_tri]
        :param env_token_num: int, number of environment map tokens (default: 0)
        :return: processed tri_vpos_list, updated valid_mask
        """
        # Calculate center position for register tokens (and env tokens if provided)
        # skip_token_num includes: perceiver_query (if any) + register_tokens + view_tokens (if any)
        # But we need to insert env_tokens between register_tokens and view_tokens
        # So we need to calculate positions separately
        
        # First, calculate the number of tokens before env_tokens
        num_register_only = self.config.num_register_tokens
        if self.config.use_perceiver:
            num_register_only += self.perceiver_query_num
        
        if self.config.register_token_pe == 'weighted_mean':
            mask_weight = (valid_mask.float() / (valid_mask.sum(dim=1, keepdim=True) + 1e-5))[..., None]
            weighted_tri_pos = mask_weight * tri_vpos_list
            center_pos = weighted_tri_pos.sum(dim=1).reshape(-1, 3, 3).mean(dim=1, keepdim=True)
            # Build position list in order: register tokens -> env tokens -> view tokens -> triangle tokens
            pos_parts = []
            # Register tokens positions (perceiver + register)
            register_pos = center_pos.repeat(1, num_register_only, 3)
            pos_parts.append(register_pos)
            # Env tokens positions (if provided)
            if env_token_num > 0:
                env_pos = center_pos.repeat(1, env_token_num, 3)
                pos_parts.append(env_pos)
            # View tokens positions (if use_view_token)
            if self.config.use_view_token:
                view_pos = center_pos.repeat(1, 2, 3)  # bos_token + mvp_emb
                pos_parts.append(view_pos)
            # Triangle tokens positions
            pos_parts.append(tri_vpos_list)
            tri_vpos_list = torch.cat(pos_parts, dim=1)
        elif self.config.register_token_pe == 'maxmin_mean':
            # mask invalid positions with large negative value so they don't affect max
            masked_pos = tri_vpos_list.masked_fill(~valid_mask[..., None].expand_as(tri_vpos_list), float('-inf'))
            max_pos = masked_pos.max(dim=1, keepdim=True)[0]
            # mask invalid positions with large positive value so they don't affect min
            masked_pos = tri_vpos_list.masked_fill(~valid_mask[..., None].expand_as(tri_vpos_list), float('inf')) 
            min_pos = masked_pos.min(dim=1, keepdim=True)[0]
            center_pos = ((max_pos + min_pos) / 2).reshape(-1, 3, 3).mean(dim=1, keepdim=True)
            # Build position list in order: register tokens -> env tokens -> view tokens -> triangle tokens
            pos_parts = []
            # Register tokens positions (perceiver + register)
            register_pos = center_pos.repeat(1, num_register_only, 3)
            pos_parts.append(register_pos)
            # Env tokens positions (if provided)
            if env_token_num > 0:
                env_pos = center_pos.repeat(1, env_token_num, 3)
                pos_parts.append(env_pos)
            # View tokens positions (if use_view_token)
            if self.config.use_view_token:
                view_pos = center_pos.repeat(1, 2, 3)  # bos_token + mvp_emb
                pos_parts.append(view_pos)
            # Triangle tokens positions
            pos_parts.append(tri_vpos_list)
            tri_vpos_list = torch.cat(pos_parts, dim=1)
        else:
            raise ValueError(f"Invalid register token pe: {self.config.register_token_pe}")

        # construct valid mask, things you want is True
        # Order: register tokens -> env tokens -> view tokens -> triangle tokens
        valid_mask_parts = []
        # Register tokens mask (perceiver + register)
        valid_mask_parts.append(torch.ones((tri_vpos_list.size(0), num_register_only), dtype=torch.bool, device=valid_mask.device))
        # Env tokens mask (if provided)
        if env_token_num > 0:
            valid_mask_parts.append(torch.ones((tri_vpos_list.size(0), env_token_num), dtype=torch.bool, device=valid_mask.device))
        # View tokens mask (if use_view_token)
        if self.config.use_view_token:
            valid_mask_parts.append(torch.ones((tri_vpos_list.size(0), 2), dtype=torch.bool, device=valid_mask.device))
        # Triangle tokens mask
        valid_mask_parts.append(valid_mask)
        valid_mask = torch.cat(valid_mask_parts, dim=1)
        # if self.config.pe_type == 'rope_only_center':
        #     tri_vpos_list = tri_vpos_list.reshape(tri_vpos_list.size(0), -1, 3, 3).mean(dim=2)

        return tri_vpos_list, valid_mask

    def construct_seq(self, tri_vpos_list, texture_patch_list, mvps, valid_mask, vns=None, light_strength=None, env_tokens=None):
        """
        From input triangle list + texture patches, construct the sequence for transformer.

        :param tri_vpos_list: [batch_size, max_num_tri, 9], padded
        :param texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        :param mvps: [batch_size, 4 * 4]
        :param valid_mask: [batch_size, max_num_tri]
        :param vns: [batch_size, max_num_tri, 3, 3], padded
        :param light_strength: [batch_size, max_num_tri, 3], padded
        :param env_tokens: [batch_size, 2*N_patches, latent_dim], optional environment map tokens
        :return: seq, valid_mask, tri_vpos_list, deepstack_texture_embs
        """
        batch_size = tri_vpos_list.size(0)

        # Prepare DeepStack texture embeddings (BEFORE any reordering)
        deepstack_texture_embs = None
        if self.config.deepstack_injection_map is not None:
            deepstack_texture_embs = self._prepare_deepstack_texture(texture_patch_list)
            # Shape: dict with keys '0', '1', '2', '3' containing [B, max_tri, feature_dim]

        # vertex normal encoding
        if self.config.use_vn_encoder:
            vn_emb = self.vn_encoder_norm(self.vn_encoding_proj(self.vn_pe(vns).to(vns.dtype)))
        else:
            vn_emb = 0.

        # texture interpolation (if needed)
        if self.config.auto_interpolate_texture:
            current_res = texture_patch_list.size(-1)  # assuming square patches
            target_res = self.config.texture_encode_patch_size
            if current_res != target_res:
                # texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size]
                batch_size, max_num_tri, texture_channel, _, _ = texture_patch_list.shape
                # reshape to [batch_size * max_num_tri, texture_channel, patch_size, patch_size] for interpolation
                texture_flat = texture_patch_list.reshape(batch_size * max_num_tri, texture_channel, current_res, current_res)
                # interpolate to target resolution
                # Use 'area' mode for downsample (better quality), 'bilinear' for upsample
                if current_res > target_res:
                    # Downsample: use area-based adaptive average pooling
                    texture_interp = torch.nn.functional.interpolate(
                        texture_flat,
                        size=(target_res, target_res),
                        mode='area',
                    )
                else:
                    # Upsample: use bilinear interpolation
                    texture_interp = torch.nn.functional.interpolate(
                        texture_flat,
                        size=(target_res, target_res),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True,
                    )
                # reshape back to [batch_size, max_num_tri, texture_channel, target_res, target_res]
                texture_patch_list = texture_interp.reshape(batch_size, max_num_tri, texture_channel, target_res, target_res)

        # texture encoding
        tri_tex_emb = self.texture_encoder_norm(self.texture_encoder(
            texture_patch_list.reshape(texture_patch_list.size(0), texture_patch_list.size(1), -1)
        ))
        if self.config.separate_light_strength:
            tri_tex_emb = tri_tex_emb + self.light_strength_encoder_norm(self.light_strength_encoder(light_strength))

        # construct sequence
        tokens = []
        if self.config.use_perceiver:
            tokens.append(self.perceiver_query.expand(batch_size, -1, -1))
        tokens.append(self.reg_tokens.expand(batch_size, -1, -1))
        
        # Add env tokens if provided
        # put env tokens right after register tokens as they are also in attention sink
        if env_tokens is not None and self.config.use_env_lighting:
            tokens.append(env_tokens)
        
        if self.config.use_view_token:
            tokens.append(self.bos_token.expand(batch_size, 1, -1))
            mvp_emb = self.mvp_proj(mvps)[:, None] + self.view_token
            tokens.append(mvp_emb)

        if self.config.pe_type == 'nerf':
            tri_vpos_pe = self.tri_vpos_pe(tri_vpos_list)
            tri_emb = self.tri_encoding_norm(self.tri_encoding_proj(tri_vpos_pe)) + self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        elif self.config.pe_type == 'rope':
            tri_emb = self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        elif self.config.pe_type == 'rope_only_center':
            tri_center_pos = tri_vpos_list.reshape(batch_size, -1, 3, 3).mean(dim=2, keepdim=True)
            inner_tri_pos = tri_vpos_list.reshape(batch_size, -1, 3, 3) - tri_center_pos
            inner_tri_pos = inner_tri_pos.reshape(batch_size, -1, 9)
            tri_emb = self.intra_tri_pos_emb_norm(self.intra_tri_pos_emb_proj(self.intra_tri_pos_emb(inner_tri_pos))) + self.tri_token + tri_tex_emb + vn_emb
            tokens.append(tri_emb)
        else:
            raise ValueError(f"Invalid positional encoding type: {self.config.pe_type}")

        seq = torch.cat(tokens, dim=1)

        # pad triangle pos (for RoPE) and valid mask (for all)
        # use center pos for RoPE on auxiliary tokens (register tokens and env tokens)
        env_token_num = env_tokens.shape[1] if env_tokens is not None else 0
        tri_vpos_list, valid_mask = self.process_tri_vpos_list(tri_vpos_list, valid_mask, env_token_num)
        
        if self.config.pe_type == 'rope_only_center':
            tri_vpos_list = tri_vpos_list.reshape(batch_size, -1, 3, 3).mean(dim=2)  # only use center pos for RoPE

        return seq, valid_mask, tri_vpos_list, deepstack_texture_embs

    def _process_env_map(self, latent_ldr: torch.Tensor, latent_hdr: torch.Tensor, env_lighting_strength: torch.Tensor, env_ray_dir: torch.Tensor) -> torch.Tensor:
        """
        Process environment map latents and ray direction map to generate env tokens.
        
        Args:
            latent_ldr: [batch_size, D, H_lat, W_lat] - VAE encoded LDR environment map
            latent_hdr: [batch_size, D, H_lat, W_lat] - VAE encoded normalized log HDR environment map
            env_lighting_strength: [batch_size] - max log brightness per batch
            env_ray_dir: [batch_size, 256, 512, 3] - ray direction map
            
        Returns:
            env_tokens: [batch_size, 2*N_patches, latent_dim]
        """
        device = latent_ldr.device
        dtype = latent_ldr.dtype
        
        # Patchify VAE latents
        patch_size = self.config.envmap_latent_patch_size
        B, D, H_lat, W_lat = latent_ldr.shape
        # Ensure dimensions are divisible by patch_size
        H_patches = H_lat // patch_size
        W_patches = W_lat // patch_size
        # Crop if needed
        H_crop = H_patches * patch_size
        W_crop = W_patches * patch_size
        latent_ldr_crop = latent_ldr[:, :, :H_crop, :W_crop]
        latent_hdr_crop = latent_hdr[:, :, :H_crop, :W_crop]
        
        latent_ldr_patches = rearrange(latent_ldr_crop, 'b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1=patch_size, p2=patch_size)  # [B, N_patches, D_patch]
        latent_hdr_patches = rearrange(latent_hdr_crop, 'b d (h p1) (w p2) -> b (h w) (d p1 p2)', p1=patch_size, p2=patch_size)  # [B, N_patches, D_patch]
        
        # Project env latent patches
        env_latent_patches = torch.cat([latent_ldr_patches, latent_hdr_patches], dim=1)  # [B, 2*N_patches, D_patch]
        env_tokens = self.env_proj_norm(self.env_proj(env_latent_patches.to(dtype)))  # [B, 2*N_patches, latent_dim]
        
        # Process ray direction map
        # Patchify ray direction map
        ray_dir_patch_size = self.config.envmap_ray_dir_map_patch_size
        ray_dir_patches = rearrange(env_ray_dir, 'b (h p1) (w p2) c -> b (h w) (c p1 p2)', p1=ray_dir_patch_size, p2=ray_dir_patch_size)  # [B, N_patches, 3*p1*p2]
        
        # Project ray_dir and get embeddings
        ray_dir_emb = self.ray_dir_norm(self.ray_dir_proj(ray_dir_patches.to(dtype)))  # [B, N_patches, latent_dim]
        
        # Add ray_dir embeddings
        env_tokens = env_tokens + ray_dir_emb.repeat(1, 2, 1)  # [B, 2*N_patches, latent_dim]
        
        # Add separate lighting strength embedding
        env_lighting_strength_expanded = env_lighting_strength[:, None, None].expand(-1, env_tokens.shape[1], 1).to(dtype)  # [B, 2*N_patches, 1]
        env_light_emb = self.env_light_strength_encoder_norm(
            self.env_light_strength_encoder(env_lighting_strength_expanded)
        )  # [B, 2*N_patches, latent_dim]
        env_tokens = env_tokens + env_light_emb  # [B, 2*N_patches, latent_dim]
        
        return env_tokens

    def encode_volumes(
        self,
        volume_density: torch.Tensor,
        volume_position: torch.Tensor,
        volume_rotation: torch.Tensor,
        volume_scale: torch.Tensor,
        volume_scattering: torch.Tensor,
        volume_absorption: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            vol_embs: [B, N_vol, D]
            vol_centers: [B, N_vol, 3]
        """
        if self.volume_encoder is None:
            raise ValueError("Volume encoder is not initialized. Set use_volumes=True in config.")
        return self.volume_encoder(
            volume_density,
            volume_position,
            volume_rotation,
            volume_scale,
            volume_scattering,
            volume_absorption,
        )

    def _prepare_deepstack_texture(self, texture_patch_list):
        """
        Prepare texture features for DeepStack injection.

        Returns a dict with separate features:
        - Feature 0: original texture (patch_size x patch_size) or global downsampled (from dual_input_size)
        - Feature 1: top-left patch (dual-res only, requires dual_input_size input)
        - Feature 2: top-right patch (dual-res only, requires dual_input_size input)
        - Feature 3: bottom-left patch (dual-res only, requires dual_input_size input)

        Args:
            texture_patch_list: [B, max_tri, C, H, W]

        Returns:
            dict: {
                '0': [B, max_tri, C*patch_size*patch_size],
                '1': [B, max_tri, C*patch_size*patch_size],  # only if input >= dual_input_size
                '2': [B, max_tri, C*patch_size*patch_size],  # only if input >= dual_input_size
                '3': [B, max_tri, C*patch_size*patch_size],  # only if input >= dual_input_size
            }
            where patch_size = texture_encode_patch_size
        """
        batch_size, max_tri, C, H, W = texture_patch_list.shape
        features = {}

        # Target patch size from config
        patch_size = self.config.texture_encode_patch_size
        dual_input_size = self.config.deepstack_dual_input_size

        # Ensure input is at least patch_size x patch_size for feature 0
        if H < patch_size or W < patch_size:
            texture_patch_list = F.interpolate(
                texture_patch_list.reshape(batch_size * max_tri, C, H, W),
                size=(patch_size, patch_size),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            ).reshape(batch_size, max_tri, C, patch_size, patch_size)
            H, W = patch_size, patch_size

        # Feature 0: original (patch_size x patch_size) or global downsampled (from dual_input_size)
        if H == patch_size and W == patch_size:
            # Already target size, use directly
            features['0'] = texture_patch_list.reshape(batch_size, max_tri, -1)
        else:
            # Downsample to patch_size x patch_size global
            texture_flat = texture_patch_list.reshape(batch_size * max_tri, C, H, W)
            global_patch = F.interpolate(
                texture_flat, size=(patch_size, patch_size), mode='area'
            ).reshape(batch_size, max_tri, C, patch_size, patch_size)
            features['0'] = global_patch.reshape(batch_size, max_tri, -1)

        # Features 1-3: local patches (only if input >= dual_input_size)
        if H >= dual_input_size and W >= dual_input_size:
            # Ensure exactly dual_input_size for patch extraction
            if H != dual_input_size or W != dual_input_size:
                texture_flat = texture_patch_list.reshape(batch_size * max_tri, C, H, W)
                if H > dual_input_size:
                    texture_flat = F.interpolate(
                        texture_flat,
                        size=(dual_input_size, dual_input_size),
                        mode='area',
                    )
                else:
                    texture_flat = F.interpolate(
                        texture_flat,
                        size=(dual_input_size, dual_input_size),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True,
                    )
                texture_patch_list = texture_flat.reshape(
                    batch_size, max_tri, C, dual_input_size, dual_input_size
                )

            # Extract 3 local patch_size patches from dual_input_size input
            # Assuming dual_input_size = 2 * patch_size (e.g., 8 = 2*4, or 64 = 2*32)
            half_size = dual_input_size // 2
            # Each patch should be patch_size x patch_size
            # So we need to downsample each half_size region to patch_size
            texture_flat = texture_patch_list.reshape(batch_size * max_tri, C, dual_input_size, dual_input_size)

            # Top-left
            patch_1 = F.interpolate(
                texture_flat[:, :, :half_size, :half_size],
                size=(patch_size, patch_size),
                mode='area'
            ).reshape(batch_size, max_tri, -1)

            # Top-right
            patch_2 = F.interpolate(
                texture_flat[:, :, :half_size, half_size:],
                size=(patch_size, patch_size),
                mode='area'
            ).reshape(batch_size, max_tri, -1)

            # Bottom-left
            patch_3 = F.interpolate(
                texture_flat[:, :, half_size:, :half_size],
                size=(patch_size, patch_size),
                mode='area'
            ).reshape(batch_size, max_tri, -1)

            features['1'] = patch_1
            features['2'] = patch_2
            features['3'] = patch_3

        return features

    def forward(self, tri_vpos_list, texture_patch_list, mvps, valid_mask, vns=None, rays_o=None, rays_d=None, tri_vpos_view_tf=None, light_strength=None, env_latent_ldr=None, env_latent_hdr=None, env_lighting_strength=None, env_ray_dir=None, tf32_view_tf=False, use_packed_sequence=False, volume_density=None, volume_position=None, volume_rotation=None, volume_scale=None, volume_scattering=None, volume_absorption=None, volume_mask=None):
        """
        Forward pass of the transformer.

        tri_vpos_list: [batch_size, max_num_tri, 9], padded
        texture_patch_list: [batch_size, max_num_tri, texture_channel, patch_size, patch_size], padded
        mvps: [batch_size, 4 * 4]
        valid_mask: [batch_size, max_num_tri], things you want is True
        vns: [batch_size, max_num_tri, 9], padded
        light_strength: [batch_size, max_num_tri, 3], padded
        env_latent_ldr: [batch_size, D, H_lat, W_lat], optional VAE encoded LDR environment map
        env_latent_hdr: [batch_size, D, H_lat, W_lat], optional VAE encoded normalized log HDR environment map
        env_lighting_strength: [batch_size], optional max log brightness per batch
        env_ray_dir: [batch_size, 256, 512, 3], optional ray direction map for environment

        rays_o: [batch_size, num_views, 3]
        rays_d: [batch_size, num_views, img_h, img_w, 3]
        tri_vpos_view_tf: [batch_size, num_views, max_num_tri, 9], padded
        tf32_view_tf: bool, whether to use tf32 for view transformer
        """
        if valid_mask.ndim != 2 or valid_mask.shape != tri_vpos_list.shape[:2]:
            raise ValueError(
                "valid_mask must have shape [batch_size, max_num_tri], matching "
                f"tri_vpos_list; got {tuple(valid_mask.shape)} and "
                f"{tuple(tri_vpos_list.shape)}"
            )
        if not valid_mask.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid triangle")

        # Check if we should use varlen packed sequence
        # Allow varlen path even without volumes (volumes will be empty tensors)
        if use_packed_sequence:
            # ========== Varlen Packed Sequence Path ==========
            return self._forward_varlen(
                tri_vpos_list, texture_patch_list, mvps, valid_mask, vns,
                rays_o, rays_d, tri_vpos_view_tf, light_strength,
                env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir,
                volume_density, volume_position, volume_rotation, volume_scale,
                volume_scattering, volume_absorption, volume_mask, tf32_view_tf
            )

        # V1 checkpoints use full 9D triangle RoPE positions. PackedBatch stores
        # center positions for the V2 rope_only_center architecture, so routing
        # V1 through it changes the checkpoint's positional semantics. Preserve
        # the original padded path for V1/legacy PE modes.
        if self.config.pe_type != 'rope_only_center':
            if volume_density is not None:
                raise NotImplementedError(
                    "Padded volume inputs require pe_type='rope_only_center'"
                )
            return self._forward_legacy_padded(
                tri_vpos_list,
                texture_patch_list,
                mvps,
                valid_mask,
                vns,
                rays_o,
                rays_d,
                tri_vpos_view_tf,
                light_strength,
                env_latent_ldr,
                env_latent_hdr,
                env_lighting_strength,
                env_ray_dir,
                tf32_view_tf,
            )

        # ========== Unified Padded Sequence Path ==========
        # All padded paths now go through SequenceBuilder for consistent handling
        return self._forward_padded_unified(
            tri_vpos_list, texture_patch_list, mvps, valid_mask, vns,
            rays_o, rays_d, tri_vpos_view_tf, light_strength,
            env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir,
            volume_density, volume_position, volume_rotation, volume_scale,
            volume_scattering, volume_absorption, volume_mask, tf32_view_tf
        )

    def _forward_legacy_padded(
        self,
        tri_vpos_list,
        texture_patch_list,
        mvps,
        valid_mask,
        vns,
        rays_o,
        rays_d,
        tri_vpos_view_tf,
        light_strength,
        env_latent_ldr,
        env_latent_hdr,
        env_lighting_strength,
        env_ray_dir,
        tf32_view_tf,
    ):
        """Checkpoint-compatible padded path for V1 full-triangle PE modes."""
        env_tokens = None
        if (
            env_latent_ldr is not None
            and env_latent_hdr is not None
            and env_lighting_strength is not None
            and env_ray_dir is not None
            and self.config.use_env_lighting
        ):
            env_tokens = self._process_env_map(
                env_latent_ldr,
                env_latent_hdr,
                env_lighting_strength,
                env_ray_dir,
            )

        light_token_mask = None
        if self.config.put_light_tokens_first:
            if light_strength is not None:
                light_token_mask = (light_strength > 0).any(dim=-1) & valid_mask
            else:
                light_token_mask = (
                    texture_patch_list[:, :, 10, 0, 0] > 0
                ) & valid_mask

        seq, valid_mask_padded, positions, deepstack_texture_embs = self.construct_seq(
            tri_vpos_list,
            texture_patch_list,
            mvps,
            valid_mask,
            vns,
            light_strength,
            env_tokens,
        )
        env_token_num = env_tokens.shape[1] if env_tokens is not None else 0

        if not self.config.use_perceiver:
            seq = self.transformer(
                seq,
                src_key_padding_mask=valid_mask_padded,
                triangle_pos=positions,
                num_register_tokens=self.skip_token_num,
                env_token_num=env_token_num,
                out_layers=self.use_layers,
                light_token_mask=light_token_mask,
                deepstack_texture_embs=deepstack_texture_embs,
                triangle_valid_mask=valid_mask,
            )
        else:
            perceiver_query = seq[:, :self.perceiver_query_num]
            context_seq = seq[:, self.perceiver_query_num:]
            triangle_pos = positions[:, self.perceiver_query_num:]
            query_pos = positions[:, :self.perceiver_query_num]
            seq = self.transformer(
                perceiver_query,
                context_seq,
                src_key_padding_mask=valid_mask_padded[:, self.perceiver_query_num:],
                triangle_pos=triangle_pos,
                ray_pos=query_pos,
            )
            seq = torch.cat([seq, context_seq], dim=1)

        if self.config.decoding_method == 'texture_patch':
            out_tex_patch_size = self.config.texture_decode_patch_size
            out = self.out_proj(
                seq.reshape(-1, self.config.latent_dim)
            ).reshape(
                seq.size(0),
                -1,
                3,
                out_tex_patch_size,
                out_tex_patch_size,
            )
            return self.act(out)[:, self.skip_token_num:]

        if self.config.decoding_method != 'view_transformer':
            raise NotImplementedError(
                f"Unsupported decoding method: {self.config.decoding_method}"
            )
        if rays_o is None or rays_d is None or tri_vpos_view_tf is None:
            raise ValueError(
                "view_transformer decoding requires rays_o, rays_d, and "
                "tri_vpos_view_tf"
            )

        batch_size, num_views = rays_o.shape[:2]
        if self.use_layers:
            seq = seq.repeat_interleave(num_views, dim=1)
        else:
            seq = seq.repeat_interleave(num_views, dim=0)
        rays_o = rays_o.reshape(-1, *rays_o.shape[2:])
        rays_d = rays_d.reshape(-1, *rays_d.shape[2:])
        tri_vpos_view_tf = tri_vpos_view_tf.reshape(
            -1, *tri_vpos_view_tf.shape[2:]
        )
        view_valid_mask = valid_mask.repeat_interleave(num_views, dim=0)
        valid_mask_padded = valid_mask_padded.repeat_interleave(num_views, dim=0)

        # Full-triangle RoPE is sensitive to bf16 position rounding. Keep the
        # position transform in float32 while preserving the original 9D layout.
        with torch.no_grad():
            pos_seq, _ = self.process_tri_vpos_list(
                tri_vpos_view_tf.float(),
                view_valid_mask,
                env_token_num,
            )

        res = self.view_transformer(
            rays_o,
            rays_d,
            seq,
            pos_seq,
            valid_mask_padded,
            tf32_mode=tf32_view_tf,
            use_indices=self.use_indices,
            env_token_num=env_token_num,
            deepstack_texture_embs=(
                {
                    key: value.repeat_interleave(num_views, dim=0)
                    for key, value in deepstack_texture_embs.items()
                }
                if deepstack_texture_embs is not None
                else None
            ),
            triangle_valid_mask=view_valid_mask,
            light_token_mask=(
                light_token_mask.repeat_interleave(num_views, dim=0)
                if light_token_mask is not None
                else None
            ),
        )
        return res.reshape(batch_size, num_views, *res.shape[1:])

    def _forward_varlen(
        self,
        tri_vpos_list, texture_patch_list, mvps, valid_mask, vns,
        rays_o, rays_d, tri_vpos_view_tf, light_strength,
        env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir,
        volume_density, volume_position, volume_rotation, volume_scale,
        volume_scattering, volume_absorption, volume_mask, tf32_view_tf
    ):
        """
        Forward pass using varlen packed sequence.
        
        This is the new varlen architecture path that:
        1. Encodes triangles
        2. Encodes volumes
        3. Processes environment map
        4. Builds packed sequence
        5. Encodes with packed sequence
        6. Decodes with packed context
        """
        if self.config.pe_type != 'rope_only_center':
            raise NotImplementedError("Varlen packed sequence only supports pe_type='rope_only_center'.")
        batch_size = tri_vpos_list.shape[0]
        device = tri_vpos_list.device
        
        # ========== 1. Encode triangles ==========
        # Use construct_seq to get triangle embeddings
        # We need to extract embeddings from the constructed sequence
        env_tokens_temp = None
        if env_latent_ldr is not None and env_latent_hdr is not None and env_lighting_strength is not None and env_ray_dir is not None and self.config.use_env_lighting:
            env_tokens_temp = self._process_env_map(env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir)

        # Position calculations in construct_seq use float32 for precision
        # (tri_vpos_list is already float32 from training script, no autocast needed)
        # Note: Removing autocast here to avoid slowing down texture/vn encoders
        seq_temp, valid_mask_padded_temp, tri_vpos_list_processed, deepstack_texture_embs = self.construct_seq(
            tri_vpos_list, texture_patch_list, mvps, valid_mask, vns, light_strength, env_tokens_temp
        )
        
        # Extract the complete fixed-token prefix. construct_seq lays it out as
        # [register/perceiver][env][view][triangles]. When environment tokens
        # are attention-sink tokens, packing the prefix as one batch-specific
        # fixed block preserves that exact checkpoint-visible order. Otherwise
        # env tokens are handed to SequenceBuilder for spatial mixing.
        num_view_tokens = 2 if self.config.use_view_token else 0
        num_register_tokens = self.skip_token_num - num_view_tokens
        env_token_num = env_tokens_temp.shape[1] if env_tokens_temp is not None else 0
        env_end_idx = num_register_tokens + env_token_num
        tri_start_idx = env_end_idx + num_view_tokens
        tri_embs = seq_temp[:, tri_start_idx:]  # [B, max_tri, D]

        if self.config.put_env_tokens_in_sink:
            fixed_tokens = seq_temp[:, :tri_start_idx]
            builder_env_tokens = None
        else:
            fixed_tokens = torch.cat(
                [seq_temp[:, :num_register_tokens], seq_temp[:, env_end_idx:tri_start_idx]],
                dim=1,
            )
            builder_env_tokens = env_tokens_temp
        
        # Extract triangle centers and fixed token positions from processed positions
        if self.config.pe_type == 'rope_only_center':
            # tri_vpos_list_processed is already [B, L, 3] (center positions)
            tri_centers = tri_vpos_list_processed[:, tri_start_idx:]  # [B, max_tri, 3]
            if self.config.put_env_tokens_in_sink:
                register_positions = tri_vpos_list_processed[:, :tri_start_idx]
                env_positions = None
            else:
                register_positions = torch.cat(
                    [
                        tri_vpos_list_processed[:, :num_register_tokens],
                        tri_vpos_list_processed[:, env_end_idx:tri_start_idx],
                    ],
                    dim=1,
                )
                env_positions = (
                    tri_vpos_list_processed[:, num_register_tokens:env_end_idx]
                    if env_token_num > 0 else None
                )
        else:
            # tri_vpos_list_processed is [B, L, 9] (full vertex positions)
            tri_centers = tri_vpos_list_processed[:, tri_start_idx:].reshape(
                batch_size, -1, 3, 3
            ).mean(dim=2)  # [B, max_tri, 3] - use center of triangle
            register_positions = None
            env_positions = None
        
        # Light mask: mirror original path behavior (only used when explicitly enabled)
        # If we are not putting light tokens first, keep original ordering (no split).
        if self.config.put_light_tokens_first:
            tri_light_mask = (light_strength > 0.).any(dim=-1) & valid_mask
        else:
            tri_light_mask = torch.zeros_like(valid_mask)

        # Store light mask for DeepStack (to skip injecting into light triangles)
        light_token_mask = tri_light_mask

        # ========== 2. Encode volumes (optional) ==========
        vol_embs = None
        vol_centers = None
        vol_mask = None
        if volume_density is not None and self.volume_encoder is not None:
            vol_embs, vol_centers = self.encode_volumes(
                volume_density, volume_position, volume_rotation,
                volume_scale, volume_scattering, volume_absorption
            )
            vol_mask = volume_mask

        # ========== 3. Build packed sequence ==========
        # Always use SequenceBuilder to keep a single packing path with all features.
        packed_batch = self.sequence_builder(
            tri_embs=tri_embs,
            tri_centers=tri_centers,
            tri_valid_mask=valid_mask,
            tri_light_mask=tri_light_mask,
            register_tokens=fixed_tokens,
            register_positions=register_positions,
            env_tokens=builder_env_tokens,
            env_positions=env_positions,
            summary_token_embed=self.transformer.summary_token_embed if self.config.add_summary_tokens else None,
            vol_embs=vol_embs,
            vol_centers=vol_centers,
            vol_valid_mask=vol_mask,
        )
        
        # ========== 4. Encode with packed sequence ==========
        rope_emb = self.transformer.rope_emb if hasattr(self.transformer, "rope_emb") else None

        encoded_batch = self.transformer.forward_packed(
            packed_batch,
            rope_emb=rope_emb,
            deepstack_texture_embs=deepstack_texture_embs,
            triangle_valid_mask=valid_mask,
            light_token_mask=light_token_mask,
        )

        # ========== 5. Decode with packed context ==========
        if self.config.decoding_method == 'view_transformer':
            assert rays_o is not None and rays_d is not None and tri_vpos_view_tf is not None
            num_views = rays_o.size(1)

            # Generate ray tokens
            ray_map = rays_d  # [B, V, H, W, 3]
            # Match padded path: ray tokens use the same dtype as context tokens.
            ray_map_pe = self.view_transformer.vdir_pe(ray_map).to(encoded_batch.packed_seq.dtype)
            from einops import rearrange
            ray_tokens = rearrange(
                ray_map_pe, 'b v (h1 p1) (w1 p2) c -> b v (h1 w1) (c p1 p2)',
                p1=self.config.patch_size, p2=self.config.patch_size
            )
            patch_h = ray_map_pe.size(2) // self.config.patch_size
            patch_w = ray_map_pe.size(3) // self.config.patch_size
            ray_tokens = self.view_transformer.ray_map_patch_token + self.view_transformer.ray_map_encoder_norm(
                self.view_transformer.ray_map_encoder(ray_tokens)
            )
            n_patches = ray_tokens.size(2)
            
            # Compute ray token positions
            # Use camera origin for each view
            camera_o = rays_o  # [B, V, 3]
            ray_token_pos = camera_o[:, :, None, :].expand(-1, -1, n_patches, -1)  # [B, V, n_patches, 3]
            
            # Flatten views for processing
            ray_tokens_flat = ray_tokens.view(batch_size * num_views, n_patches, -1)  # [B*V, n_patches, D]
            ray_token_pos_flat = ray_token_pos.view(batch_size * num_views, n_patches, 3)  # [B*V, n_patches, 3]
            
            if self.config.use_dpt_decoder:
                # For DPT, we need to get layer outputs from transformer
                out_features = self.view_transformer.transformer.forward_with_packed_context(
                    query=ray_tokens_flat,
                    context_packed=encoded_batch,
                    query_positions=ray_token_pos_flat,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    out_layers=self.view_transformer.out_layers,
                )
                # out_features is a list of lists: [[layer_output], ...] for each out_layer
                decoded_img = self.view_transformer.out_dpt(
                    out_features, patch_h, patch_w, patch_size=self.config.patch_size
                )
                decoded_img = self.view_transformer.out_proj_act(decoded_img)
                decoded_img = decoded_img.view(batch_size, num_views, *decoded_img.size()[1:])
            else:
                # Decode with packed context
                output = self.view_transformer.transformer.forward_with_packed_context(
                    query=ray_tokens_flat,
                    context_packed=encoded_batch,
                    query_positions=ray_token_pos_flat,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                # Reshape output back
                output = output.view(batch_size, num_views, n_patches, -1)
                decoded_patches = self.view_transformer.out_proj_act(
                    self.view_transformer.out_proj(output)
                )
                decoded_img = rearrange(
                    decoded_patches, 'b v (h1 w1) (c p1 p2) -> b v c (h1 p1) (w1 p2)',
                    p1=self.config.output_patch_size or self.config.patch_size,
                    p2=self.config.output_patch_size or self.config.patch_size,
                    h1=patch_h, w1=patch_w
                )
            
            return decoded_img
        else:
            raise NotImplementedError("Varlen packed sequence only supports view_transformer decoding method")

    def _forward_padded_unified(
        self,
        tri_vpos_list, texture_patch_list, mvps, valid_mask, vns,
        rays_o, rays_d, tri_vpos_view_tf, light_strength,
        env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir,
        volume_density, volume_position, volume_rotation, volume_scale,
        volume_scattering, volume_absorption, volume_mask, tf32_view_tf
    ):
        """
        Unified forward pass for all padded sequences.
        
        This builds the packed sequence via SequenceBuilder (with optional z-order/light handling),
        then unpacks it to a padded layout for the standard transformer path.
        All reordering, summarization, and light token handling is done by SequenceBuilder.
        """
        batch_size = tri_vpos_list.shape[0]
        device = tri_vpos_list.device

        # ========== 1. Process environment map ==========
        env_tokens_temp = None
        if (
            env_latent_ldr is not None and env_latent_hdr is not None
            and env_lighting_strength is not None and env_ray_dir is not None
            and self.config.use_env_lighting
        ):
            env_tokens_temp = self._process_env_map(
                env_latent_ldr, env_latent_hdr, env_lighting_strength, env_ray_dir
            )

        # ========== 2. Encode triangles ==========
        seq_temp, valid_mask_padded_temp, tri_vpos_list_processed, deepstack_texture_embs = self.construct_seq(
            tri_vpos_list, texture_patch_list, mvps, valid_mask, vns, light_strength, env_tokens_temp
        )

        # Keep the full construct_seq prefix intact. In particular, the
        # batch-specific BOS/MVP tokens must not be dropped during repacking.
        num_view_tokens = 2 if self.config.use_view_token else 0
        num_register_tokens = self.skip_token_num - num_view_tokens
        env_token_num = env_tokens_temp.shape[1] if env_tokens_temp is not None else 0
        env_end_idx = num_register_tokens + env_token_num
        tri_start_idx = env_end_idx + num_view_tokens
        tri_embs = seq_temp[:, tri_start_idx:]

        if self.config.put_env_tokens_in_sink:
            fixed_tokens = seq_temp[:, :tri_start_idx]
            builder_env_tokens = None
            fixed_token_num = tri_start_idx
        else:
            fixed_tokens = torch.cat(
                [seq_temp[:, :num_register_tokens], seq_temp[:, env_end_idx:tri_start_idx]],
                dim=1,
            )
            builder_env_tokens = env_tokens_temp
            fixed_token_num = self.skip_token_num

        # Extract triangle centers and fixed token positions
        if self.config.pe_type == 'rope_only_center':
            tri_centers = tri_vpos_list_processed[:, tri_start_idx:]  # [B, max_tri, 3]
            if self.config.put_env_tokens_in_sink:
                register_positions = tri_vpos_list_processed[:, :tri_start_idx]
                env_positions = None
            else:
                register_positions = torch.cat(
                    [
                        tri_vpos_list_processed[:, :num_register_tokens],
                        tri_vpos_list_processed[:, env_end_idx:tri_start_idx],
                    ],
                    dim=1,
                )
                env_positions = (
                    tri_vpos_list_processed[:, num_register_tokens:env_end_idx]
                    if env_token_num > 0 else None
                )
        else:
            tri_centers = tri_vpos_list_processed[:, tri_start_idx:].reshape(
                batch_size, -1, 3, 3
            ).mean(dim=2)  # [B, max_tri, 3]
            register_positions = None
            env_positions = None

        # ========== 3. Extract light mask ==========
        if self.config.put_light_tokens_first:
            tri_light_mask = (light_strength > 0.).any(dim=-1) & valid_mask
        else:
            tri_light_mask = torch.zeros_like(valid_mask)

        # Store light mask for DeepStack (to skip injecting into light triangles)
        light_token_mask = tri_light_mask

        # ========== 4. Encode volumes (optional) ==========
        vol_embs = None
        vol_centers = None
        vol_mask = None
        if volume_density is not None and self.volume_encoder is not None:
            vol_embs, vol_centers = self.encode_volumes(
                volume_density, volume_position, volume_rotation,
                volume_scale, volume_scattering, volume_absorption
            )
            vol_mask = volume_mask

        # ========== 5. Build packed sequence via SequenceBuilder ==========
        packed_batch = self.sequence_builder(
            tri_embs=tri_embs,
            tri_centers=tri_centers,
            tri_valid_mask=valid_mask,
            tri_light_mask=tri_light_mask,
            register_tokens=fixed_tokens,
            register_positions=register_positions,
            env_tokens=builder_env_tokens,
            env_positions=env_positions,
            summary_token_embed=self.transformer.summary_token_embed if self.config.add_summary_tokens else None,
            vol_embs=vol_embs,
            vol_centers=vol_centers,
            vol_valid_mask=vol_mask,
        )

        # ========== 6. Unpack to padded layout (vectorized) ==========
        max_seq_len = packed_batch.max_seq_len
        seq_unpacked = torch.zeros(
            batch_size, max_seq_len, packed_batch.packed_seq.shape[-1],
            device=packed_batch.packed_seq.device,
            dtype=packed_batch.packed_seq.dtype,
        )
        pos_unpacked = torch.zeros(
            batch_size, max_seq_len, 3,
            device=packed_batch.packed_pos.device,
            dtype=packed_batch.packed_pos.dtype,
        )
        valid_mask_padded = torch.zeros(
            batch_size, max_seq_len,
            device=packed_batch.packed_seq.device,
            dtype=torch.bool,
        )

        # Vectorized unpack: construct mask and indices
        seq_lens = (packed_batch.cu_seqlens[1:] - packed_batch.cu_seqlens[:-1])  # [B]
        mask_indices = torch.arange(max_seq_len, device=device).unsqueeze(0)  # [1, L]
        valid_mask_padded = mask_indices < seq_lens.unsqueeze(1)  # [B, L]

        # Construct source indices: each sample's start offset + 0..L-1
        sample_starts = packed_batch.cu_seqlens[:-1].unsqueeze(1)  # [B, 1]
        src_indices = (sample_starts + mask_indices)[valid_mask_padded]  # [Total_Valid_Tokens]

        # Vectorized copy
        seq_unpacked[valid_mask_padded] = packed_batch.packed_seq[src_indices]
        pos_unpacked[valid_mask_padded] = packed_batch.packed_pos[src_indices]

        # ========== 7. Align DeepStack texture embeddings ==========
        deepstack_texture_embs_aligned = None
        packed_tri_ids = getattr(packed_batch, 'packed_tri_ids', None)
        tri_ids_unpacked = None

        if packed_tri_ids is not None:
            # Preserve the mapping back to the caller's original padded
            # triangle order. SequenceBuilder may compact or spatially reorder
            # triangles, so decoding by sequence position is not sufficient.
            tri_ids_unpacked = torch.full(
                (batch_size, max_seq_len), -1,
                device=packed_batch.packed_seq.device,
                dtype=packed_tri_ids.dtype,
            )
            tri_ids_unpacked[valid_mask_padded] = packed_tri_ids[src_indices]

        if tri_ids_unpacked is not None and deepstack_texture_embs is not None:
            # Align DeepStack embeddings
            deepstack_texture_embs_aligned = {}
            for k, v in deepstack_texture_embs.items():
                # v: [B, max_tri, D]
                valid_idx_mask = tri_ids_unpacked >= 0
                safe_indices = tri_ids_unpacked.clone()
                safe_indices[~valid_idx_mask] = 0

                # Gather
                gathered = v.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, v.shape[-1]))

                # Mask out invalid positions AND light tokens
                if light_token_mask is not None:
                    is_light = light_token_mask.gather(1, safe_indices)
                    keep_mask = valid_idx_mask & (~is_light)
                else:
                    keep_mask = valid_idx_mask

                gathered = gathered * keep_mask.unsqueeze(-1).to(gathered.dtype)
                deepstack_texture_embs_aligned[k] = gathered
        else:
            deepstack_texture_embs_aligned = deepstack_texture_embs

        # ========== 8. Encode with padded transformer ==========
        # Note: SequenceBuilder has already handled reordering, summarization, and light tokens.
        # Transformer should NOT do these operations again.
        seq = self.transformer(
            seq_unpacked,
            src_key_padding_mask=valid_mask_padded,
            triangle_pos=pos_unpacked,
            num_register_tokens=fixed_token_num,
            env_token_num=0,
            out_layers=self.use_layers,
            light_token_mask=None,  # Light tokens are already in sink, no need for special handling
            deepstack_texture_embs=deepstack_texture_embs_aligned,
            triangle_valid_mask=None,  # Not needed since we have packed_tri_ids
            pre_ordered=True,  # Tell transformer that sequence is already ordered
            sink_lens=packed_batch.sink_lens,  # Pass sink lengths for attention mask generation
        )

        # ========== 9. Decode ==========
        if self.config.decoding_method == 'texture_patch':
            out_tex_patch_size = self.config.texture_decode_patch_size
            out = self.out_proj(
                seq.reshape(-1, self.config.latent_dim)
            ).reshape(seq.size(0), -1, 3, out_tex_patch_size, out_tex_patch_size)
            out = self.act(out)
            if tri_ids_unpacked is None:
                raise RuntimeError("Packed texture decoding requires original triangle IDs")

            # Restore the public padded API: [B, input_num_tri, 3, H, W].
            # Invalid input slots stay zero and every valid triangle is placed
            # back at its original ID, regardless of compaction or ordering.
            restored = torch.zeros(
                batch_size,
                tri_vpos_list.shape[1],
                3,
                out_tex_patch_size,
                out_tex_patch_size,
                device=out.device,
                dtype=out.dtype,
            )
            valid_tri_tokens = tri_ids_unpacked >= 0
            batch_indices = torch.arange(batch_size, device=out.device).unsqueeze(1).expand_as(tri_ids_unpacked)
            restored[
                batch_indices[valid_tri_tokens],
                tri_ids_unpacked[valid_tri_tokens],
            ] = out[valid_tri_tokens]
            return restored
        elif self.config.decoding_method == 'view_transformer':
            assert rays_o is not None and rays_d is not None and tri_vpos_view_tf is not None
            batch_size, num_views = rays_o.size(0), rays_o.size(1)
            if self.use_layers:
                seq = seq.repeat_interleave(num_views, dim=1)
            else:
                seq = seq.repeat_interleave(num_views, dim=0)
            rays_o = rays_o.view(-1, *rays_o.shape[2:])
            rays_d = rays_d.view(-1, *rays_d.shape[2:])
            valid_mask_padded_flat = valid_mask_padded.repeat_interleave(num_views, dim=0)
            pos_seq = pos_unpacked.repeat_interleave(num_views, dim=0)

            # Prepare DeepStack texture embeddings for decoder
            deepstack_texture_embs_decoder = None
            source_tex_embs = deepstack_texture_embs_aligned if deepstack_texture_embs_aligned is not None else deepstack_texture_embs

            if source_tex_embs is not None:
                deepstack_texture_embs_decoder = {
                    k: v.repeat_interleave(num_views, dim=0)
                    for k, v in source_tex_embs.items()
                }

            res = self.view_transformer(
                rays_o,
                rays_d,
                seq,
                pos_seq,
                valid_mask_padded_flat,
                tf32_mode=tf32_view_tf,
                use_indices=self.use_indices,
                deepstack_texture_embs=deepstack_texture_embs_decoder,
                triangle_valid_mask=None,
                light_token_mask=None,
                env_token_num=0,
            )
            res = res.view(batch_size, num_views, *res.size()[1:])
            return res
        else:
            raise NotImplementedError("Padded unified path only supports texture_patch and view_transformer decoding methods")
