import torch
import torch.nn as nn

from .config import RenderTransformerConfig
from .attention import TransformerDecoder
from .pe import NeRFEncoding
from .dpt import DPTHead

from einops import rearrange


class ViewTransformer(nn.Module):
    def __init__(self, config: RenderTransformerConfig):
        super().__init__()
        self.config = config
        # Set output_patch_size to patch_size if not specified
        self.output_patch_size = config.output_patch_size if config.output_patch_size is not None else config.patch_size
        # Set output_channel based on config
        if config.output_channel is not None:
            self.output_channel = config.output_channel
        else:
            self.output_channel = 4 if config.include_alpha else 3
        if config.pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=config.vertex_pe_num_freqs,
                include_input=True
            )
            self.pe_token_proj = nn.Linear(
                self.pos_pe.get_out_dim(),
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.token_pos_pe_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.token_pos_pe_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
            self.rope_dim = None
        elif config.pe_type == 'rope':
            self.rope_dim = min(config.vertex_pe_num_freqs, config.view_transformer_latent_dim // config.view_transformer_n_heads // 18 * 2)
            print(f'Using RoPE with dim {self.rope_dim}')
        elif config.pe_type == 'rope_only_center':
            self.rope_dim = min(config.intra_rope_pe_num_freqs, config.view_transformer_latent_dim // config.view_transformer_n_heads // 6 * 2)
            print(f'Using RoPE with dim {self.rope_dim}')
        else:
            raise ValueError(f"Unsupported positional encoding type: {config.pe_type}")

        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, config.view_transformer_latent_dim))
        if config.vdir_pe_type == 'nerf':
            self.vdir_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=config.vdir_num_freqs,
                include_input=True
            )
            self.ray_map_encoder = nn.Linear(
                self.vdir_pe.get_out_dim() * config.patch_size * config.patch_size,
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.ray_map_encoder_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.ray_map_encoder_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
        else:
            raise ValueError(f"Unsupported view direction positional encoding type: {config.vdir_pe_type}")


        self.transformer = TransformerDecoder(
            num_layers=self.config.view_transformer_n_layers,
            num_heads=self.config.view_transformer_n_heads,
            num_kv_heads=self.config.view_transformer_num_kv_heads,
            hidden_dim=self.config.view_transformer_latent_dim,
            ctx_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.view_transformer_ffn_hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            rope_type=self.config.rope_type,
            rope_double_max_freq=self.config.rope_double_max_freq,
            qk_norm=self.config.qk_norm,
            bias=self.config.bias,
            include_self_attn=self.config.view_transformer_include_self_attn,
            self_attn_before_cross=self.config.view_transformer_self_attn_before_cross,
            use_swin_attn=self.config.view_transformer_use_swin_attn,
            use_triangle_ordering=self.config.view_transformer_use_triangle_ordering,
            triangle_ordering_order=self.config.view_transformer_triangle_ordering_order,
            # DeepStack config - let decoder create its own projectors internally
            deepstack_injection_map=self.config.deepstack_decoder_injection_map,
            deepstack_feature_dim=(
                self.config.texture_channels *
                self.config.texture_encode_patch_size *
                self.config.texture_encode_patch_size
            ) if self.config.deepstack_decoder_injection_map is not None else None,
        )
        if not config.use_dpt_decoder:
            # self.norm_last = nn.RMSNorm(self.config.view_transformer_latent_dim, eps=1e-6) if self.config.norm_type == 'rms_norm' else nn.LayerNorm(self.config.view_transformer_latent_dim)
            self.out_proj = nn.Linear(self.config.view_transformer_latent_dim, self.output_patch_size * self.output_patch_size * self.output_channel)
        else:
            self.out_dpt = DPTHead(
                in_channels=self.config.view_transformer_latent_dim,
                features=self.config.dpt_features,
                use_clstoken=False,
                out_channels=self.config.dpt_out_channels, 
                out_dim=4 if config.include_alpha else 3
            )
            self.out_layers = list(range(self.config.view_transformer_n_layers - 4, self.config.view_transformer_n_layers)) if self.config.dpt_out_layers is None else self.config.dpt_out_layers
            print(f"Using DPT decoder with layers: {self.out_layers}")
        self.out_proj_act = nn.ELU(alpha=1e-3) if self.config.latent_vae_model_id is None else nn.Identity()

    def forward(self, camera_o, ray_map, tri_tokens, tri_pos, valid_mask, tf32_mode=False, use_indices=[], deepstack_texture_embs=None, triangle_valid_mask=None, light_token_mask=None, env_token_num=0):
        """
        Cross attention between ray map and triangle tokens.

        Args:
            camera_o (torch.Tensor): (B, 3)
            ray_map (torch.Tensor): (B, H, W, 3)
            tri_tokens (torch.Tensor): (B, N_TRIS, D)
            tri_pos (torch.Tensor): (B, N_TRIS, 9)
            valid_mask (torch.Tensor): (B, N_TRIS)
            tf32_mode (bool): whether to use tf32 mode
            deepstack_texture_embs (dict): DeepStack texture features
            triangle_valid_mask (torch.Tensor): [B, max_tri] for masking invalid triangles
            light_token_mask (torch.Tensor): [B, max_tri] True for light triangles
        Returns:
            decoded_img: (B, 3, H, W)
        """

        # query sequence
        ray_map = self.vdir_pe(ray_map).to(tri_tokens.dtype)
        ray_tokens = rearrange(ray_map, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=self.config.patch_size, p2=self.config.patch_size)
        patch_h = ray_map.size(1) // self.config.patch_size
        patch_w = ray_map.size(2) // self.config.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        n_patches = ray_tokens.size(1)
        ray_token_pos = camera_o[:, None].repeat(1, n_patches, 3 if self.config.pe_type != 'rope_only_center' else 1)  # [B, N_PATCHES, 3 or 1]

        # positional encoding if use 'nerf' pe
        if self.config.pe_type == 'nerf':
            ray_tokens = ray_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(ray_token_pos)))
            tri_tokens = tri_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(tri_pos)))

        # do per-ray attention
        if self.config.use_dpt_decoder:
            with torch.autocast(device_type="cuda", dtype=torch.float32 if tf32_mode else torch.bfloat16):
                out_features = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, out_layers=self.out_layers, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w, use_indices=use_indices, num_register_tokens=self.config.num_register_tokens, env_token_num=env_token_num, deepstack_texture_embs=deepstack_texture_embs, triangle_valid_mask=triangle_valid_mask, light_token_mask=light_token_mask)
            decoded_img = self.out_dpt(out_features, patch_h, patch_w, patch_size=self.config.patch_size)
            return self.out_proj_act(decoded_img)
        else:
            seq = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w, use_indices=use_indices, num_register_tokens=self.config.num_register_tokens, env_token_num=env_token_num, deepstack_texture_embs=deepstack_texture_embs, triangle_valid_mask=triangle_valid_mask, light_token_mask=light_token_mask)  # [B, N_PATCHES, D]
            decoded_patches = self.out_proj_act(self.out_proj(seq))  # [B, N_PATCHES, P*P*C]
            # Reconstruct image from patches using output_patch_size
            decoded_img = rearrange(decoded_patches, 'b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)', p1=self.output_patch_size, p2=self.output_patch_size, h1=patch_h, w1=patch_w)
            return decoded_img
