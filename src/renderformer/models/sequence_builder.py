import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .packed_batch import PackedBatch
from .ordering.default import encode_mask_batch


class SequenceBuilder(nn.Module):
    """
    Build packed sequence from triangle and volume embeddings.

    Flow:
        1. Separate light_geo vs normal_geo
        2. Concat normal_tris + volumes -> normal_geo
        3. Spatial sort normal_geo (z-order, hilbert, etc.)
        4. Generate summary tokens from sorted normal_geo
        5. Pack: [register][env][summary][light_geo][sorted_normal_geo]
    """

    def __init__(
        self,
        latent_dim: int = 1024,
        summary_block_size: int = 64,
        z_order_depth: int = 16,
        add_summary_tokens: bool = True,
        use_z_order: bool = True,
        put_light_tokens_in_sink: bool = True,
        put_env_tokens_in_sink: bool = True,
        ordering_order: Optional[str] = None,
    ):
        """
        Args:
            ordering_order: Ordering type for spatial sorting.
                           Options: 'z', 'z-trans', 'hilbert', 'hilbert-trans', or None (no sorting).
                           If None, will use 'z' if use_z_order=True, otherwise no sorting.
            put_env_tokens_in_sink: If True (default), env tokens are placed in sink region.
                           If False, env tokens are mixed with normal_geo and sorted together.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.summary_block_size = summary_block_size
        self.z_order_depth = z_order_depth
        self.add_summary_tokens = add_summary_tokens
        self.use_z_order = use_z_order
        self.put_light_tokens_in_sink = put_light_tokens_in_sink
        self.put_env_tokens_in_sink = put_env_tokens_in_sink
        
        # Determine ordering order
        if ordering_order is not None:
            assert ordering_order in {'z', 'z-trans', 'hilbert', 'hilbert-trans'}, \
                f"ordering_order must be one of {{'z', 'z-trans', 'hilbert', 'hilbert-trans'}}, got {ordering_order}"
            self.ordering_order = ordering_order
        elif use_z_order:
            # Default to 'z' if use_z_order is True but no explicit order given
            self.ordering_order = 'z'
        else:
            # No ordering
            self.ordering_order = None

    def forward(
        self,
        # Triangle inputs (padded)
        tri_embs: torch.Tensor,          # [B, max_tri, D]
        tri_centers: torch.Tensor,       # [B, max_tri, 3]
        tri_valid_mask: torch.Tensor,    # [B, max_tri] bool
        tri_light_mask: torch.Tensor,    # [B, max_tri] bool

        # Fixed tokens
        register_tokens: torch.Tensor,   # [n_reg, D] or [B, n_reg, D]
        register_positions: Optional[torch.Tensor] = None, # [n_reg, 3], [B, n_reg, 3], or None
        env_tokens: Optional[torch.Tensor] = None, # [B, n_env, D] or None
        env_positions: Optional[torch.Tensor] = None, # [B, n_env, 3] or None
        summary_token_embed: Optional[torch.Tensor] = None, # [1, 1, D] or None

        # Volume inputs (padded, optional - can be None)
        vol_embs: Optional[torch.Tensor] = None,          # [B, max_vol, D] or None
        vol_centers: Optional[torch.Tensor] = None,       # [B, max_vol, 3] or None
        vol_valid_mask: Optional[torch.Tensor] = None,    # [B, max_vol] bool or None
    ) -> PackedBatch:
        """
        Build packed batch from inputs.

        Returns:
            PackedBatch with packed_seq, packed_pos, cu_seqlens, sink_lens
        """
        B, max_tri, D = tri_embs.shape
        device = tri_embs.device

        # New: Track original triangle indices
        tri_ids = torch.arange(max_tri, device=device).unsqueeze(0).expand(B, -1)  # [B, max_tri]

        # ========== Step 1: Separate light_geo vs normal_geo ==========
        normal_tri_mask = tri_valid_mask & ~tri_light_mask  # [B, max_tri]
        light_tri_mask = tri_valid_mask & tri_light_mask    # [B, max_tri]

        # ========== Step 2: Build normal_geo (tris + vols + optionally env) ==========
        # If put_env_tokens_in_sink=False, include env tokens in normal_geo for sorting
        if not self.put_env_tokens_in_sink and env_tokens is not None:
            # Compute env positions first if needed
            _, env_pos_for_geo = self._compute_fixed_positions(
                register_tokens, register_positions,
                env_tokens, env_positions,
                tri_centers, tri_valid_mask,
                vol_centers, vol_valid_mask,
            )
            build_env_embs = env_tokens
            build_env_centers = env_pos_for_geo
        else:
            build_env_embs = None
            build_env_centers = None

        normal_geo_embs, normal_geo_centers, normal_geo_mask, n_normal_geo, is_triangle_mask, normal_geo_tri_ids, is_env_mask = \
            self._build_normal_geo(
                tri_embs, tri_centers, normal_tri_mask,
                vol_embs, vol_centers, vol_valid_mask,
                tri_ids,
                env_embs=build_env_embs, env_centers=build_env_centers,
            )
        # normal_geo_*: [B, max_normal_geo, ...]

        # ========== Step 3: Spatial sort normal_geo ==========
        sorted_normal_geo_embs, sorted_normal_geo_centers, sorted_is_triangle_mask, sorted_normal_geo_tri_ids, sorted_is_env_mask = self._spatial_sort(
            normal_geo_embs, normal_geo_centers, normal_geo_mask, is_triangle_mask, normal_geo_tri_ids, is_env_mask
        )

        # ========== Step 4: Generate summary tokens ==========
        if self.add_summary_tokens:
            summary_embs, summary_centers, n_summary = self._generate_summary(
                sorted_normal_geo_embs, sorted_normal_geo_centers,
                normal_geo_mask, n_normal_geo, summary_token_embed
            )
        else:
            # No summary tokens
            summary_embs = torch.zeros(B, 0, self.latent_dim, device=device, dtype=tri_embs.dtype)
            summary_centers = torch.zeros(B, 0, 3, device=device, dtype=torch.float32)  # Always float32 for positions
            n_summary = torch.zeros(B, dtype=torch.long, device=device)
        # summary_*: [B, max_summary, ...]

        # ========== Step 5: Extract light_geo ==========
        light_geo_embs, light_geo_centers, n_light, light_geo_tri_ids = self._extract_light_geo(
            tri_embs, tri_centers, light_tri_mask, tri_ids
        )

        # ========== Step 6: Compute register/env positions ==========
        reg_positions, env_pos = self._compute_fixed_positions(
            register_tokens, register_positions,
            env_tokens, env_positions,
            tri_centers, tri_valid_mask,
            vol_centers, vol_valid_mask,
        )

        # ========== Step 7: Pack everything ==========
        return self._pack_batch(
            register_tokens=register_tokens,
            reg_positions=reg_positions,
            env_tokens=env_tokens if self.put_env_tokens_in_sink else None,  # Only pass if in sink
            env_positions=env_pos if self.put_env_tokens_in_sink else None,
            summary_embs=summary_embs,
            summary_centers=summary_centers,
            n_summary=n_summary,
            light_geo_embs=light_geo_embs,
            light_geo_centers=light_geo_centers,
            n_light=n_light,
            light_geo_tri_ids=light_geo_tri_ids,
            sorted_normal_geo_embs=sorted_normal_geo_embs,
            sorted_normal_geo_centers=sorted_normal_geo_centers,
            n_normal_geo=n_normal_geo,
            normal_geo_mask=normal_geo_mask,
            sorted_is_triangle_mask=sorted_is_triangle_mask,
            sorted_normal_geo_tri_ids=sorted_normal_geo_tri_ids,
            sorted_is_env_mask=sorted_is_env_mask,
        )

    def _build_normal_geo(
        self,
        tri_embs, tri_centers, normal_tri_mask,
        vol_embs, vol_centers, vol_valid_mask,
        tri_ids,
        env_embs=None, env_centers=None,
    ):
        """
        Concatenate normal triangles, volumes, and optionally env tokens into one tensor.

        Uses vectorized operations to avoid Python loops over batch dimension.
        If vol_embs is None, only triangles are used.
        If env_embs is provided, env tokens are included and will be sorted with geo.

        Returns:
            normal_geo_embs: [B, max_normal_geo, D]
            normal_geo_centers: [B, max_normal_geo, 3]
            normal_geo_mask: [B, max_normal_geo]
            n_normal_geo: [B] number of valid normal_geo per sample
            is_triangle_mask: [B, max_normal_geo] - True for triangles, False for volumes/env
            normal_geo_ids: [B, max_normal_geo] - original triangle index or -1
            is_env_mask: [B, max_normal_geo] - True for env tokens, False otherwise
        """
        B, max_tri, D = tri_embs.shape
        device = tri_embs.device
        emb_dtype = tri_embs.dtype
        center_dtype = tri_centers.dtype  # centers may have different dtype (e.g., float32 vs bfloat16)

        n_normal_tris = normal_tri_mask.sum(dim=1)  # [B]

        # Handle optional volumes
        if vol_embs is not None and vol_valid_mask is not None:
            n_vols = vol_valid_mask.sum(dim=1)  # [B]
        else:
            n_vols = torch.zeros(B, dtype=torch.long, device=device)

        # Handle optional env tokens
        if env_embs is not None:
            n_env = env_embs.shape[1]  # fixed size, all valid
        else:
            n_env = 0

        n_normal_geo = n_normal_tris + n_vols + n_env  # [B]

        max_normal_geo = n_normal_geo.max().item() if n_normal_geo.max().item() > 0 else 1

        if max_normal_geo == 0:
            # Edge case: no normal geometry
            return (
                torch.zeros(B, 1, D, device=device, dtype=emb_dtype),
                torch.zeros(B, 1, 3, device=device, dtype=torch.float32),  # Always float32 for positions
                torch.zeros(B, 1, dtype=torch.bool, device=device),
                n_normal_geo,
                torch.zeros(B, 1, dtype=torch.bool, device=device),
                torch.full((B, 1), -1, device=device, dtype=tri_ids.dtype),
                torch.zeros(B, 1, dtype=torch.bool, device=device),  # is_env_mask
            )

        # Build combined tensors step by step
        # Start with triangles
        combined_embs = tri_embs
        combined_centers = tri_centers
        combined_mask = normal_tri_mask
        combined_ids = tri_ids
        combined_is_tri = torch.ones(B, max_tri, dtype=torch.bool, device=device)
        combined_is_env = torch.zeros(B, max_tri, dtype=torch.bool, device=device)

        # Add volumes if present
        if vol_embs is not None and vol_valid_mask is not None:
            combined_embs = torch.cat([combined_embs, vol_embs], dim=1)
            combined_centers = torch.cat([combined_centers, vol_centers], dim=1)
            combined_mask = torch.cat([combined_mask, vol_valid_mask], dim=1)
            combined_ids = torch.cat([
                combined_ids,
                torch.full((B, vol_embs.shape[1]), -1, dtype=tri_ids.dtype, device=device)
            ], dim=1)
            combined_is_tri = torch.cat([
                combined_is_tri,
                torch.zeros(B, vol_embs.shape[1], dtype=torch.bool, device=device)
            ], dim=1)
            combined_is_env = torch.cat([
                combined_is_env,
                torch.zeros(B, vol_embs.shape[1], dtype=torch.bool, device=device)
            ], dim=1)

        # Add env tokens if present
        if env_embs is not None:
            env_valid_mask = torch.ones(B, n_env, dtype=torch.bool, device=device)
            combined_embs = torch.cat([combined_embs, env_embs], dim=1)
            combined_centers = torch.cat([combined_centers, env_centers], dim=1)
            combined_mask = torch.cat([combined_mask, env_valid_mask], dim=1)
            combined_ids = torch.cat([
                combined_ids,
                torch.full((B, n_env), -1, dtype=tri_ids.dtype, device=device)
            ], dim=1)
            combined_is_tri = torch.cat([
                combined_is_tri,
                torch.zeros(B, n_env, dtype=torch.bool, device=device)
            ], dim=1)
            combined_is_env = torch.cat([
                combined_is_env,
                torch.ones(B, n_env, dtype=torch.bool, device=device)
            ], dim=1)

        # For each sample, compute destination indices for valid tokens
        # We need to pack valid tokens to the left within each sample
        # Use cumsum trick: cumsum of mask gives 1-indexed positions for valid tokens
        cumsum = combined_mask.long().cumsum(dim=1)
        # Destination index is cumsum - 1 (0-indexed), but only for valid positions
        dest_indices = (cumsum - 1).clamp(min=0)

        # Create output tensors
        normal_geo_embs = torch.zeros(B, max_normal_geo, D, device=device, dtype=emb_dtype)
        normal_geo_centers = torch.zeros(B, max_normal_geo, 3, device=device, dtype=center_dtype)
        normal_geo_ids = torch.full((B, max_normal_geo), -1, device=device, dtype=tri_ids.dtype)

        # Scatter valid tokens to their packed positions
        # Expand dest_indices for gather/scatter
        dest_indices_embs = dest_indices.unsqueeze(-1).expand(-1, -1, D)
        dest_indices_centers = dest_indices.unsqueeze(-1).expand(-1, -1, 3)

        # Only scatter where mask is True
        mask_expanded_embs = combined_mask.unsqueeze(-1).expand(-1, -1, D)
        mask_expanded_centers = combined_mask.unsqueeze(-1).expand(-1, -1, 3)

        # Use scatter_add to avoid invalid positions overwriting valid tokens.
        normal_geo_embs.scatter_add_(1, dest_indices_embs, combined_embs * mask_expanded_embs.to(emb_dtype))
        normal_geo_centers.scatter_add_(1, dest_indices_centers, combined_centers * mask_expanded_centers.to(center_dtype))

        # Build output mask: positions 0 to n_normal_geo-1 are valid for each sample
        position_indices = torch.arange(max_normal_geo, device=device).unsqueeze(0).expand(B, -1)  # [B, max_normal_geo]
        normal_geo_mask = position_indices < n_normal_geo.unsqueeze(1)  # [B, max_normal_geo]

        # Scatter type mask (DeepStack)
        is_triangle_mask = torch.zeros(B, max_normal_geo, dtype=torch.bool, device=device)
        is_triangle_mask.scatter_(1, dest_indices, (combined_is_tri & combined_mask))

        # Scatter env mask
        is_env_mask = torch.zeros(B, max_normal_geo, dtype=torch.bool, device=device)
        is_env_mask.scatter_(1, dest_indices, (combined_is_env & combined_mask))

        # Scatter IDs using safe redirect
        # Redirect invalid indices to a dummy position at the end
        extended_ids = torch.full((B, max_normal_geo + 1), -1, device=device, dtype=tri_ids.dtype)
        safe_dest_indices = dest_indices.clone()
        safe_dest_indices[~combined_mask] = max_normal_geo
        extended_ids.scatter_(1, safe_dest_indices, combined_ids)
        normal_geo_ids = extended_ids[:, :max_normal_geo]

        return normal_geo_embs, normal_geo_centers, normal_geo_mask, n_normal_geo, is_triangle_mask, normal_geo_ids, is_env_mask

    def _spatial_sort(self, embs, centers, mask, is_triangle_mask, tri_ids, is_env_mask):
        """
        Sort embeddings, centers, and type mask by spatial ordering of centers.

        Args:
            embs: [B, L, D]
            centers: [B, L, 3]
            mask: [B, L]
            is_triangle_mask: [B, L]
            tri_ids: [B, L]
            is_env_mask: [B, L]

        Returns:
            sorted_embs: [B, L, D]
            sorted_centers: [B, L, 3]
            sorted_is_triangle_mask: [B, L]
            sorted_tri_ids: [B, L]
            sorted_is_env_mask: [B, L]
        """
        if self.ordering_order is None:
            return embs, centers, is_triangle_mask, tri_ids, is_env_mask

        order, _ = encode_mask_batch(centers, mask=mask, order=self.ordering_order, depth=self.z_order_depth)
        # order: [B, L]

        B, L, D = embs.shape
        sorted_embs = embs.gather(1, order.unsqueeze(-1).expand(-1, -1, D))
        sorted_centers = centers.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
        sorted_is_tri = is_triangle_mask.gather(1, order)
        sorted_tri_ids = tri_ids.gather(1, order)
        sorted_is_env = is_env_mask.gather(1, order)

        return sorted_embs, sorted_centers, sorted_is_tri, sorted_tri_ids, sorted_is_env

    def _generate_summary(self, embs, centers, mask, n_valid, summary_token_embed):
        """
        Generate summary tokens via masked average pooling.

        Args:
            embs: [B, L, D]
            centers: [B, L, 3]
            mask: [B, L]
            n_valid: [B] number of valid tokens per sample
            summary_token_embed: [1, 1, D] learnable summary token embedding

        Returns:
            summary_embs: [B, max_summary, D]
            summary_centers: [B, max_summary, 3]
            n_summary: [B] number of summary tokens per sample
        """
        B, L, D = embs.shape
        block_size = self.summary_block_size
        n_blocks = L // block_size

        if n_blocks == 0:
            # Not enough tokens for even one summary
            return (
                torch.zeros(B, 0, D, device=embs.device, dtype=embs.dtype),
                torch.zeros(B, 0, 3, device=embs.device, dtype=torch.float32),  # Always float32 for positions
                torch.zeros(B, dtype=torch.long, device=embs.device),
            )

        # Reshape to blocks
        embs_blocks = embs[:, :n_blocks * block_size].reshape(B, n_blocks, block_size, D)
        centers_blocks = centers[:, :n_blocks * block_size].reshape(B, n_blocks, block_size, 3)
        mask_blocks = mask[:, :n_blocks * block_size].reshape(B, n_blocks, block_size)

        # Masked average
        mask_expanded = mask_blocks.unsqueeze(-1).float()  # [B, n_blocks, block_size, 1]
        block_count = mask_blocks.sum(dim=2, keepdim=True).clamp(min=1).float()  # [B, n_blocks, 1]

        summary_embs = (embs_blocks * mask_expanded).sum(dim=2) / block_count  # [B, n_blocks, D]
        summary_centers = (centers_blocks * mask_expanded).sum(dim=2) / block_count  # [B, n_blocks, 3]

        # Add learnable summary token embedding
        summary_embs = summary_embs + summary_token_embed

        # Number of valid summary tokens per sample
        n_summary = n_valid // block_size  # [B]

        return summary_embs, summary_centers, n_summary

    def _extract_light_geo(self, tri_embs, tri_centers, light_tri_mask, tri_ids):
        """
        Extract light geometry tokens using vectorized operations.

        Returns:
            light_geo_embs: [B, max_light, D]
            light_geo_centers: [B, max_light, 3]
            n_light: [B]
            light_geo_ids: [B, max_light]
        """
        B, max_tri, D = tri_embs.shape
        device = tri_embs.device
        emb_dtype = tri_embs.dtype
        center_dtype = tri_centers.dtype  # centers may have different dtype

        n_light = light_tri_mask.sum(dim=1)  # [B]
        max_light = n_light.max().item() if n_light.max().item() > 0 else 0

        if max_light == 0:
            return (
                torch.zeros(B, 0, D, device=device, dtype=emb_dtype),
                torch.zeros(B, 0, 3, device=device, dtype=torch.float32),  # Always float32 for positions
                n_light,
                torch.zeros(B, 0, dtype=tri_ids.dtype, device=device),
            )

        # Vectorized packing using cumsum trick (same as _build_normal_geo)
        cumsum = light_tri_mask.long().cumsum(dim=1)  # [B, max_tri]
        dest_indices = (cumsum - 1).clamp(min=0)  # [B, max_tri]

        # Create output tensors
        light_geo_embs = torch.zeros(B, max_light, D, device=device, dtype=emb_dtype)
        light_geo_centers = torch.zeros(B, max_light, 3, device=device, dtype=center_dtype)
        light_geo_ids = torch.full((B, max_light), -1, device=device, dtype=tri_ids.dtype)

        # Expand indices for scatter
        dest_indices_embs = dest_indices.unsqueeze(-1).expand(-1, -1, D)
        dest_indices_centers = dest_indices.unsqueeze(-1).expand(-1, -1, 3)

        # Mask for scatter (only scatter valid light tokens)
        mask_expanded_embs = light_tri_mask.unsqueeze(-1).expand(-1, -1, D)
        mask_expanded_centers = light_tri_mask.unsqueeze(-1).expand(-1, -1, 3)

        # Use scatter_add to avoid invalid positions overwriting valid tokens.
        light_geo_embs.scatter_add_(1, dest_indices_embs, tri_embs * mask_expanded_embs.to(emb_dtype))
        light_geo_centers.scatter_add_(1, dest_indices_centers, tri_centers * mask_expanded_centers.to(center_dtype))
        
        # Scatter IDs using safe redirect
        extended_ids = torch.full((B, max_light + 1), -1, device=device, dtype=tri_ids.dtype)
        safe_dest_indices = dest_indices.clone()
        safe_dest_indices[~light_tri_mask] = max_light
        extended_ids.scatter_(1, safe_dest_indices, tri_ids)
        light_geo_ids = extended_ids[:, :max_light]

        return light_geo_embs, light_geo_centers, n_light, light_geo_ids

    def _compute_fixed_positions(
        self,
        register_tokens, register_positions,
        env_tokens, env_positions,
        tri_centers, tri_valid_mask,
        vol_centers, vol_valid_mask,
    ):
        """
        Compute positions for register and env tokens.
        If not provided, use mean of all geometry centers.
        Volumes are optional (can be None).
        """
        B = tri_centers.shape[0]
        device = tri_centers.device
        n_reg = register_tokens.shape[-2]

        # Compute mean geometry center (volumes are optional)
        if vol_centers is not None and vol_valid_mask is not None:
            all_centers = torch.cat([tri_centers, vol_centers], dim=1)
            all_mask = torch.cat([tri_valid_mask, vol_valid_mask], dim=1)
        else:
            all_centers = tri_centers
            all_mask = tri_valid_mask

        masked_centers = all_centers * all_mask.unsqueeze(-1).float()
        mean_center = masked_centers.sum(dim=1) / all_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        # mean_center: [B, 3]

        # Register positions
        if register_positions is None:
            reg_positions = mean_center[:, None, :].expand(-1, n_reg, -1)  # [B, n_reg, 3]
        else:
            if register_positions.ndim == 3:
                reg_positions = register_positions  # [B, n_reg, 3]
            else:
                reg_positions = register_positions[None, :, :].expand(B, -1, -1)  # [B, n_reg, 3]

        # Env positions
        if env_tokens is not None:
            n_env = env_tokens.shape[1]
            if env_positions is None:
                env_pos = mean_center[:, None, :].expand(-1, n_env, -1)  # [B, n_env, 3]
            else:
                env_pos = env_positions
        else:
            env_pos = None

        return reg_positions, env_pos

    def _pack_batch(
        self,
        register_tokens,
        reg_positions,
        env_tokens,
        env_positions,
        summary_embs,
        summary_centers,
        n_summary,
        light_geo_embs,
        light_geo_centers,
        n_light,
        light_geo_tri_ids,
        sorted_normal_geo_embs,
        sorted_normal_geo_centers,
        n_normal_geo,
        normal_geo_mask,
        sorted_is_triangle_mask,
        sorted_normal_geo_tri_ids,
        sorted_is_env_mask,
    ) -> PackedBatch:
        """
        Pack all components into a single PackedBatch.

        Uses vectorized operations where possible. The final packing still requires
        some sequential logic due to variable sequence lengths, but individual
        component copying is vectorized.

        Note: When put_env_tokens_in_sink=False, env_tokens/env_positions will be None
        and env tokens are already included in sorted_normal_geo_*.
        """
        B = reg_positions.shape[0]
        D = register_tokens.shape[-1]
        device = register_tokens.device
        # Use summary_embs dtype as reference (comes from tri_embs which has trainer's dtype)
        # register_tokens may be float32 from nn.Parameter, but we want bf16 for embeddings
        emb_dtype = summary_embs.dtype
        # IMPORTANT: Always use float32 for positions to maintain precision in RoPE and spatial operations
        pos_dtype = torch.float32
        n_reg = register_tokens.shape[-2]
        n_env = env_tokens.shape[1] if env_tokens is not None else 0

        # Compute sequence lengths and cumulative offsets
        # seq_len per sample = n_reg + n_env + n_summary + n_light + n_normal_geo
        # Note: when env tokens are mixed into normal_geo, n_env=0 here but they're counted in n_normal_geo
        seq_lens = n_reg + n_env + n_summary + n_light + n_normal_geo  # [B]
        total_tokens = seq_lens.sum().item()

        # Build cu_seqlens
        cu_seqlens = torch.zeros(B + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = seq_lens.cumsum(dim=0).to(torch.int32)

        # Compute sink_lens
        # When env tokens are in sink: sink = reg + env + summary (+ light if put_light_tokens_in_sink)
        # When env tokens are NOT in sink: sink = reg + summary (+ light if put_light_tokens_in_sink)
        if self.put_light_tokens_in_sink:
            sink_lens = (n_reg + n_env + n_summary + n_light).to(torch.int32)
        else:
            sink_lens = (n_reg + n_env + n_summary).to(torch.int32)

        # Allocate output tensors
        packed_seq = torch.zeros(total_tokens, D, device=device, dtype=emb_dtype)
        packed_pos = torch.zeros(total_tokens, 3, device=device, dtype=pos_dtype)
        packed_tri_ids = torch.full((total_tokens,), -1, device=device, dtype=sorted_normal_geo_tri_ids.dtype)

        # For each sample, compute start offsets for each component
        # Layout: [register][env][summary][light/normal_geo depending on config]
        base_offsets = cu_seqlens[:-1].long()  # [B] - start of each sample

        # Vectorized copy for fixed-length components (register, env)
        # Build index tensors for register tokens
        # For each sample b, register tokens go to positions [base_offsets[b] : base_offsets[b] + n_reg]
        reg_indices = base_offsets.unsqueeze(1) + torch.arange(n_reg, device=device)  # [B, n_reg]
        reg_indices_flat = reg_indices.view(-1)  # [B * n_reg]

        # Expand shared fixed tokens or preserve batch-specific ones (for example,
        # an MVP token) before flattening. Cast to the sequence dtype because
        # learned parameters are commonly float32 under mixed precision.
        if register_tokens.ndim == 2:
            register_tokens_batched = register_tokens.unsqueeze(0).expand(B, -1, -1)
        elif register_tokens.ndim == 3:
            if register_tokens.shape[0] != B:
                raise ValueError(
                    "Batch-specific register tokens must match the geometry batch size: "
                    f"{register_tokens.shape[0]} != {B}"
                )
            register_tokens_batched = register_tokens
        else:
            raise ValueError(
                "register_tokens must have shape [n_reg, D] or [B, n_reg, D], "
                f"got {tuple(register_tokens.shape)}"
            )
        packed_seq.index_copy_(
            0,
            reg_indices_flat,
            register_tokens_batched.reshape(-1, D).to(emb_dtype),
        )
        # Ensure positions are float32
        packed_pos.index_copy_(0, reg_indices_flat, reg_positions.float().reshape(-1, 3))
        # packed_tri_ids are already -1

        # Vectorized copy for env tokens
        if env_tokens is not None and n_env > 0:
            env_indices = base_offsets.unsqueeze(1) + n_reg + torch.arange(n_env, device=device)  # [B, n_env]
            env_indices_flat = env_indices.view(-1)  # [B * n_env]
            # Ensure env_tokens has the same dtype as packed_seq
            env_tokens_typed = env_tokens.to(emb_dtype).reshape(-1, D)
            packed_seq.index_copy_(0, env_indices_flat, env_tokens_typed)
            # Ensure positions are float32
            packed_pos.index_copy_(0, env_indices_flat, env_positions.float().reshape(-1, 3))
            # packed_tri_ids are already -1

        # For variable-length components (summary, light, normal_geo)
        # OPTIMIZATION: Use vectorized scatter operations instead of Python loops

        # Compute maximum sizes needed
        max_summary = n_summary.max() if n_summary.numel() > 0 and n_summary.max() > 0 else 0
        max_light = n_light.max() if n_light.numel() > 0 and n_light.max() > 0 else 0
        max_normal = n_normal_geo.max() if n_normal_geo.numel() > 0 and n_normal_geo.max() > 0 else 0

        # Build position masks and indices using vectorized operations
        summary_offset = n_reg + n_env

        # Copy summary tokens (if any)
        if max_summary > 0:
            # Create position indices: [B, max_summary]
            summary_pos_in_batch = torch.arange(max_summary, device=device).unsqueeze(0).expand(B, -1)
            summary_valid = summary_pos_in_batch < n_summary.unsqueeze(1)  # [B, max_summary]
            summary_global_pos = base_offsets.unsqueeze(1) + summary_offset + summary_pos_in_batch  # [B, max_summary]

            # Scatter summary tokens
            summary_indices_seq = summary_global_pos.unsqueeze(-1).expand(-1, -1, D)[summary_valid]  # [N_valid, D]
            summary_indices_pos = summary_global_pos.unsqueeze(-1).expand(-1, -1, 3)[summary_valid]  # [N_valid, 3]
            summary_values_seq = summary_embs[summary_valid]  # [N_valid, D]
            summary_values_pos = summary_centers[summary_valid].float()  # [N_valid, 3]

            packed_seq.scatter_(0, summary_indices_seq, summary_values_seq)
            packed_pos.scatter_(0, summary_indices_pos, summary_values_pos)
            # packed_tri_ids are already -1

        # Copy light and normal_geo tokens based on config
        if self.put_light_tokens_in_sink:
            # Order: [register][env][summary][light][normal_geo]

            # Copy light tokens
            if max_light > 0:
                light_pos_in_batch = torch.arange(max_light, device=device).unsqueeze(0).expand(B, -1)
                light_valid = light_pos_in_batch < n_light.unsqueeze(1)
                light_global_pos = base_offsets.unsqueeze(1) + summary_offset + n_summary.unsqueeze(1) + light_pos_in_batch

                light_indices_seq = light_global_pos.unsqueeze(-1).expand(-1, -1, D)[light_valid]
                light_indices_pos = light_global_pos.unsqueeze(-1).expand(-1, -1, 3)[light_valid]
                light_indices_ids = light_global_pos[light_valid] # [N_valid]
                
                light_values_seq = light_geo_embs[light_valid]
                light_values_pos = light_geo_centers[light_valid].float()
                light_values_ids = light_geo_tri_ids[light_valid]

                packed_seq.scatter_(0, light_indices_seq, light_values_seq)
                packed_pos.scatter_(0, light_indices_pos, light_values_pos)
                packed_tri_ids.scatter_(0, light_indices_ids, light_values_ids)

            # Copy normal_geo tokens
            if max_normal > 0:
                normal_pos_in_batch = torch.arange(max_normal, device=device).unsqueeze(0).expand(B, -1)
                normal_valid = normal_pos_in_batch < n_normal_geo.unsqueeze(1)
                normal_global_pos = base_offsets.unsqueeze(1) + summary_offset + n_summary.unsqueeze(1) + n_light.unsqueeze(1) + normal_pos_in_batch

                normal_indices_seq = normal_global_pos.unsqueeze(-1).expand(-1, -1, D)[normal_valid]
                normal_indices_pos = normal_global_pos.unsqueeze(-1).expand(-1, -1, 3)[normal_valid]
                normal_indices_ids = normal_global_pos[normal_valid]
                
                normal_values_seq = sorted_normal_geo_embs[normal_valid]
                normal_values_pos = sorted_normal_geo_centers[normal_valid].float()
                normal_values_ids = sorted_normal_geo_tri_ids[normal_valid]

                packed_seq.scatter_(0, normal_indices_seq, normal_values_seq)
                packed_pos.scatter_(0, normal_indices_pos, normal_values_pos)
                packed_tri_ids.scatter_(0, normal_indices_ids, normal_values_ids)
        else:
            # Order: [register][env][summary][normal_geo][light]

            # Copy normal_geo tokens
            if max_normal > 0:
                normal_pos_in_batch = torch.arange(max_normal, device=device).unsqueeze(0).expand(B, -1)
                normal_valid = normal_pos_in_batch < n_normal_geo.unsqueeze(1)
                normal_global_pos = base_offsets.unsqueeze(1) + summary_offset + n_summary.unsqueeze(1) + normal_pos_in_batch

                normal_indices_seq = normal_global_pos.unsqueeze(-1).expand(-1, -1, D)[normal_valid]
                normal_indices_pos = normal_global_pos.unsqueeze(-1).expand(-1, -1, 3)[normal_valid]
                normal_indices_ids = normal_global_pos[normal_valid]
                
                normal_values_seq = sorted_normal_geo_embs[normal_valid]
                normal_values_pos = sorted_normal_geo_centers[normal_valid].float()
                normal_values_ids = sorted_normal_geo_tri_ids[normal_valid]

                packed_seq.scatter_(0, normal_indices_seq, normal_values_seq)
                packed_pos.scatter_(0, normal_indices_pos, normal_values_pos)
                packed_tri_ids.scatter_(0, normal_indices_ids, normal_values_ids)

            # Copy light tokens
            if max_light > 0:
                light_pos_in_batch = torch.arange(max_light, device=device).unsqueeze(0).expand(B, -1)
                light_valid = light_pos_in_batch < n_light.unsqueeze(1)
                light_global_pos = base_offsets.unsqueeze(1) + summary_offset + n_summary.unsqueeze(1) + n_normal_geo.unsqueeze(1) + light_pos_in_batch

                light_indices_seq = light_global_pos.unsqueeze(-1).expand(-1, -1, D)[light_valid]
                light_indices_pos = light_global_pos.unsqueeze(-1).expand(-1, -1, 3)[light_valid]
                light_indices_ids = light_global_pos[light_valid]
                
                light_values_seq = light_geo_embs[light_valid]
                light_values_pos = light_geo_centers[light_valid].float()
                light_values_ids = light_geo_tri_ids[light_valid]

                packed_seq.scatter_(0, light_indices_seq, light_values_seq)
                packed_pos.scatter_(0, light_indices_pos, light_values_pos)
                packed_tri_ids.scatter_(0, light_indices_ids, light_values_ids)

        max_seq_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        return PackedBatch(
            packed_seq=packed_seq,
            packed_pos=packed_pos,
            cu_seqlens=cu_seqlens,
            sink_lens=sink_lens,
            max_seq_len=max_seq_len,
            is_triangle_mask=sorted_is_triangle_mask,
            n_normal_geo=n_normal_geo,
            packed_tri_ids=packed_tri_ids,
            is_env_mask=sorted_is_env_mask,
        )
