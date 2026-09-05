from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class PackedBatch:
    """
    Packed sequence batch for varlen attention.

    Sequence structure per sample (when put_env_tokens_in_sink=True):

    [register][env][summary][light_geo][sorted_normal_geo]

    |<------------- sink ------------->|<---- normal ---->|

    When put_env_tokens_in_sink=False, env tokens are mixed into sorted_normal_geo:

    [register][summary][light_geo][sorted_normal_geo (includes env)]

    |<-------- sink -------->|<-------- normal ---------->|
    """

    # Packed tokens
    packed_seq: torch.Tensor      # [total_tokens, D]
    packed_pos: torch.Tensor      # [total_tokens, 3] for RoPE

    # Batch structure
    cu_seqlens: torch.Tensor      # [B+1], cumulative sequence lengths
    sink_lens: torch.Tensor       # [B], sink length per sample

    # Optional: for decoding back to per-sample
    max_seq_len: int              # max(seq_lens)

    # DeepStack: token type tracking (for triangle vs volume)
    is_triangle_mask: Optional[torch.Tensor] = None  # [B, max_normal_geo]
    n_normal_geo: Optional[torch.Tensor] = None      # [B]

    # DeepStack: mapping from packed sequence to original triangle index
    packed_tri_ids: Optional[torch.Tensor] = None  # [total_tokens]

    # Ablation: env token tracking (when env tokens are mixed into normal_geo)
    is_env_mask: Optional[torch.Tensor] = None  # [B, max_normal_geo]

    @property
    def batch_size(self) -> int:
        return len(self.cu_seqlens) - 1

    @property
    def total_tokens(self) -> int:
        return self.packed_seq.shape[0]

    @property
    def seq_lens(self) -> torch.Tensor:
        """Per-sample sequence lengths: [B]"""
        return self.cu_seqlens[1:] - self.cu_seqlens[:-1]

    def get_sample(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get tokens and positions for a single sample."""
        start = self.cu_seqlens[idx].item()
        end = self.cu_seqlens[idx + 1].item()
        return self.packed_seq[start:end], self.packed_pos[start:end]

