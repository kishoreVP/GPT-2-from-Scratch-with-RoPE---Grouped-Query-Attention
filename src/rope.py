"""
Rotary Position Embedding (RoPE)
Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           Su et al., 2021 — https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding applied inside attention to Q and K.

    Key properties:
      - Encodes relative position (m-n) via rotation, not absolute position
      - Zero learnable parameters
      - Generalises better than learned absolute PE
      - Used in: LLaMA, Mistral, Gemma, PaLM 2, Qwen, ...

    Math:
      θ_i  = 1 / 10000^(2i / d)          (frequency for each dimension pair)
      RoPE(x, m) = x ⊗ cos(mθ) + rotate_half(x) ⊗ sin(mθ)

      Key: <RoPE(q,m), RoPE(k,n)> depends only on (m-n), not m or n separately.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # θ_i = 1 / (base^(2i/dim)),  shape [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/cos cache for speed
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Pre-compute and cache sin/cos tables for positions [0, seq_len)."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Outer product: [seq_len, dim/2]
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        # Concat to get [seq_len, dim] (each freq appears twice: for real & imag)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate each 2D subspace by 90°: [x1, x2] → [-x2, x1]."""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: [batch, n_heads,    seq_len, head_dim]
            k: [batch, n_kv_heads, seq_len, head_dim]
            seq_len: current sequence length

        Returns:
            q_rot, k_rot: rotated tensors of same shape
        """
        if seq_len > self.max_seq_len:
            # Extend cache on-the-fly if needed
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # [1,1,T,d]
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot
