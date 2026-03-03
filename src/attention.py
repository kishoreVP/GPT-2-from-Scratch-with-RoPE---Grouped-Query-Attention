"""
Grouped Query Attention (GQA) with RoPE.

Reference: "GQA: Training Generalised Multi-Query Transformer Models from
            Multi-Head Checkpoints" — Ainslie et al., 2023
            https://arxiv.org/abs/2305.13245

Attention variants:
  MHA  : n_heads == n_kv_heads  (standard multi-head)
  GQA  : 1 < n_kv_heads < n_heads  (grouped — our default)
  MQA  : n_kv_heads == 1  (multi-query — maximum compression)

KV-cache memory:
  MHA  : n_heads   × seq_len × d_head  per layer  (baseline)
  GQA  : n_kv_heads × seq_len × d_head per layer  (reduced by factor n_heads/n_kv_heads)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention with RoPE and optional KV-cache for inference.

    Config fields used:
        d_model    : model dimension
        n_heads    : number of query heads
        n_kv_heads : number of key/value heads  (n_heads must be divisible by this)
        dropout    : attention dropout probability
        bias       : whether to use bias in projections
        max_seq_len: for RoPE cache pre-computation
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        assert config.n_heads % config.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_heads // config.n_kv_heads   # queries per KV head
        self.d_head     = config.d_model  // config.n_heads
        self.d_model    = config.d_model
        self.dropout    = config.dropout

        # Projections — note K/V project to SMALLER dimension (n_kv_heads * d_head)
        self.q_proj   = nn.Linear(config.d_model, self.n_heads    * self.d_head, bias=config.bias)
        self.k_proj   = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.v_proj   = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=config.bias)
        self.out_proj = nn.Linear(self.n_heads * self.d_head, config.d_model,    bias=config.bias)

        self.attn_drop = nn.Dropout(config.dropout)
        self.rope      = RotaryEmbedding(self.d_head, max_seq_len=config.max_seq_len)

        self.scale = self.d_head ** -0.5

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads to match query heads.
        Input : [B, n_kv_heads, T, d_head]
        Output: [B, n_heads,    T, d_head]
        """
        if self.n_groups == 1:
            return x  # MHA — already aligned
        B, _, T, D = x.shape
        # Interleave via expand (no copy if contiguous)
        x = x[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_groups, T, D)
        return x.reshape(B, self.n_heads, T, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x       : [B, T, d_model]
            mask    : [T, T] causal mask (1 = attend, 0 = masked)
            past_kv : cached (K, V) from previous steps for autoregressive generation

        Returns:
            output  : [B, T, d_model]
            kv_cache: (K, V) tensors for caching
        """
        B, T, C = x.shape

        # ── Project ──────────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)  # [B,nH,T,dH]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,nKV,T,dH]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # [B,nKV,T,dH]

        # ── RoPE ─────────────────────────────────────────────────────
        q, k = self.rope(q, k, T)

        # ── KV Cache (inference) ──────────────────────────────────────
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        kv_cache = (k, v)

        # ── Expand KV to match query heads ────────────────────────────
        k_exp = self._expand_kv(k)  # [B, nH, T_full, dH]
        v_exp = self._expand_kv(v)

        # ── Scaled Dot-Product Attention ──────────────────────────────
        T_full = k_exp.shape[2]
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale  # [B,nH,T,T_full]

        if mask is not None:
            # mask: [T, T_full], 0 → -inf
            attn = attn.masked_fill(mask[:T, :T_full] == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # ── Output ───────────────────────────────────────────────────
        out = torch.matmul(attn, v_exp)              # [B, nH, T, dH]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, nH*dH]
        out = self.out_proj(out)

        return out, kv_cache
