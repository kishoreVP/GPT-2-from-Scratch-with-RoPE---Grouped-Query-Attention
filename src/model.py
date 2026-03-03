"""
GPT-2 with Rotary Position Embedding (RoPE) and Grouped Query Attention (GQA).

Architecture:
  - Decoder-only transformer (causal LM)
  - Token embeddings only — no absolute positional embedding table
  - N × [Pre-LayerNorm → GQA (w/ RoPE) → residual → Pre-LN → FFN → residual]
  - Final LayerNorm → Linear LM head (weight-tied with token embeddings)

Differences from vanilla GPT-2:
  ✓ RoPE instead of absolute learned positional embeddings
  ✓ GQA (n_kv_heads < n_heads) instead of MHA
  ✓ Pre-LayerNorm (more training-stable than GPT-2's post-LN)
  ✓ No bias in linear layers (modern practice, better regularisation)
  ✓ Weight tying: embedding and LM head share weights

Parameter count: ~85M  (comparable to GPT-2 Small)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """
    Model hyperparameters.

    Defaults produce a ~85M parameter model matching GPT-2 Small capacity
    but with GQA (n_kv_heads=4, saving ~40% KV-cache vs n_kv_heads=12).
    """
    # Vocabulary / sequence
    vocab_size:  int   = 32_000   # Custom BPE vocabulary size
    max_seq_len: int   = 1_024    # Maximum context window

    # Model dimensions
    d_model:     int   = 768      # Embedding / hidden dimension
    n_layers:    int   = 12       # Number of transformer blocks
    n_heads:     int   = 12       # Query attention heads
    n_kv_heads:  int   = 4        # Key/Value heads (GQA: 3 queries share each KV)
    d_ff:        int   = 3_072    # Feed-forward inner dim (4 × d_model)

    # Regularisation
    dropout:     float = 0.1
    bias:        bool  = False    # Biases in linear layers

    # Derived (computed automatically)
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# ─────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Architecture: Linear → GELU → Linear → Dropout
    Hidden dim is 4× d_model (standard GPT-2 configuration).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1  = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2  = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Single GPT-2 transformer block with Pre-LayerNorm.

    Pre-LN: LayerNorm is applied BEFORE each sublayer (not after).
    This is more training-stable and is used in GPT-3, PaLM, LLaMA, etc.

    Residual stream:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1  = nn.LayerNorm(config.d_model)
        self.attn  = GroupedQueryAttention(config)
        self.ln_2  = nn.LayerNorm(config.d_model)
        self.mlp   = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv_cache = self.attn(self.ln_1(x), mask=mask, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class GPT2RoPEGQA(nn.Module):
    """
    GPT-2 Language Model with RoPE and Grouped Query Attention.

    Features:
      - Causal (autoregressive) language modelling
      - RoPE applied inside every attention layer — no positional embedding table
      - GQA with configurable n_kv_heads
      - KV-cache support for efficient autoregressive generation
      - Weight tying: LM head shares weights with token embedding
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Embedding (token only — RoPE handles position) ─────────────
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_drop   = nn.Dropout(config.dropout)

        # ── Transformer Blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ── Head ───────────────────────────────────────────────────────
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding ↔ LM head (reduces params, improves ppl)
        self.lm_head.weight = self.embed_tokens.weight

        # ── Causal Mask (registered buffer — lives on same device as model) ─
        # Lower-triangular: token i can attend to j ≤ i
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask)

        # ── Weight Initialisation ──────────────────────────────────────
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2 × n_layers) (GPT-2 paper)
        scale = (2 * config.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard GPT-2 initialisation: Normal(0, 0.02) for weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Parameter count helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return total parameter count.
        If non_embedding=True, exclude the (tied) token embedding table
        since those weights are counted once in lm_head.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
        return n

    def kv_cache_memory_mb(self, seq_len: int, batch_size: int = 1) -> float:
        """Estimate KV-cache memory in MB (float16)."""
        # 2 tensors (K + V) × layers × n_kv_heads × seq_len × d_head × 2 bytes (fp16)
        return (2 * self.config.n_layers * self.config.n_kv_heads
                * seq_len * self.config.d_head * batch_size * 2) / (1024 ** 2)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets:   Optional[torch.Tensor] = None,
        past_kvs:  Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids : [B, T]  token IDs
            targets   : [B, T]  next-token targets (for training loss)
            past_kvs  : list of (K, V) per layer, for inference caching

        Returns:
            logits    : [B, T, vocab_size]
            loss      : scalar cross-entropy (only if targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} > max_seq_len {self.config.max_seq_len}"
        )

        # ── Embed tokens (no positional embedding!) ─────────────────────
        x = self.embed_drop(self.embed_tokens(input_ids))  # [B, T, d_model]

        # ── Causal mask ─────────────────────────────────────────────────
        mask = self.causal_mask[:T, :T]  # [T, T]

        # ── Transformer blocks ──────────────────────────────────────────
        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, kv = block(x, mask=mask, past_kv=past_kv)
            new_kvs.append(kv)

        # ── Head ────────────────────────────────────────────────────────
        x      = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # ── Loss ────────────────────────────────────────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids:      torch.Tensor,
        max_new_tokens: int   = 200,
        temperature:    float = 0.8,
        top_k:          int   = 50,
        top_p:          float = 0.95,
        eos_token_id:   Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature, top-k, and top-p (nucleus) sampling.

        Args:
            input_ids     : [B, T] prompt token IDs
            max_new_tokens: tokens to generate
            temperature   : softmax temperature (higher = more random)
            top_k         : keep top-k logits (0 to disable)
            top_p         : nucleus sampling — keep min tokens whose cumulative prob ≥ top_p
            eos_token_id  : stop generation when this token is produced

        Returns:
            [B, T + max_new_tokens] token IDs
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate to max context
            ctx = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature  # [B, vocab]

            # Top-k filtering
            if top_k > 0:
                top_k_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < top_k_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                # Remove tokens whose cumulative prob exceeds top_p
                sorted_remove = cumulative_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs   = logits.softmax(dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        return input_ids
