"""
GPT-2 with RoPE & GQA — source package.
"""
from .model     import GPT2RoPEGQA, GPTConfig
from .attention import GroupedQueryAttention
from .rope      import RotaryEmbedding
from .tokenizer import BPETokenizer
from .dataset   import TokenizedDataset, get_dataloaders, load_wikitext, load_custom_text
from .trainer   import Trainer, cosine_schedule

__all__ = [
    "GPT2RoPEGQA", "GPTConfig",
    "GroupedQueryAttention",
    "RotaryEmbedding",
    "BPETokenizer",
    "TokenizedDataset", "get_dataloaders", "load_wikitext", "load_custom_text",
    "Trainer", "cosine_schedule",
]
