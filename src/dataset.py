"""
Dataset and data utilities for language model training.

Supports:
  - WikiText-103  (103M tokens, Wikipedia — default)
  - WikiText-2    (2M tokens, for quick smoke tests)
  - Custom text files

The entire corpus is tokenized once and stored as a flat tensor.
During training, we slice fixed-length windows with targets shifted by 1.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class TokenizedDataset(Dataset):
    """
    Flat-token language modelling dataset.

    Tokenises the full split once (or loads a cached .pt file),
    then yields (input, target) pairs of length `seq_len`.

    input  = tokens[i : i+seq_len]
    target = tokens[i+1 : i+seq_len+1]   ← shifted by 1 for next-token prediction
    """

    def __init__(
        self,
        tokens:  torch.Tensor,
        seq_len: int = 1_024,
    ):
        self.tokens  = tokens
        self.seq_len = seq_len
        self.n       = (len(tokens) - 1) // seq_len  # number of non-overlapping chunks

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start           : start + self.seq_len]
        y = self.tokens[start + 1       : start + self.seq_len + 1]
        return x, y


def load_wikitext(
    tokenizer,
    split:       str  = 'train',
    seq_len:     int  = 1_024,
    cache_dir:   str  = 'data',
    dataset_name: str = 'wikitext-103-v1',
    max_tokens:  Optional[int] = None,
    verbose:     bool = True,
) -> TokenizedDataset:
    """
    Load and tokenise WikiText-103 (or WikiText-2) for a given split.

    Args:
        tokenizer   : trained BPETokenizer instance
        split       : 'train', 'validation', or 'test'
        seq_len     : chunk size for training windows
        cache_dir   : directory to cache tokenised tensors
        dataset_name: 'wikitext-103-v1' or 'wikitext-2-v1'
        max_tokens  : truncate to this many tokens (None = use all)
        verbose     : print progress info

    Returns:
        TokenizedDataset ready for DataLoader
    """
    from datasets import load_dataset  # lazy import

    cache_path = Path(cache_dir) / f"{dataset_name}_{split}_tokens.pt"

    if cache_path.exists():
        if verbose:
            print(f"Loading cached tokens from {cache_path}")
        tokens = torch.load(cache_path)
    else:
        if verbose:
            print(f"Downloading and tokenising {dataset_name} [{split}]...")

        dataset = load_dataset('wikitext', dataset_name, split=split)

        # Concatenate all non-empty article texts
        all_text = '\n'.join(
            row['text'] for row in dataset if row['text'].strip()
        )

        if verbose:
            print(f"  Raw text: {len(all_text):,} characters")

        ids = tokenizer.encode(all_text, add_eos=False)
        tokens = torch.tensor(ids, dtype=torch.long)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)

        if verbose:
            print(f"  Cached {len(tokens):,} tokens → {cache_path}")

    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    ds = TokenizedDataset(tokens, seq_len)

    if verbose:
        print(f"  {split}: {len(tokens):,} tokens → {len(ds):,} chunks of {seq_len}")

    return ds


def get_dataloaders(
    tokenizer,
    seq_len:      int  = 1_024,
    batch_size:   int  = 16,
    num_workers:  int  = 4,
    cache_dir:    str  = 'data',
    dataset_name: str  = 'wikitext-103-v1',
    max_train_tokens: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for WikiText.

    Returns:
        (train_loader, val_loader)
    """
    train_ds = load_wikitext(
        tokenizer, split='train', seq_len=seq_len,
        cache_dir=cache_dir, dataset_name=dataset_name,
        max_tokens=max_train_tokens,
    )
    val_ds = load_wikitext(
        tokenizer, split='validation', seq_len=seq_len,
        cache_dir=cache_dir, dataset_name=dataset_name,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader


def load_custom_text(
    path:      str,
    tokenizer,
    seq_len:   int  = 1_024,
    train_pct: float = 0.9,
) -> Tuple[TokenizedDataset, TokenizedDataset]:
    """
    Load and split a custom plaintext file into train/val datasets.

    Args:
        path      : path to .txt file
        tokenizer : trained BPETokenizer
        seq_len   : sequence length for chunking
        train_pct : fraction used for training (rest = validation)

    Returns:
        (train_dataset, val_dataset)
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    ids    = tokenizer.encode(text)
    tokens = torch.tensor(ids, dtype=torch.long)

    split_idx = int(len(tokens) * train_pct)
    train_ds  = TokenizedDataset(tokens[:split_idx], seq_len)
    val_ds    = TokenizedDataset(tokens[split_idx:], seq_len)

    print(f"Custom dataset: {len(tokens):,} tokens → "
          f"train: {len(train_ds):,} | val: {len(val_ds):,} chunks")

    return train_ds, val_ds
