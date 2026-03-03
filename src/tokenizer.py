"""
Custom Byte-Pair Encoding (BPE) Tokenizer.

Trains from scratch on a raw text corpus, learning subword merges
that optimally compress the domain-specific vocabulary.

Why a custom tokenizer?
  - GPT-2's tokenizer is trained on WebText (web data)
  - A tokenizer trained on WikiText-103 / scientific text achieves
    lower average tokens-per-word (better compression → lower perplexity)
  - Training your own tokenizer is a key part of the pipeline

Algorithm:
  1. Pre-tokenise: split corpus into words
  2. Initialise vocabulary with all unique characters
  3. Iteratively find most frequent adjacent symbol pair
  4. Merge pair → new token, add to vocabulary
  5. Repeat until desired vocab size

Reference: "Neural Machine Translation of Rare Words with Subword Units"
           Sennrich et al., 2016 — https://arxiv.org/abs/1508.07909
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "<bos>": 3,
}


class BPETokenizer:
    """
    A trainable Byte-Pair Encoding tokenizer.

    Usage:
        tokenizer = BPETokenizer(vocab_size=32_000)
        tokenizer.train(text)                # train on raw string
        ids = tokenizer.encode("hello world")
        text = tokenizer.decode(ids)
        tokenizer.save("tokenizer.json")
        tokenizer = BPETokenizer.load("tokenizer.json")
    """

    def __init__(self, vocab_size: int = 32_000):
        self.vocab_size = vocab_size
        self.vocab:   Dict[str, int] = {}    # token string → int id
        self.merges:  Dict[Tuple[str, str], str] = {}  # (a, b) → merged
        self.id2token: Dict[int, str] = {}

    # ─────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────

    def train(self, corpus: str, verbose: bool = True) -> None:
        """
        Train BPE on a raw text corpus.

        Args:
            corpus : raw text string (can be very large)
            verbose: print merge progress every 1000 steps
        """
        # Step 1: Build word frequency table
        # Each word is represented as space-separated characters + </w> end marker
        words = re.findall(r'\S+', corpus.lower())
        word_freq: Dict[str, int] = Counter(
            ' '.join(list(word)) + ' </w>' for word in words
        )

        # Step 2: Collect initial character vocabulary
        chars: set = set()
        for word in word_freq:
            chars.update(word.split())

        # Assign IDs: special tokens first, then sorted chars
        self.vocab = dict(SPECIAL_TOKENS)
        offset = len(self.vocab)
        for i, ch in enumerate(sorted(chars)):
            if ch not in self.vocab:
                self.vocab[ch] = i + offset

        self.merges = {}
        num_merges = self.vocab_size - len(self.vocab)

        if verbose:
            print(f"Initial vocab size: {len(self.vocab)} characters")
            print(f"Running {num_merges} BPE merges...")

        # Step 3: BPE merge loop
        for step in range(num_merges):
            # Count all adjacent pairs
            pair_freq = self._count_pairs(word_freq)
            if not pair_freq:
                break

            # Pick most frequent pair
            best_pair = max(pair_freq, key=pair_freq.get)
            merged    = ''.join(best_pair)

            # Register merge and new token
            self.merges[best_pair] = merged
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

            # Apply merge to word_freq
            word_freq = self._apply_merge(word_freq, best_pair, merged)

            if verbose and step % 2000 == 0:
                print(f"  Step {step:6d}/{num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' → '{merged}' "
                      f"(freq={pair_freq[best_pair]})")

        self.id2token = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    @staticmethod
    def _count_pairs(word_freq: Dict[str, int]) -> Counter:
        """Count frequency of all adjacent symbol pairs across the corpus."""
        pair_freq: Counter = Counter()
        for word, freq in word_freq.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_freq[(symbols[i], symbols[i + 1])] += freq
        return pair_freq

    @staticmethod
    def _apply_merge(
        word_freq: Dict[str, int],
        pair:      Tuple[str, str],
        merged:    str,
    ) -> Dict[str, int]:
        """Replace all occurrences of `pair` in the vocabulary with `merged`."""
        a, b = pair
        bigram = re.escape(f'{a} {b}')
        replacement = a + b  # no space — tokens merged
        new_freq = {}
        for word, freq in word_freq.items():
            new_word = re.sub(r'(?<!\S)' + bigram + r'(?!\S)', replacement, word)
            new_freq[new_word] = new_freq.get(new_word, 0) + freq
        return new_freq

    # ─────────────────────────────────────────────────────────────────
    # Encode / Decode
    # ─────────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to a single word and return list of subword tokens."""
        symbols = list(word) + ['</w>']
        # Greedily apply merges in training order (earlier merges = higher priority)
        merge_order = {pair: i for i, pair in enumerate(self.merges)}
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            mergeable = [(merge_order[p], p) for p in pairs if p in merge_order]
            if not mergeable:
                break
            _, best = min(mergeable)  # lowest index = earliest merge
            a, b = best
            merged = self.merges[best]
            # Merge all occurrences of best pair (left-to-right)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """
        Encode a string to a list of token IDs.

        Args:
            text    : input string
            add_eos : append EOS token ID at the end

        Returns:
            list of integer token IDs
        """
        ids = []
        for word in re.findall(r'\S+', text.lower()):
            tokens = self._tokenize_word(word)
            ids.extend(
                self.vocab.get(t, SPECIAL_TOKENS['<unk>']) for t in tokens
            )
        if add_eos:
            ids.append(SPECIAL_TOKENS['<eos>'])
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs back to a string.

        Args:
            ids          : list of token IDs
            skip_special : if True, skip special tokens in output

        Returns:
            decoded string
        """
        special_ids = set(SPECIAL_TOKENS.values())
        tokens = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            tokens.append(self.id2token.get(i, '<unk>'))
        return ''.join(tokens).replace('</w>', ' ').strip()

    def __len__(self) -> int:
        return len(self.vocab)

    # ─────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save tokenizer to a JSON file."""
        data = {
            'vocab_size': self.vocab_size,
            'vocab':      self.vocab,
            'merges':     [list(k) for k in self.merges.keys()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load a saved tokenizer from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=data['vocab_size'])
        tok.vocab  = {k: int(v) for k, v in data['vocab'].items()}
        tok.merges = {(m[0], m[1]): ''.join(m) for m in data['merges']}
        tok.id2token = {v: k for k, v in tok.vocab.items()}
        return tok

    def get_vocab_size(self) -> int:
        return len(self.vocab)
