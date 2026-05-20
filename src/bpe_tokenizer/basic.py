"""A minimal byte-level BPE tokenizer."""

from __future__ import annotations

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """Minimal byte-pair encoding tokenizer without regex splitting or special tokens."""

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        tokens = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(vocab_size - 256):
            stats = get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = merge(tokens, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")
