"""Shared utilities and base class for byte-pair encoding tokenizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import unicodedata


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
    """Count adjacent token pairs in a sequence of integer token ids."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with the merged token ``idx``."""
    new_ids: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def replace_control_characters(text: str) -> str:
    """Escape control characters so tokens can be printed safely."""
    chars = []
    for ch in text:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)


def render_token(token: bytes) -> str:
    """Convert a byte token to readable text for vocabulary inspection."""
    return replace_control_characters(token.decode("utf-8", errors="replace"))


@dataclass
class Tokenizer:
    """Base class for tokenizer implementations."""

    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    pattern: str = ""
    special_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str):
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix: str | Path) -> None:
        """Save a machine-readable `.model` and human-readable `.vocab` file."""
        file_prefix = str(file_prefix)
        with open(file_prefix + ".model", "w", encoding="utf-8") as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(file_prefix + ".vocab", "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                rendered = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    rendered0 = render_token(self.vocab[idx0])
                    rendered1 = render_token(self.vocab[idx1])
                    f.write(f"[{rendered0}][{rendered1}] -> [{rendered}] {idx}\n")
                else:
                    f.write(f"[{rendered}] {idx}\n")

    def load(self, model_file: str | Path) -> None:
        """Load a tokenizer from a `.model` file saved by :meth:`save`."""
        model_file = str(model_file)
        if not model_file.endswith(".model"):
            raise ValueError("model_file must end with .model")

        merges: dict[tuple[int, int], int] = {}
        special_tokens: dict[str, int] = {}
        idx = 256
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            if version != "minbpe v1":
                raise ValueError(f"unsupported model version: {version}")
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
