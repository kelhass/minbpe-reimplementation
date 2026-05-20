#!/usr/bin/env python3
"""Train a regex BPE tokenizer on the included Taylor Swift article text."""

from pathlib import Path

from bpe_tokenizer import RegexTokenizer


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    text_path = project_root / "data" / "taylorswift.txt"
    text = text_path.read_text(encoding="utf-8")

    tokenizer = RegexTokenizer()
    vocab_size = 1000
    tokenizer.train(text, vocab_size=vocab_size, verbose=False)

    sample = "Taylor Swift is an American singer-songwriter."
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print(f"Training text characters: {len(text):,}")
    print(f"Vocab size: {256 + len(tokenizer.merges):,}")
    print(f"Sample: {sample}")
    print(f"Token ids: {ids}")
    print(f"Decoded: {decoded}")
    print(f"Round trip successful: {sample == decoded}")


if __name__ == "__main__":
    main()
