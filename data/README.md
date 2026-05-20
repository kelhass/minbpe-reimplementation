# Data

This folder contains the text corpus used for tokenizer training experiments.

## Included Dataset

### `taylorswift.txt`

This file is a plain-text copy of the Wikipedia article on Taylor Swift, dated February 16, 2024.

The text is used as a small natural-language corpus for experimenting with byte-pair encoding tokenizers. The tokenizer training scripts use it to learn frequent byte-pair merges and build a larger vocabulary from raw UTF-8 bytes.

## Expected Format

The training scripts expect a plain text file encoded as UTF-8.

```text
Taylor Alison Swift ...
...
```

There is no header row or structured schema. The file is treated as one continuous text corpus.

## Usage

The example script reads this file, trains a regex-based BPE tokenizer, and prints a short encode/decode demonstration.

```bash
python examples/train_taylor_swift_tokenizer.py
```

## Attribution

This corpus was created from the public Wikipedia article on Taylor Swift. The file itself includes a note describing it as a copy-paste of the article as of February 16, 2024.

For new experiments, this file can be replaced with any UTF-8 text corpus.
