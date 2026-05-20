# Byte-Pair Encoding Tokenizer Reimplementation

This repository contains my personal learning reimplementation of byte-pair encoding (BPE) tokenizers inspired by Andrej Karpathy's `minbpe` project.

The project explores how modern language model tokenizers work by building tokenizer components from the ground up: byte-level tokenization, pair-frequency counting, iterative BPE merges, regex-based text splitting, special-token handling, saving/loading tokenizer files, and a small training script using an included text corpus.

## Project Purpose

I created this project while studying tokenization as part of my broader self-study in language models and neural networks. The goal was to move beyond treating tokenizers as black boxes and understand how text becomes integer token sequences before being passed into a model.

This is best understood as a guided learning project and reimplementation rather than an original tokenizer library.

## What This Project Demonstrates

- byte-level text encoding using UTF-8
- byte-pair encoding merge rules
- pair-frequency counting
- iterative vocabulary construction
- regex-based tokenization chunks
- special token registration and handling
- tokenizer encode/decode round-trip validation
- lightweight test coverage with `pytest`
- a small training script using a public text corpus

## Repository Structure

```text
.
├── data/
│   ├── taylorswift.txt
│   └── README.md
├── examples/
│   └── train_taylor_swift_tokenizer.py
├── original/
│   ├── basic_tokenizer.py
│   ├── gpt4_tokenizer.py
│   ├── minbpe_base.py
│   ├── regex_tokenizer.py
│   ├── test_tokenizer.py
│   └── tswift_tokens.py
├── src/
│   └── bpe_tokenizer/
│       ├── __init__.py
│       ├── base.py
│       ├── basic.py
│       └── regex.py
├── tests/
│   └── test_tokenizer.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Main Components

### `BasicTokenizer`

A minimal byte-level BPE tokenizer. It converts text into UTF-8 bytes, learns frequent adjacent byte-pair merges, and supports encoding and decoding text.

### `RegexTokenizer`

A more advanced tokenizer that first splits text into regex-based chunks before applying BPE merges. This mirrors the idea used by GPT-style tokenizers, where text is chunked before byte-pair merges are applied.

### Training Example

The example script trains a regex tokenizer on the included Taylor Swift text corpus and demonstrates encode/decode behavior on a short sample sentence.

Run it with:

```bash
python examples/train_taylor_swift_tokenizer.py
```

## Data

This repository includes `data/taylorswift.txt`, a text corpus used for tokenizer training experiments. See `data/README.md` for details and attribution.

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the tests:

```bash
pytest
```

Run the training example:

```bash
python examples/train_taylor_swift_tokenizer.py
```

## Requirements

Core dependencies are intentionally minimal:

```text
regex
pytest
```

The original exploratory files also reference `tiktoken` for comparison with OpenAI tokenizer behavior, so it is included in `requirements.txt` as an optional learning dependency.

## Attribution

This project builds on ideas from Andrej Karpathy's `minbpe` project and tokenizer lecture materials.

Original project: https://github.com/karpathy/minbpe

Video: https://www.youtube.com/watch?v=zduSFxRajkE&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=9

The files in `original/` preserve my original learning versions. The code in `src/` is a cleaned and organized version prepared for portfolio presentation.

## License

This repository is licensed under the MIT License for original code, notes, and documentation created by Kelly Hassett.

Any concepts, structure, or code patterns adapted from Andrej Karpathy's `minbpe` project remain subject to the original project's MIT License. This repository is intended for educational and portfolio purposes.
