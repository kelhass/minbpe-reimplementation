# minbpe exercise

import unicodedata

# get stats function
def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair,0) + 1
    return counts

# merge function
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids): # can't do while i < len(ids)-1 because excludes last element
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# helper functions
def replace_control_characters(s: str) -> str: # removes control characters from output
    # don't want to print control chars that distort output (ex. \n)
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != 'C':
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return ''.join(chars)

def render_token(t: bytes) -> str: # converts bytes to strings to produce readable output
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s) # removes control chars
    return s

# Tokenizer Class ------------------------------------------------------------

class Tokenizer: # base class for all tokenizers

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {} # str -> int
        self.vocab = self._build_vocab() # int -> byte (size = 256)
    
    def train(self, text, vocab_size, verbose=False):
        # will implement in child classes
        raise NotImplementedError
    
    def encode(self, text):
        # will implement in child classes
        raise NotImplementedError
    
    def decode(self, ids):
        # will implement in child classes
        raise NotImplementedError
    
    def _build_vocab(self):
        # vocab is deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)} # original 255 bytes
        for (p0,p1), idx in self.merges.items(): # adds merges to vocab
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items(): # adds special tokens to vocab
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        Inspired by sentencepiece model saving (but not equivalent)
        --- model file is the critical one, intended for load()
        --- vocab file is just printed version for human inspection
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern, and merges
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number then the token
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # merges dict
            for (idx1, idx2), idx in self.merges.items():
                f.write(f"{idx1} {idx2}\n")
        # write the vocab that's readable for a human
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges: # find children of token s (if any)
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else: # means it's one of the first 255 tokens (leaf token)
                    f.write(f"[{s}] {idx}\n")
    def load(self, model_file): # inverse of save but only for model file
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
