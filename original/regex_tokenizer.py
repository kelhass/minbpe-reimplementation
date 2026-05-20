# Algorithmically follows ChatGPT tokenizer

import regex as re
from minbpe_base import Tokenizer, get_stats, merge

# GPT text split patterns
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        # pattern: optionally override the default pattern (we're overriding w GPT4)
        # special_tokens: str -> int dictionary of special tokens
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split text into chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text pre-processing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks] # divides based on GPT4 pattern

        # iteratively merge most common pairs to create new tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges): # processes chunks/linguistic units (punctuation, words, etc) separately
            # count number of times consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts of diff chunks
                get_stats(chunk_ids, stats)
            # find pair with highest count
            pair = max(stats, key=stats.get)
            idx = 256 + i # mint new token
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids] # replace all occurences
            merges[pair] = idx # save the merge
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose: # prints
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab # used in decode()
    
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()} # int -> str
    
    def decode(self, ids): # given list of ints, return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes): # encoding algorithm from before
        # first convert all bytes to ints in range 0-255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find pair w lowest merge
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            # otherwise we merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text): # ignores special tokens
        text_chunks = re.findall(self.compiled_pattern, text) # splits text into chunks
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"): # handles special tokens
        # allowed_special = "none_raise" means error is raised if special token is encountered
        # this is the default behavior for tiktoken and is what we want

        # decode user desire with respect to handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        if not special: # shortcut: if no special tokens, just use ordinary encoding
            return self.encode_ordinary(text)
        
        # handle tokens by splitting the text based on occurence of any exact match with any special token
        # use re.split -> surrounding the pattern with ( ) makes it into a capturing group so special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text) # separates special chars from the rest of the text

        ids = []
        for part in special_chunks:
            if part in special: # this is a special token
                ids.append(special[part])
            else: # not special
                ids.extend(self.encode_ordinary(part))
        
        return ids