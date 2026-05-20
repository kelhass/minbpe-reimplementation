# minimal BPE tokenizer
# algorithmically follows GPT-2 tokenizer
# does NOT handle regular expression splitting pattern or special tokens

from minbpe_base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input pre-processing (he has this as 2 steps)
        tokens = list(text.encode("utf-8")) # list of ints in range 0-255

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> byte
        for i in range(num_merges):
            stats = get_stats(tokens)
            pair = max(stats, key=stats.get)
            idx = 256 + i # new token to replace max pair with
            tokens = merge(tokens, pair, idx) # replace pair with new merge token
            merges[pair] = idx # adds the merge to merge dict
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        if verbose:
            print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        self.merges = merges # used in encode()
        self.vocab = vocab # used in decode()
        
    def encode(self, text):
        tokens = list(text.encode("utf-8")) # list of ints in range 0-255
        while len(tokens) >= 2: # fixes case if len(tokens) = 1
            stats = get_stats(tokens) # note: key = byte pair, value = num occurrences

            # identifies the min/earliest merge from all pairs in tokens
            # iterates over the byte pairs (keys of stats), returns the index of the corresponding merge if it exists (inf if not)
            pair = min(stats, key=lambda p:self.merges.get(p,float("inf")))

            if pair not in self.merges: # happens when all pairs return as inf
                break

            idx = self.merges[pair] # idx becomes the merge token (value of merges dict) for the identfiied pair
            tokens = merge(tokens, pair, idx)
        return tokens
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids) # look up bytes for each token in ids and join them together
        text = text_bytes.decode("utf-8", errors="replace") # not all bytes are valid
        return text