from bpe_tokenizer import BasicTokenizer, RegexTokenizer, get_stats, merge


def test_get_stats_counts_pairs():
    assert get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_merge_replaces_pairs():
    assert merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]


def test_basic_tokenizer_round_trip_without_training():
    text = "hello world!!!? (안녕하세요!) lol123 😉"
    tokenizer = BasicTokenizer()
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_basic_tokenizer_round_trip_after_training():
    text = "banana bandana banana"
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=270)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_round_trip_after_training():
    text = "Taylor Swift is an American singer-songwriter. 안녕하세요! 12345"
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=270)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens():
    tokenizer = RegexTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 100257})
    ids = tokenizer.encode("hello<|endoftext|>world", allowed_special="all")
    assert 100257 in ids
    assert tokenizer.decode(ids) == "hello<|endoftext|>world"
