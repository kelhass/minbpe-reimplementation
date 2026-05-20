import pytest
import tiktoken
import os

from basic_tokenizer import BasicTokenizer

test_strings = [
    "", # empty string
    "?", # single char
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as special string
]

def unpack(text):
    
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__)) # finds directory of the file
        taylorswift_file = os.path.join(dirname, text[5:]) # joins directory with the file name
        contents = open(taylorswift_file, "r", encoding="utf-8").read() # reads the file
        return contents
    else:
        return text # if not a FILE: string, return the text as is
    
specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
special_tokens = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}
llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()
    
# ------------------------------------------------------------
# tests

# test encode/decode identity
@pytest.mark.parameterize("tokenizer_factory", [BasicTokenizer]) # will need to add others later
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text) # checks if text is a FILE: string and unpacks it
    tokenizer = tokenizer_factory() # calls specified tokenizer
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded # checks if the decoded text is the same as the original text
    #print(text == decoded)

# test that tokenizer matches official GPT-4 tokenizer
@pytest.mark.parameterize("text", test_strings)
def test_gpt4_tiktoken_equality(text):
    text = unpack(text) # checks if text is a FILE: string and unpacks it
    tokenizer = GPT4Tokenizer() # calls GPT-4 tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    tiktoken_ids = enc.encode(text)
    gpt4_tokenizer_ids = tokenizer.encode(text)
    assert gpt4_tokenizer_ids == tiktoken_ids
    #print(gpt4_tokenizer_ids == tiktoken_ids)

