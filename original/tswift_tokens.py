#!/usr/bin/env python3
"""
Script to train BasicTokenizer on Taylor Swift text and print merges
"""

from basic_tokenizer import BasicTokenizer
from regex_tokenizer import RegexTokenizer
from test_tokenizer import test_gpt4_tiktoken_equality

def main():
    # Read the Taylor Swift text file
    with open('taylorswift.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Training tokenizer on {len(text)} characters of Taylor Swift text...")
    print(f"Text preview: {text[:200]}...")
    print()
    
    # Create and train the tokenizer
    tokenizer = RegexTokenizer()
    
    # Train with a reasonable vocab size (you can adjust this)
    vocab_size = 1000  # This will create 1000 - 256 = 744 merges
    print(f"Training with vocab_size={vocab_size} (will create {vocab_size - 256} merges)")
    print()
    
    # Train the tokenizer with verbose output to see merges
    tokenizer.train(text, vocab_size, verbose=True)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Print all merges in a nice format
    print(f"\nAll {len(tokenizer.merges)} merges:")
    print("-" * 40)
    
    for i, ((pair0, pair1), merge_idx) in enumerate(tokenizer.merges.items(), 1):
        # Get the byte representation of the merged token
        merged_bytes = tokenizer.vocab[merge_idx]
        # Convert to readable string (with error handling)
        try:
            merged_str = merged_bytes.decode('utf-8')
        except UnicodeDecodeError:
            merged_str = f"<bytes:{merged_bytes.hex()}>"
        
        print(f"Merge {i:3d}: ({pair0:3d}, {pair1:3d}) -> {merge_idx:3d} | '{merged_str}'")
    
    # Test the tokenizer on a sample
    print("\n" + "="*60)
    print("TESTING TOKENIZER")
    print("="*60)
    
    test_text = "Taylor Swift is an American singer-songwriter."
    print(f"Original text: '{test_text}'")
    
    # Encode
    tokens = tokenizer.encode(test_text)
    print(f"Encoded tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Decode
    decoded_text = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded_text}'")
    print(f"Round-trip successful: {test_text == decoded_text}")
    
    # Print the raw dictionaries
    '''
    print("\n" + "="*60)
    print("Just printing the merges dictionary:")
    print("="*60)
    print(tokenizer.merges)
    print("="*60)

    print("\nJust printing the vocab dictionary:")
    print("="*60)
    print(tokenizer.vocab)
    print("="*60)
    '''
    # test equivalence to GPT-4 tokenizer
    

if __name__ == "__main__":
    main()
