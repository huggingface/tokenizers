#!/usr/bin/env python3
"""
Example script demonstrating the RandomChunkSplit pre-tokenizer for training 
BPE models that can learn tokens across whitespace boundaries.
"""

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import RandomChunkSplit, WhitespaceSplit

def main():
    # Create a simple corpus with multi-word expressions
    corpus = [
        "Machine learning is a subfield of artificial intelligence.",
        "New York City is the most populous city in the United States.",
        "Natural language processing is a field of computer science.",
        "Deep learning models require large amounts of data.",
        "In machine learning, models learn from examples.",
        "The United States of America is a federal republic.",
        "Artificial intelligence systems can recognize patterns.",
        "New York Times is a well-known newspaper.",
        "We use deep learning for image recognition."
    ]
    
    # Example 1: BPE with traditional whitespace splitting
    print("\n=== Training BPE with WhitespaceSplit ===")
    whitespace_tokenizer = Tokenizer(BPE())
    whitespace_tokenizer.pre_tokenizer = WhitespaceSplit()
    
    # Train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=100,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    whitespace_tokenizer.train_from_iterator(corpus, trainer)
    
    # Encode example text
    example = "Machine learning helps in New York."
    encoded = whitespace_tokenizer.encode(example)
    print(f"Original: {example}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Number of tokens: {len(encoded.tokens)}")
    
    # Example 2: BPE with RandomChunkSplit pre-tokenizer
    print("\n=== Training BPE with RandomChunkSplit ===")
    random_chunk_tokenizer = Tokenizer(BPE())
    random_chunk_tokenizer.pre_tokenizer = RandomChunkSplit(min_length=2, max_length=5)
    
    # Train the tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=100,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    random_chunk_tokenizer.train_from_iterator(corpus, trainer)
    
    # Encode example text
    encoded = random_chunk_tokenizer.encode(example)
    print(f"Original: {example}")
    print(f"Tokens: {encoded.tokens}")
    print(f"Number of tokens: {len(encoded.tokens)}")
    
    # Example 3: Analyzing vocabulary differences
    print("\n=== Vocabulary Analysis ===")
    
    # Get vocabularies
    ws_vocab = set(whitespace_tokenizer.get_vocab().keys())
    rc_vocab = set(random_chunk_tokenizer.get_vocab().keys())
    
    # Count multi-word expressions in vocabulary
    def contains_whitespace(token):
        return ' ' in token
    
    ws_multiword = [token for token in ws_vocab if contains_whitespace(token)]
    rc_multiword = [token for token in rc_vocab if contains_whitespace(token)]
    
    print(f"WhitespaceSplit vocabulary size: {len(ws_vocab)}")
    print(f"RandomChunkSplit vocabulary size: {len(rc_vocab)}")
    print(f"Multiword tokens in WhitespaceSplit vocab: {len(ws_multiword)} ({len(ws_multiword)/len(ws_vocab)*100:.1f}%)")
    print(f"Multiword tokens in RandomChunkSplit vocab: {len(rc_multiword)} ({len(rc_multiword)/len(rc_vocab)*100:.1f}%)")
    
    print("\nMultiword tokens in RandomChunkSplit vocabulary:")
    for token in sorted(rc_multiword)[:10]:  # Show up to 10 examples
        print(f"  '{token}'")

if __name__ == "__main__":
    main()