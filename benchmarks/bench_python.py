#!/usr/bin/env python3
import sys
import time
from tokenizers import Tokenizer

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <tokenizer.json> <input.txt>", file=sys.stderr)
        sys.exit(1)
    
    tokenizer_path = sys.argv[1]
    input_path = sys.argv[2]
    
    # Load tokenizer
    load_start = time.time()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    load_time = time.time() - load_start
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Benchmark encoding
    encode_start = time.time()
    encoding = tokenizer.encode(text)
    encode_time = time.time() - encode_start
    
    num_tokens = len(encoding.ids)
    num_chars = len(text)
    
    # Print results in a parseable format
    print(f"load_time_ms:{load_time * 1000:.0f}")
    print(f"encode_time_ms:{encode_time * 1000:.0f}")
    print(f"num_tokens:{num_tokens}")
    print(f"num_chars:{num_chars}")
    print(f"tokens_per_sec:{num_tokens / encode_time:.2f}")

if __name__ == "__main__":
    main()
