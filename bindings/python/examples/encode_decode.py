"""
Load a tokenizer.json, encode one text, look at the encoding, decode it back.

    python examples/encode_decode.py [path/to/tokenizer.json]

Defaults to the Llama 3 fixture `make test` fetches into `data/`.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "llama-3-tokenizer.json")
tokenizer = Tokenizer.from_file(path)

text = "Hello there, how are you?"
encoding = tokenizer.encode(text)
print(f"{len(encoding)} tokens")
print("ids:            ", encoding.ids)
print("type_ids:       ", encoding.type_ids)
print("attention_mask: ", encoding.attention_mask)
print("ids, as a numpy array:", encoding.ids_array)

without_specials = tokenizer.encode(text, add_special_tokens=False)
print("without special tokens:", without_specials.ids)

print("decoded:", tokenizer.decode(encoding.ids))
print("decoded, special tokens kept:", tokenizer.decode(encoding.ids, skip_special_tokens=False))

print("tokens:                       ", tokenizer.tokenize(text))
print("tokens (add specials = False):", tokenizer.tokenize(text, add_special_tokens=False))
print("tokens from the ids:          ", tokenizer.decode_tokens(encoding.ids))
print("tokens (skip specials = True):", tokenizer.decode_tokens(encoding.ids, skip_special_tokens=True))
