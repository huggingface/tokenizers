"""
Encode a batch of text with padding

    python examples/batch_padding.py [path/to/tokenizer.json]

Defaults to the GPT-2 fixture `make test` fetches into `data/`.
"""

import sys
from pathlib import Path

from tokenizers import Padding, Tokenizer

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "gpt2.json")
prompts = ["Hello, I'm a", "The weather today is quite a bit warmer than expected"]


def show(title, tokenizer):
    print(f"== {title}")
    print("padding:", tokenizer.padding)
    for encoding in tokenizer.encode_batch(prompts):
        print(f"  {len(encoding):>2} ids  {encoding.ids}")
        print(f"          mask {encoding.attention_mask}")


left = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>")
tokenizer = Tokenizer.from_file(path, padding=left)
show("left, to the longest in the batch", tokenizer)

tokenizer.padding = Padding(length=16, pad_id=50256, pad_token="<|endoftext|>")
show("right, to a fixed length of 16", tokenizer)

tokenizer.padding = Padding(pad_to_multiple_of=8, pad_id=50256, pad_token="<|endoftext|>")
show("right, to a multiple of 8", tokenizer)

tokenizer.padding = None
show("switched off", tokenizer)
