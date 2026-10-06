"""
Encode text with truncation

    python examples/truncation.py [path/to/tokenizer.json]

Defaults to the GPT-2 fixture `make test` fetches into `data/`.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer, Truncation

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "gpt2.json")
prompt = "The weather today is quite a bit warmer than expected"


def show(title, tokenizer):
    print(f"== {title}")
    print("truncation:", tokenizer.truncation)
    encoding = tokenizer.encode(prompt)
    print(f"  {len(encoding):>2} ids  {encoding.ids}")


tokenizer = Tokenizer.from_file(path)
tokenizer.truncation = Truncation(6)
show("right, to 6 tokens", tokenizer)

tokenizer.truncation = Truncation(6, direction="left")
show("left, to 6 tokens", tokenizer)

tokenizer.truncation = None
show("switched off", tokenizer)

print("== right, to 3 tokens, for this call only")
print(f"   3 ids  {tokenizer.encode(prompt, truncation=Truncation(3)).ids}")
print("truncation, still:", tokenizer.truncation)
