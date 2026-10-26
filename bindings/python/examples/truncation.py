"""
Encode text with truncation

    python examples/truncation.py [path/to/tokenizer.json]

Defaults to the GPT-2 fixture `make test` fetches into `data/`.
"""

import sys
from pathlib import Path

from tokenizers import TokenizerSpec, Truncation

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "gpt2.json")
prompt = "The weather today is quite a bit warmer than expected"


def show(title, spec):
    print(f"== {title}")
    print("truncation:", spec.truncation)
    encoding = spec.compile().encode(prompt)
    print(f"  {len(encoding):>2} ids  {encoding.ids}")


spec = TokenizerSpec.from_file(path)
spec.truncation = Truncation(6)
show("right, to 6 tokens", spec)

spec.truncation = Truncation(6, direction="left")
show("left, to 6 tokens", spec)

spec.truncation = None
show("switched off", spec)

print("== right, to 3 tokens, for this call only")
print(f"   3 ids  {spec.compile().encode(prompt, truncation=Truncation(3)).ids}")
