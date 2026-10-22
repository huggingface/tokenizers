"""
Read and edit a tokenizer's components, the way transformers does on load and generation.

    python examples/composition.py

Uses the GPT-2 and Llama 3 fixtures `make test` fetches into `data/`.
"""

import copy
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE

data = Path(__file__).parent.parent / "data"
gpt2 = Tokenizer.from_file(str(data / "gpt2.json"))
llama = Tokenizer.from_file(str(data / "llama-3-tokenizer.json"))
prompt = "Hello world"

print("type(gpt2.model).__name__:", type(gpt2.model).__name__)
print("isinstance(gpt2.model, BPE)", isinstance(gpt2.model, BPE))

if isinstance(gpt2.model, BPE):
    gpt2.model.ignore_merges = True

copied = copy.deepcopy(llama)
copied.post_processor = None
print(copied.encode(prompt).ids)
print(llama.encode(prompt).ids)

copied.post_processor = llama.post_processor
print(copied.encode(prompt).ids)
print(llama.encode(prompt).ids)

if (decoder := llama.decoder) is not None:
    print("decoder type:", json.loads(decoder.__getstate__())["type"])
