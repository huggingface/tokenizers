"""
Build a tokenizer from parts: read, share, chain and edit its components

    python examples/composition.py

Runs on the GPT-2 and BERT fixtures.
Run `make test` to fetch them into `data/`.
A `Tokenizer` holds the components and can be edited; `compile()` turns it into the `CompiledTokenizer` that encodes.
"""

import copy
from pathlib import Path

from tokenizers import Tokenizer, normalizers

data = Path(__file__).parent.parent / "data"
text = "Hello World"


def show(title, tokenizer):
    print(f"{title:<28} {tokenizer.compile().tokenize(text)}")


gpt2 = Tokenizer.from_file(data / "gpt2.json")
bert = Tokenizer.from_file(data / "bert-wiki.json")
print("gpt2 model:     ", gpt2.model)
print("gpt2 normalizer:", gpt2.normalizer)
print("bert normalizer:", bert.normalizer)
show("gpt2", gpt2)

# The getters are typed with the base class. `isinstance` narrows to the concrete one.
assert isinstance(bert.normalizer, normalizers.Sequence)
sequence = normalizers.Sequence([*bert.normalizer, normalizers.Prepend("Hey ")])
gpt2.normalizer = sequence
compiled = gpt2.compile()
show("gpt2, bert normalizer + Hey", gpt2)

prepend = sequence[-1]
assert isinstance(prepend, normalizers.Prepend)
prepend.prepend = "Hi "
show("after the edit", gpt2)
print(f"{'compiled before the edit':<28} {compiled.tokenize(text)}")

twin = copy.deepcopy(gpt2)
twin.normalizer = None
show("copy, normalizer removed", twin)
show("original", gpt2)
