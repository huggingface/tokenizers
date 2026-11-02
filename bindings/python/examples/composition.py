"""
Build a tokenizer from parts: create, share, chain and edit its components

    python examples/composition.py

A `Tokenizer` holds the components and can be edited; `compile()` turns it into the `CompiledTokenizer` that encodes.
"""

from tokenizers import Tokenizer, models, normalizers

bpe = models.BPE(
    vocab={"[UNK]": 0, "_": 1, "h": 2, "u": 3, "g": 4, "s": 5, "hu": 6, "hug": 7, "hugs": 8, "ug": 9},
    merges=[("h", "u"), ("hu", "g"), ("hug", "s")],
    unk_token="[UNK]",
)
tokenizer = Tokenizer(bpe)
tokenizer.normalizer = normalizers.Sequence([normalizers.Prepend("_")])

bpe.ignore_merges = True

compiled = tokenizer.compile()

encoding = compiled.encode("hugs")
print(encoding.ids)
