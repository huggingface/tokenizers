"""
Encode a pair of texts, as a BERT-like model sees a question and its context.

    python examples/encode_pair.py [path/to/tokenizer.json]

Defaults to the BERT fixture `make test` fetches into `data/`.
"""

import sys
from pathlib import Path

from tokenizers import Tokenizer

path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent / "data" / "bert-wiki.json")
tokenizer = Tokenizer.from_file(path)

pair = ("Where is the Eiffel Tower?", "It stands in Paris.")
encoding = tokenizer.encode(pair)
print("tokens:  ", tokenizer.decode_tokens(encoding.ids))
print("type_ids:", encoding.type_ids)

encodings = tokenizer.encode_batch([pair, "A single text, in the same batch."])
for idx, encoding in enumerate(encodings):
    print(f"batch[{idx}]:  IDS=      ", encoding.ids)
    print(f"batch[{idx}]:  TYPE_IDS= ", encoding.type_ids)
    print(f"batch[{idx}]:  TOKENS=   ", tokenizer.decode_tokens(encoding.ids))
    
